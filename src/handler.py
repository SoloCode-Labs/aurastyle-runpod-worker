import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
import sys
import types
import gc
import urllib.parse
import boto3
import runpod
import torch
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps
from diffusers import FluxKontextInpaintPipeline
from transformers import pipeline

# Monkeypatch torch.xpu dynamically to prevent AttributeError on older PyTorch versions
class MockXPU:
    def __getattr__(self, name):
        def dummy_func(*args, **kwargs):
            if name == "device_count":
                return 0
            if name == "is_available":
                return False
            return None
        return dummy_func

if not hasattr(torch, "xpu"):
    torch.xpu = MockXPU()

# Ensure torch.distributed.device_mesh is imported or mocked
try:
    import torch.distributed.device_mesh
except ImportError:
    device_mesh_mock = types.ModuleType("device_mesh")
    class DeviceMesh:
        pass
    device_mesh_mock.DeviceMesh = DeviceMesh
    if hasattr(torch, "distributed"):
        torch.distributed.device_mesh = device_mesh_mock
    else:
        dist = types.ModuleType("distributed")
        dist.device_mesh = device_mesh_mock
        torch.distributed = dist
    sys.modules["torch.distributed.device_mesh"] = device_mesh_mock

# Configure HuggingFace cache directory (must match target location in Dockerfile)
def get_cache_root():
    if os.path.exists("/runpod-volume"):
        return "/runpod-volume"
    elif os.path.exists("/workspace"):
        return "/workspace"
    else:
        return "/cache"

CACHE_ROOT = get_cache_root()
os.environ["HF_HOME"] = os.path.join(CACHE_ROOT, "huggingface")

import shutil
def log_disk_space():
    for name, path in [("Cache Root", CACHE_ROOT), ("Temp Dir", "/tmp"), ("System Root", "/")]:
        if os.path.exists(path):
            total, used, free = shutil.disk_usage(path)
            print(f"[Disk Space] {name} ({path}): Total={total / (1024**3):.2f} GB, Used={used / (1024**3):.2f} GB, Free={free / (1024**3):.2f} GB")
        else:
            print(f"[Disk Space] {name} ({path}) does not exist.")

log_disk_space()

# Initialize S3 Client
s3_client = boto3.client("s3")

# Pipeline warm caches
pipe = None
segmenter = None
app = None

def ensure_models_downloaded():
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.join(CACHE_ROOT, "insightface/models/antelopev2"), exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")
    
    antelope_files = [
        "1k3d68.onnx",
        "2d106det.onnx",
        "genderage.onnx",
        "glintr100.onnx",
        "scrfd_10g_bnkps.onnx"
    ]
    for f in antelope_files:
        path = os.path.join(CACHE_ROOT, "insightface/models/antelopev2", f)
        if not os.path.exists(path):
            print(f"InsightFace model {f} not found. Downloading dynamically...")
            hf_hub_download(
                repo_id="DIAMONIK7777/antelopev2",
                filename=f,
                local_dir=os.path.join(CACHE_ROOT, "insightface/models/antelopev2"),
                token=hf_token
            )

def get_flux_pipeline():
    global pipe
    if pipe is None:
        print("Loading FLUX.1 Kontext Dev pipeline...")
        flux_repo = os.environ.get("FLUX_KONTEXT_REPO", "black-forest-labs/FLUX.1-Kontext-dev")
        hf_token = os.environ.get("HF_TOKEN")
        
        pipe = FluxKontextInpaintPipeline.from_pretrained(
            flux_repo,
            torch_dtype=torch.bfloat16,
            cache_dir=os.path.join(CACHE_ROOT, "huggingface"),
            token=hf_token
        )
        
        # Configure GPU offloading/loading based on environment
        disable_cpu_offload = os.environ.get("DISABLE_CPU_OFFLOAD", "False").lower() in ("true", "1", "yes")
        if not disable_cpu_offload:
            print("Enabling model CPU offload for VRAM safety...")
            pipe.enable_model_cpu_offload()
        else:
            print("DISABLE_CPU_OFFLOAD is set. Loading model fully to CUDA GPU...")
            pipe = pipe.to("cuda")
            
        print("FLUX.1 Kontext Dev pipeline loaded successfully!")
    return pipe

def get_segmenter():
    global segmenter
    if segmenter is None:
        print("Loading face-parsing segmentation pipeline manually...")
        try:
            from transformers import SegformerImageProcessor as ImageProcessor
        except ImportError:
            from transformers import SegformerFeatureExtractor as ImageProcessor
        from transformers import SegformerForSemanticSegmentation

        model_name = "jonathandinu/face-parsing"
        cache_dir = os.path.join(CACHE_ROOT, "huggingface")

        print("Loading SegFormer image processor...")
        processor = ImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        print("Loading SegFormer model...")
        model = SegformerForSemanticSegmentation.from_pretrained(model_name, cache_dir=cache_dir)

        device = 0 if torch.cuda.is_available() else -1

        segmenter = pipeline(
            "image-segmentation",
            model=model,
            image_processor=processor,
            feature_extractor=processor,
            device=device
        )
        print("Segmentation pipeline loaded successfully!")
    return segmenter

def get_face_analysis():
    global app
    if app is None:
        print("Loading FaceAnalysis antelopev2...")
        ensure_models_downloaded()
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name='antelopev2',
            root=os.path.join(CACHE_ROOT, "insightface"),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("FaceAnalysis loaded successfully!")
    return app

def parse_s3_uri(s3_uri: str):
    parsed = urllib.parse.urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Invalid S3 URI scheme: {s3_uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key

def download_from_s3(s3_uri: str, local_path: str):
    bucket, key = parse_s3_uri(s3_uri)
    print(f"Downloading from S3: bucket={bucket}, key={key} to {local_path}")
    s3_client.download_file(bucket, key, local_path)

def upload_to_s3(local_path: str, s3_uri: str):
    bucket, key = parse_s3_uri(s3_uri)
    print(f"Uploading to S3: {local_path} to bucket={bucket}, key={key}")
    s3_client.upload_file(local_path, bucket, key)

def handler(job):
    job_input = job["input"]
    input_image_uri = job_input.get("input_image")
    output_bucket_uri = job_input.get("output_bucket")
    
    # Model parameters
    prompt = job_input.get("prompt", "Change hairstyle to a realistic classic bob haircut. Preserve facial features exactly. Keep identity unchanged. Natural lighting. Photorealistic.")
    guidance_scale = float(job_input.get("guidance_scale", 2.5))
    num_inference_steps = int(job_input.get("num_inference_steps", 28))
    seed = int(job_input.get("seed", 42))
    
    # Face Preservation settings
    face_preservation = job_input.get("face_preservation", True)
    face_preservation_blur = int(job_input.get("face_preservation_blur", 8))
    face_preservation_overlay = job_input.get("face_preservation_overlay", True)
    preserve_skin = job_input.get("preserve_skin", True)
    
    # Custom preserve labels list (default matches face parsing classes)
    preserve_labels = job_input.get("preserve_labels")
    if preserve_labels is None:
        if preserve_skin:
            preserve_labels = ["skin", "l_brow", "r_brow", "l_eye", "r_eye", "nose", "mouth", "u_lip", "l_lip", "l_ear", "r_ear", "eye_g"]
        else:
            preserve_labels = ["l_brow", "r_brow", "l_eye", "r_eye", "nose", "mouth", "u_lip", "l_lip", "l_ear", "r_ear", "eye_g"]
    elif isinstance(preserve_labels, str):
        preserve_labels = [label.strip() for label in preserve_labels.split(",")]
    preserve_set = set(preserve_labels)
    
    # Validation settings
    validation_check = job_input.get("validation_check", False)

    if not input_image_uri:
        return {"error": "Missing input_image in job payload. FLUX.1 Kontext Dev requires an input selfie image."}
    if not output_bucket_uri:
        return {"error": "Missing output_bucket in job payload."}

    local_input = "/tmp/input.jpg"
    local_output = "/tmp/output.jpg"

    try:
        # 1. Download input selfie from S3
        download_from_s3(input_image_uri, local_input)
        init_image = Image.open(local_input)
        init_image = ImageOps.exif_transpose(init_image).convert("RGB")
        original_size = init_image.size

        # 2. Validation Check (Optional Face Detection)
        if validation_check:
            print("Performing face validation check...")
            face_img = cv2.imread(local_input)
            face_analysis = get_face_analysis()
            face_info = face_analysis.get(face_img)
            if len(face_info) == 0:
                return {"error": "Validation failed: No face detected in the input image. Please supply a clear portrait selfie."}
            print("Face validation passed successfully.")

        # 3. Generate Face/Inpaint Masks
        if face_preservation:
            print("Applying SegFormer-based face preservation...")
            seg_pipeline = get_segmenter()
            
            # SegFormer expects PIL image and returns list of dicts with label and mask PIL image
            segmentation_results = seg_pipeline(init_image)
            
            # First, find the highest pixel of the eyebrows (l_brow, r_brow) to detect the forehead boundary
            eyebrow_y_min = None
            for res in segmentation_results:
                label = res.get("label")
                if label in ("l_brow", "r_brow"):
                    mask_pil = res.get("mask")
                    if mask_pil is not None:
                        import numpy as np
                        mask_arr = np.array(mask_pil)
                        y_indices = np.where(mask_arr > 0)[0]
                        if len(y_indices) > 0:
                            local_min_y = int(np.min(y_indices))
                            if mask_pil.size != original_size:
                                scale_y = original_size[1] / mask_pil.size[1]
                                local_min_y = int(local_min_y * scale_y)
                            if eyebrow_y_min is None or local_min_y < eyebrow_y_min:
                                eyebrow_y_min = local_min_y

            # Fallback if no eyebrows are detected (e.g. assume forehead starts at 35% height)
            if eyebrow_y_min is None:
                eyebrow_y_min = int(original_size[1] * 0.35)
                
            face_mask = Image.new("L", original_size, 0)
            found_labels = []
            
            for res in segmentation_results:
                label = res.get("label")
                mask_pil = res.get("mask")
                if label in preserve_set and mask_pil is not None:
                    found_labels.append(label)
                    mask_pil = mask_pil.convert("L")
                    # Ensure mask size matches original image
                    if mask_pil.size != original_size:
                        mask_pil = mask_pil.resize(original_size, Image.Resampling.NEAREST)
                    
                    # If label is skin, clear the forehead region (above eyebrows) to allow bangs/hair blending
                    if label == "skin":
                        import numpy as np
                        skin_arr = np.array(mask_pil)
                        skin_arr[:eyebrow_y_min, :] = 0
                        mask_pil = Image.fromarray(skin_arr)
                        
                    face_mask = ImageChops.lighter(face_mask, mask_pil)
            
            print(f"Preserving facial components: {found_labels}")
            
            # Feather the edges of the face mask to blend the original face with the new hair smoothly
            face_mask_blurred = face_mask.filter(ImageFilter.GaussianBlur(radius=face_preservation_blur))
            
            # Invert the blurred face mask to get the inpaint mask (white=change, black=preserve)
            inpaint_mask = ImageOps.invert(face_mask_blurred)
        else:
            print("Face preservation is disabled. Generating complete image.")
            face_mask_blurred = None
            inpaint_mask = Image.new("L", original_size, 255)

        # 4. Load Pipeline & Run Inference
        strength = float(job_input.get("strength", 1.0))
        print(f"Running FLUX.1 Kontext Dev Inference. Prompt: '{prompt}', guidance_scale: {guidance_scale}, strength: {strength}, steps: {num_inference_steps}")
        pipeline = get_flux_pipeline()
        
        generator = torch.Generator().manual_seed(seed)
        
        with torch.inference_mode():
            output_img = pipeline(
                image=init_image,
                mask_image=inpaint_mask,
                prompt=prompt,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator
            ).images[0]
            
        print("Generation completed successfully.")

        # Ensure generated output image is resized to match the original image coordinates
        if output_img.size != original_size:
            print(f"Resizing generated image from {output_img.size} to original size {original_size}...")
            output_img = output_img.resize(original_size, Image.Resampling.LANCZOS)

        # 5. Post-inference Face Preservation Overlay (Double Protection)
        if face_preservation and face_preservation_overlay and face_mask_blurred is not None:
            # Composite original face on top of the generated image
            output_img = Image.composite(init_image, output_img, face_mask_blurred)
            print("Post-inference Face Preservation overlay applied.")

        # 6. Save generated output
        output_img.save(local_output, "JPEG")
        print(f"Saved final output image to {local_output}")

        # 7. Upload final result to target S3 bucket
        upload_to_s3(local_output, output_bucket_uri)

        # Cleanup local cache files
        if os.path.exists(local_input):
            os.remove(local_input)
        if os.path.exists(local_output):
            os.remove(local_output)

        return {"status": "COMPLETED"}

    except Exception as e:
        import traceback
        error_msg = f"Inference execution failed: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {"error": error_msg}

# Start RunPod serverless service
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
