import os
import sys
import types
import math
import cv2
import gc
import numpy as np
import urllib.parse
import boto3
import runpod
import torch

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

# Ensuring torch.distributed.device_mesh is imported to populates the attribute on torch.distributed
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

from diffusers import ControlNetModel, FluxPipeline, FluxImg2ImgPipeline
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
from PIL import Image
from insightface.app import FaceAnalysis

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

# Initialize boto3 S3 client
s3_client = boto3.client("s3")

# Pipeline and FaceAnalysis warm cache
pipe = None
app = None
current_engine = None  # Tracks which engine is loaded: None, "sdxl", or "flux"
current_flux_class = None  # Tracks loaded Flux pipeline class: None, FluxPipeline, or FluxImg2ImgPipeline

def ensure_models_downloaded():
    from huggingface_hub import hf_hub_download
    
    os.makedirs(os.path.join(CACHE_ROOT, "huggingface/models"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_ROOT, "insightface/models/antelopev2"), exist_ok=True)
    
    hf_token = os.environ.get("HF_TOKEN")
    
    # 1. Download IP-Adapter weights if not present
    ip_adapter_path = os.path.join(CACHE_ROOT, "huggingface/models/ip-adapter.bin")
    if not os.path.exists(ip_adapter_path):
        print(f"ip-adapter.bin not found. Downloading dynamically to {ip_adapter_path}...")
        hf_hub_download(
            repo_id="InstantX/InstantID",
            filename="ip-adapter.bin",
            local_dir=os.path.join(CACHE_ROOT, "huggingface/models"),
            token=hf_token
        )
        print("ip-adapter.bin downloaded successfully.")
        
    # 2. Download InsightFace antelopev2 models if not present
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
            print(f"InsightFace model {f} not found. Downloading dynamically to {path}...")
            hf_hub_download(
                repo_id="DIAMONIK7777/antelopev2",
                filename=f,
                local_dir=os.path.join(CACHE_ROOT, "insightface/models/antelopev2"),
                token=hf_token
            )
            print(f"{f} downloaded successfully.")

def unload_pipeline():
    global pipe, current_engine, current_flux_class
    if pipe is not None:
        print(f"Unloading current pipeline ({current_engine}) from VRAM...")
        try:
            if hasattr(pipe, "to"):
                pipe.to("cpu")
        except Exception as e:
            print(f"Error offloading pipeline to CPU: {e}")
        pipe = None
        current_engine = None
        current_flux_class = None
        
        # Explicit garbage collection and CUDA cache clearing
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("VRAM cleared successfully.")

def get_sdxl_pipeline():
    global pipe, current_engine, current_flux_class
    if current_engine == "sdxl" and pipe is not None:
        return pipe
        
    if current_engine is not None:
        unload_pipeline()
        
    print("Loading InstantID ControlNet and SDXL pipeline...")
    ensure_models_downloaded()
    
    # 1. Load the ControlNet model (IdentityNet)
    controlnet = ControlNetModel.from_pretrained(
        "InstantX/InstantID",
        subfolder="ControlNetModel",
        torch_dtype=torch.float16,
        local_files_only=False,
        cache_dir=os.path.join(CACHE_ROOT, "huggingface"),
        token=os.environ.get("HF_TOKEN")
    )
    
    base_model = os.environ.get("BASE_MODEL", "RunDiffusion/Juggernaut-XL-v9")
    print(f"Loading base SDXL model: {base_model}")
    
    # 2. Load the base SDXL pipeline
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        local_files_only=False,
        cache_dir=os.path.join(CACHE_ROOT, "huggingface"),
        token=os.environ.get("HF_TOKEN")
    )
    
    # 3. Load IP-Adapter weights
    pipe.load_ip_adapter_instantid(os.path.join(CACHE_ROOT, "huggingface/models/ip-adapter.bin"))
    
    # Move pipeline to CUDA
    pipe = pipe.to("cuda")
    
    # Explicitly move image_proj_model to CUDA
    if hasattr(pipe, "image_proj_model") and pipe.image_proj_model is not None:
        pipe.image_proj_model = pipe.image_proj_model.to(device="cuda", dtype=torch.float16)
        
    # Performance tuning
    pipe.enable_attention_slicing()
    
    current_engine = "sdxl"
    current_flux_class = None
    print("SDXL InstantID Pipeline loaded successfully!")
    return pipe

def get_flux_pipeline(target_class):
    global pipe, current_engine, current_flux_class
    if current_engine == "flux" and current_flux_class == target_class and pipe is not None:
        return pipe
        
    if current_engine is not None:
        unload_pipeline()
        
    flux_repo = os.environ.get("FLUX_REPO", "black-forest-labs/FLUX.2-dev")
    print(f"Loading FLUX pipeline ({target_class.__name__}) from {flux_repo}...")
    
    hf_token = os.environ.get("HF_TOKEN")
    
    pipe = target_class.from_pretrained(
        flux_repo,
        torch_dtype=torch.bfloat16,
        cache_dir=os.path.join(CACHE_ROOT, "huggingface"),
        token=hf_token
    )
    
    # Enable CPU offloading as recommended for 24GB GPUs running Flux
    pipe.enable_model_cpu_offload()
    
    current_engine = "flux"
    current_flux_class = target_class
    print(f"FLUX Pipeline ({target_class.__name__}) loaded successfully!")
    return pipe

def get_face_analysis():
    global app
    if app is None:
        print("Loading FaceAnalysis antelopev2...")
        ensure_models_downloaded()
        app = FaceAnalysis(
            name='antelopev2',
            root=os.path.join(CACHE_ROOT, "insightface"),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("FaceAnalysis loaded successfully!")
    return app

def draw_kps(image_pil, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)
    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])
    
    # Draw limbs as polygons
    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]
        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img, polygon, color)
    
    out_img = (out_img * 0.6).astype(np.uint8)
    
    # Draw keypoints as circles
    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img, (int(x), int(y)), 10, color, -1)
        
    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

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
    prompt = job_input.get("prompt", "handsome young man, showcasing a modern fade haircut, photorealistic, 8k, professional headshot, same person, same face, same identity, same eyes, same nose, same facial structure, preserve identity")
    negative_prompt = job_input.get("negative_prompt", "deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limbs, ugly, blurry, monochrome, long hair, bald, different person, different face, beautified face, altered facial features, plastic surgery, different eyes, different nose")
    instantid_strength = float(job_input.get("instantid_strength", 1.1))
    guidance_scale = float(job_input.get("guidance_scale", 4.0))
    num_inference_steps = int(job_input.get("num_inference_steps", 28))

    # Engine Selection & Routing
    engine = job_input.get("engine")
    if not engine:
        # Prompt-based auto-switching
        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in ["flux", "dev", "[flux]", "flux.2", "flux.1"]):
            engine = "flux"
        else:
            engine = "sdxl"

    print(f"Routing inference job to engine: {engine.upper()}")

    if not output_bucket_uri:
        return {"error": "Missing output_bucket in job payload"}

    local_input = "/tmp/input.jpg"
    local_output = "/tmp/output.jpg"

    try:
        if engine == "flux":
            # ----------------------------------------------------
            # FLUX.2 Inference Flow
            # ----------------------------------------------------
            seed = int(job_input.get("seed", 42))
            generator = torch.Generator().manual_seed(seed)
            flux_guidance = float(job_input.get("guidance_scale", 3.5))  # FLUX works best with lower CFG (3.0 - 5.0)
            flux_steps = int(job_input.get("num_inference_steps", 28))

            if input_image_uri:
                # Image-to-Image (Hair simulation/edit)
                download_from_s3(input_image_uri, local_input)
                init_image = Image.open(local_input).convert("RGB")

                # Flux img2img strength determines how much to modify the input image (hair editing)
                # Defaults to 0.6 if not provided.
                flux_strength = float(job_input.get("flux_strength", job_input.get("strength", 0.6)))
                print(f"Running FLUX Img2Img. Prompt: '{prompt}', strength: {flux_strength}, guidance: {flux_guidance}, steps: {flux_steps}")

                pipeline = get_flux_pipeline(FluxImg2ImgPipeline)

                with torch.inference_mode():
                    output_img = pipeline(
                        prompt=prompt,
                        image=init_image,
                        strength=flux_strength,
                        guidance_scale=flux_guidance,
                        num_inference_steps=flux_steps,
                        generator=generator
                    ).images[0]
            else:
                # Text-to-Image (New generation)
                print(f"Running FLUX Text2Img. Prompt: '{prompt}', guidance: {flux_guidance}, steps: {flux_steps}")
                pipeline = get_flux_pipeline(FluxPipeline)

                with torch.inference_mode():
                    output_img = pipeline(
                        prompt=prompt,
                        guidance_scale=flux_guidance,
                        num_inference_steps=flux_steps,
                        generator=generator
                    ).images[0]

        else:
            # ----------------------------------------------------
            # SDXL InstantID Inference Flow
            # ----------------------------------------------------
            if not input_image_uri:
                return {"error": "SDXL InstantID engine requires an input_image selfie."}

            download_from_s3(input_image_uri, local_input)
            init_image = Image.open(local_input).convert("RGB")

            # Extract facial embeddings and landmarks using InsightFace
            face_img = cv2.imread(local_input)
            face_analysis = get_face_analysis()
            face_info = face_analysis.get(face_img)

            if len(face_info) == 0:
                return {"error": "No face detected in the input image. Please provide a clear portrait selfie."}

            # Get the largest face if multiple are found
            face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]

            face_emb = face_info['embedding']
            face_kps = face_info['kps']

            # Draw keypoints for ControlNet conditioning
            kps_image = draw_kps(init_image, face_kps)

            print(f"Running SDXL InstantID. Prompt: '{prompt}', strength: {instantid_strength}, guidance: {guidance_scale}, steps: {num_inference_steps}")
            pipeline = get_sdxl_pipeline()

            generator = torch.Generator("cuda").manual_seed(42)
            face_emb_tensor = torch.from_numpy(face_emb).unsqueeze(0).to(device="cuda", dtype=torch.float16)

            with torch.inference_mode():
                output_img = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image_embeds=face_emb_tensor,
                    image=kps_image,
                    controlnet_conditioning_scale=instantid_strength,
                    ip_adapter_scale=instantid_strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator
                ).images[0]

        # 5. Save generated output
        output_img.save(local_output, "JPEG")
        print(f"Generation successful. Saved output to {local_output}")

        # 6. Upload generated output back to temporal S3 bucket
        upload_to_s3(local_output, output_bucket_uri)

        # Cleanup temp files
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
