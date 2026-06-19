import os
import sys
import types
import math
import cv2
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

from diffusers import ControlNetModel
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

def ensure_models_downloaded():
    from huggingface_hub import hf_hub_download
    
    os.makedirs(os.path.join(CACHE_ROOT, "huggingface/models"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_ROOT, "insightface/models/antelopev2"), exist_ok=True)
    
    # 1. Download IP-Adapter weights if not present
    ip_adapter_path = os.path.join(CACHE_ROOT, "huggingface/models/ip-adapter.bin")
    if not os.path.exists(ip_adapter_path):
        print(f"ip-adapter.bin not found. Downloading dynamically to {ip_adapter_path}...")
        hf_hub_download(
            repo_id="InstantX/InstantID",
            filename="ip-adapter.bin",
            local_dir=os.path.join(CACHE_ROOT, "huggingface/models")
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
                local_dir=os.path.join(CACHE_ROOT, "insightface/models/antelopev2")
            )
            print(f"{f} downloaded successfully.")

def get_pipeline():
    global pipe
    if pipe is None:
        print("Loading InstantID ControlNet and SDXL pipeline...")
        # Ensure supplementary files are downloaded
        ensure_models_downloaded()
        
        # 1. Load the ControlNet model (IdentityNet)
        controlnet = ControlNetModel.from_pretrained(
            "InstantX/InstantID",
            subfolder="ControlNetModel",
            torch_dtype=torch.float16,
            local_files_only=False,
            cache_dir=os.path.join(CACHE_ROOT, "huggingface")
        )
        
        base_model = os.environ.get("BASE_MODEL", "SG161222/RealVisXL_V4.0")
        print(f"Loading base SDXL model: {base_model}")
        
        # 2. Load the base SDXL pipeline
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            local_files_only=False,
            cache_dir=os.path.join(CACHE_ROOT, "huggingface")
        )
        
        # 3. Load IP-Adapter weights
        pipe.load_ip_adapter_instantid(os.path.join(CACHE_ROOT, "huggingface/models/ip-adapter.bin"))
        
        # Move pipeline to CUDA
        pipe = pipe.to("cuda")
        
        # Performance tuning
        pipe.enable_attention_slicing()
        
        print("Pipeline loaded successfully!")
    return pipe

def get_face_analysis():
    global app
    if app is None:
        print("Loading FaceAnalysis antelopev2...")
        # Ensure supplementary files are downloaded
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
    prompt = job_input.get("prompt", "handsome young man, showcasing a modern fade haircut, photorealistic, 8k, professional headshot")
    negative_prompt = job_input.get("negative_prompt", "deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limbs, ugly, blurry, monochrome, long hair, bald")
    strength = float(job_input.get("instantid_strength", 0.8))

    if not input_image_uri or not output_bucket_uri:
        return {"error": "Missing input_image or output_bucket in job payload"}

    local_input = "/tmp/input.jpg"
    local_output = "/tmp/output.jpg"

    try:
        # 1. Download input selfie from S3
        download_from_s3(input_image_uri, local_input)

        # 2. Extract facial embeddings and landmarks using InsightFace
        init_image = Image.open(local_input).convert("RGB")
        
        # Load OpenCV BGR image for InsightFace
        face_img = cv2.imread(local_input)
        
        face_analysis = get_face_analysis()
        face_info = face_analysis.get(face_img)
        
        if len(face_info) == 0:
            return {"error": "No face detected in the input image. Please provide a clear portrait selfie."}
            
        # Get the largest face if multiple are found
        face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
        
        face_emb = face_info['embedding']
        face_kps = face_info['kps']

        # 3. Draw keypoints for ControlNet conditioning
        kps_image = draw_kps(init_image, face_kps)

        # 4. Run InstantID Inference
        print(f"Running InstantID inference. Prompt: '{prompt}', strength: {strength}")
        pipeline = get_pipeline()
        
        # Use a fixed generator for reproducibility
        generator = torch.Generator("cuda").manual_seed(42)
        
        # Convert face embedding to float16 tensor on GPU
        face_emb_tensor = torch.from_numpy(face_emb).unsqueeze(0).to(device="cuda", dtype=torch.float16)

        with torch.inference_mode():
            output_img = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_embeds=face_emb_tensor,
                image=kps_image,
                controlnet_conditioning_scale=strength,
                ip_adapter_scale=strength,
                guidance_scale=5.0,
                num_inference_steps=30,
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
