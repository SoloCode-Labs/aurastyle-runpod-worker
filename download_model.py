import os
from huggingface_hub import snapshot_download, hf_hub_download

# Configure cache directories dynamically to match container mount paths
def get_cache_root():
    if os.path.exists("/runpod-volume"):
        return "/runpod-volume"
    elif os.path.exists("/workspace"):
        return "/workspace"
    else:
        return "/cache"

CACHE_ROOT = get_cache_root()
HF_CACHE = os.path.join(CACHE_ROOT, "huggingface")

# Set cache environment variables to ensure Hugging Face Hub clients write to the correct directory
os.environ["HF_HOME"] = HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

hf_token = os.environ.get("HF_TOKEN")
flux_repo = os.environ.get("FLUX_REPO", "black-forest-labs/FLUX.2-dev")

print(f"Hugging Face cache root resolved to: {HF_CACHE}")
if hf_token:
    print("Hugging Face authentication token found in environment.")
else:
    print("Warning: HF_TOKEN environment variable not set. Downloading gated models like FLUX.2-dev might fail.")

print("Step 1: Downloading InstantID ControlNet Model...")
snapshot_download(
    repo_id="InstantX/InstantID",
    allow_patterns=["ControlNetModel/*"],
    cache_dir=HF_CACHE,
    token=hf_token
)

print("Step 2: Downloading Juggernaut-XL-v9 Base SDXL Model...")
snapshot_download(
    repo_id="RunDiffusion/Juggernaut-XL-v9",
    cache_dir=HF_CACHE,
    token=hf_token
)

print("Step 3: Downloading InstantID IP-Adapter weights...")
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ip-adapter.bin",
    local_dir=os.path.join(CACHE_ROOT, "huggingface/models"),
    token=hf_token
)

print("Step 4: Downloading Antelopev2 models for InsightFace...")
antelope_files = [
    "1k3d68.onnx",
    "2d106det.onnx",
    "genderage.onnx",
    "glintr100.onnx",
    "scrfd_10g_bnkps.onnx"
]
for f in antelope_files:
    print(f"Downloading {f}...")
    hf_hub_download(
        repo_id="DIAMONIK7777/antelopev2",
        filename=f,
        local_dir=os.path.join(CACHE_ROOT, "insightface/models/antelopev2"),
        token=hf_token
    )

print(f"Step 5: Downloading FLUX model {flux_repo}...")
try:
    snapshot_download(
        repo_id=flux_repo,
        cache_dir=HF_CACHE,
        token=hf_token
    )
    print("FLUX model downloaded successfully!")
except Exception as e:
    print(f"Warning: Could not download FLUX model '{flux_repo}'. Error: {e}")
    print("Please make sure you have accepted the license terms on Hugging Face and provided a valid HF_TOKEN.")

print("Model caching completed successfully!")

