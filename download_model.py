import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
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

import shutil
def log_disk_space():
    for name, path in [("Cache Root", CACHE_ROOT), ("Temp Dir", "/tmp"), ("System Root", "/")]:
        if os.path.exists(path):
            total, used, free = shutil.disk_usage(path)
            print(f"[Disk Space] {name} ({path}): Total={total / (1024**3):.2f} GB, Used={used / (1024**3):.2f} GB, Free={free / (1024**3):.2f} GB")
        else:
            print(f"[Disk Space] {name} ({path}) does not exist.")

log_disk_space()

hf_token = os.environ.get("HF_TOKEN")
flux_kontext_repo = os.environ.get("FLUX_KONTEXT_REPO", "black-forest-labs/FLUX.1-Kontext-dev")

print(f"Hugging Face cache root resolved to: {HF_CACHE}")
if hf_token:
    print("Hugging Face authentication token found in environment.")
else:
    print("Warning: HF_TOKEN environment variable not set. Downloading gated models like FLUX.1-Kontext-dev will fail.")

print("Step 1: Downloading Antelopev2 models for InsightFace (optional validation)...")
antelope_files = [
    "1k3d68.onnx",
    "2d106det.onnx",
    "genderage.onnx",
    "glintr100.onnx",
    "scrfd_10g_bnkps.onnx"
]
for f in antelope_files:
    print(f"Downloading {f}...")
    try:
        hf_hub_download(
            repo_id="DIAMONIK7777/antelopev2",
            filename=f,
            local_dir=os.path.join(CACHE_ROOT, "insightface/models/antelopev2"),
            token=hf_token
        )
    except Exception as e:
        print(f"Error downloading {f}: {e}")

print("Step 2: Downloading jonathandinu/face-parsing for Face Preservation Segmentation...")
try:
    snapshot_download(
        repo_id="jonathandinu/face-parsing",
        cache_dir=HF_CACHE,
        token=hf_token
    )
    print("Face parsing segmenter model downloaded successfully!")
except Exception as e:
    print(f"Error downloading face parsing model: {e}")

print(f"Step 3: Downloading FLUX.1 Kontext Dev model ({flux_kontext_repo})...")
try:
    snapshot_download(
        repo_id=flux_kontext_repo,
        cache_dir=HF_CACHE,
        token=hf_token
    )
    print("FLUX.1 Kontext Dev model downloaded successfully!")
except Exception as e:
    print(f"Warning: Could not download FLUX.1 Kontext Dev model '{flux_kontext_repo}'. Error: {e}")
    print("Please make sure you have accepted the license terms on Hugging Face and provided a valid HF_TOKEN.")

print("Model caching completed successfully!")
