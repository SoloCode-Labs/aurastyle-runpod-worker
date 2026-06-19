import os
import sys
import types
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

from huggingface_hub import hf_hub_download
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

# Configure cache directories to match container mount paths
os.environ["HF_HOME"] = "/cache/huggingface"

print("Step 1: Downloading InstantID ControlNet Model...")
controlnet = ControlNetModel.from_pretrained(
    "InstantX/InstantID",
    subfolder="ControlNetModel",
    torch_dtype=torch.float16,
    cache_dir="/cache/huggingface"
)

print("Step 2: Downloading RealVisXL_V4.0 Base SDXL Model...")
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    cache_dir="/cache/huggingface"
)

print("Step 3: Downloading InstantID IP-Adapter weights...")
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ip-adapter.bin",
    local_dir="/cache/huggingface/models"
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
        local_dir="/cache/insightface/models/antelopev2"
    )

print("Model caching completed successfully!")

