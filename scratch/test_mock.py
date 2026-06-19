import sys
import torch
import types

try:
    import torch.distributed as dist
except ImportError:
    dist = types.ModuleType("distributed")
    torch.distributed = dist

if not hasattr(dist, "device_mesh"):
    device_mesh = types.ModuleType("device_mesh")
    class DeviceMesh:
        pass
    device_mesh.DeviceMesh = DeviceMesh
    dist.device_mesh = device_mesh
    sys.modules["torch.distributed.device_mesh"] = device_mesh

print("Mocking successful!")
print("dist.device_mesh:", dist.device_mesh)
print("dist.device_mesh.DeviceMesh:", dist.device_mesh.DeviceMesh)
