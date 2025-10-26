import sys

try:
    import torch
except Exception as e:
    print("torch_import_error:", repr(e))
    sys.exit(1)

print("torch_version:", torch.__version__)
print("python:", sys.version.split()[0])
print("cuda_available:", torch.cuda.is_available())
mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
print("mps_available:", mps_available)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("device:", device, torch.cuda.get_device_name(0))
elif mps_available:
    device = torch.device("mps")
    print("device:", device)
else:
    device = torch.device("cpu")
    print("device:", device)

x = torch.randn(1024, 1024, device=device)
y = torch.randn(1024, 1024, device=device)
z = x @ y
print("compute_ok:", z.shape, z.dtype, z.device)


