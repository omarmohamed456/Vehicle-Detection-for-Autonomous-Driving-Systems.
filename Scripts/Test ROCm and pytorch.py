import torch
print("Torch version:", torch.__version__)
print("HIP (ROCm) version:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")




# save as /tmp/rocm_test.py
import torch, time
print("torch:", torch.__version__, "hip:", torch.version.hip)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", dev)
a = torch.randn((4096, 4096), device=dev)
b = torch.randn((4096, 4096), device=dev)
t0 = time.time()
for i in range(5):
    c = a @ b
    torch.cuda.synchronize()  # on ROCm this synchronizes
    print("iter", i, "sum", c.sum().item())
print("done in", time.time()-t0)
