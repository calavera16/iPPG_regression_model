import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA is available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print(
        "PyTorch was installed without CUDA support or the CUDA driver is not compatible."
    )
