import torch

# Check if a CUDA-compatible GPU is available
is_available = torch.cuda.is_available()
print(f"Is a compatible GPU available? {is_available}")

if is_available:
    # Get the number of available GPUs
    device_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {device_count}")

    # Get the name of the current GPU
    current_device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"Current GPU name: {current_device_name}")
else:
    print("PyTorch did not find any compatible GPUs.")