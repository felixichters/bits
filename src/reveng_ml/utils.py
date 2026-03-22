"""
Utility functions for the project
"""
import torch

def get_pytorch_device() -> torch.device:
    """
    Selects and returns the appropriate PyTorch device (GPU or CPU).
    """
    if torch.cuda.is_available(): # pragma: no cover
        gpu_capability = torch.cuda.get_device_capability(0) # Get capability of first GPU
        # NOTE: My local NVIDIA GTX1050 uses CUDA 6.1, so I need a special case here to fall back to CPU
        if gpu_capability[0] >= 7: # PyTorch typically requires CUDA capability >= 7.0 for recent versions
            device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)} "
                  f"(Capability: {gpu_capability[0]}.{gpu_capability[1]})")
        else:
            device = torch.device("cpu")
            print(f"CUDA device (Capability: {gpu_capability[0]}.{gpu_capability[1]}) "
                  f"is not compatible with PyTorch (requires >= 7.0). Falling back to CPU.")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    return device