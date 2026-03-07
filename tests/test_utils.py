
import torch

from reveng_ml.utils import get_pytorch_device


def test_get_pytorch_device_returns_device():
    """get_pytorch_device returns a torch.device instance."""
    device = get_pytorch_device()
    assert isinstance(device, torch.device)


def test_get_pytorch_device_is_cpu_or_cuda():
    """The returned device type is either 'cpu' or 'cuda'."""
    device = get_pytorch_device()
    assert device.type in ("cpu", "cuda"), (
        f"Expected device type 'cpu' or 'cuda', got '{device.type}'"
    )
