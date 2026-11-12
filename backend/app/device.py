"""Resolve the best available PyTorch device across Windows, macOS, and Linux."""
import os
import torch

from .config import settings


def get_device() -> torch.device:
    want = os.getenv("TORCH_DEVICE", settings.torch_device).lower()
    if want in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if want in ("auto", "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # unsupported ops → CPU instead of crash
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
