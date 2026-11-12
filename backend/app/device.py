"""Resolve the best available PyTorch device across Windows, macOS, and Linux.

TORCH_DEVICE=auto prefers CUDA, then Apple MPS, then CPU. On CUDA the encoders run
under float16 autocast (roughly 2x faster and half the memory on consumer GPUs);
MPS and CPU stay in float32 because fp16 autocast there is either unsupported or
slower for these models.
"""
import contextlib
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
# GTX 16-series cards (Turing TU116/TU117) return NaN/empty results in float16: on a
# GTX 1650 the layout detector found 0 regions in fp16 and the correct 16/15/5 regions
# per page in fp32. Half precision is therefore disabled on those cards.
_FP16_BROKEN_GPUS = ("GTX 16",)
CUDA_NAME = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else ""
USE_FP16 = (DEVICE.type == "cuda" and settings.cuda_fp16
            and not any(marker in CUDA_NAME for marker in _FP16_BROKEN_GPUS))

if DEVICE.type == "cuda":
    # Let cuDNN benchmark convolution algorithms once per input shape and keep the fastest.
    # Measured on a GTX 1650: layout detection 4.6 s → 0.15 s per page. Page renders share
    # one size, so the one-off tuning cost is paid only on the first page.
    torch.backends.cudnn.benchmark = True

if DEVICE.type == "cpu":
    # Use every physical core for intra-op parallelism; PyTorch sometimes picks fewer on Windows.
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))


def autocast():
    """Context manager for model inference: fp16 autocast on CUDA, a no-op elsewhere."""
    if USE_FP16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def describe() -> str:
    """Human-readable device name for the health endpoint and logs."""
    if DEVICE.type == "cuda":
        return f"CUDA · {CUDA_NAME} · {'fp16' if USE_FP16 else 'fp32'}"
    if DEVICE.type == "mps":
        return "Apple MPS"
    return f"CPU · {torch.get_num_threads()} threads"
