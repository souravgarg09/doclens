"""SigLIP region-grounded multimodal embeddings.

Each region is embedded as the L2-normalised mean of its *image crop* embedding
and its *text/structured content* embedding, so a query (text) can match the
visual appearance of a chart as well as its decoded values.
"""
from functools import lru_cache

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from ..config import settings
from ..device import DEVICE, autocast

# The worker embeds regions on the GPU. The API only embeds one short question at a time,
# so it runs SigLIP on the CPU (a fraction of a second) and leaves the GPU memory to the
# worker: on a 4 GB card two GPU copies of SigLIP 2 overflowed into shared memory and made
# every stage ~10x slower. The API calls use_cpu() before anything is loaded.
_device = DEVICE


def use_cpu() -> None:
    global _device
    if _load.cache_info().currsize:
        raise RuntimeError("use_cpu() must be called before the model is loaded")
    _device = torch.device("cpu")


@lru_cache(maxsize=1)
def _load():
    proc = AutoProcessor.from_pretrained(settings.siglip_model)
    model = AutoModel.from_pretrained(settings.siglip_model).to(_device).eval()
    return proc, model


def _as_tensor(features) -> torch.Tensor:
    """transformers 4 returns a tensor from get_*_features; transformers 5 returns an output object."""
    return features if isinstance(features, torch.Tensor) else features.pooler_output


@torch.inference_mode()
def embed_text(texts: list[str]) -> torch.Tensor:
    proc, model = _load()
    # SigLIP was trained with padding="max_length" (64 tokens)
    inputs = proc(text=[t[:400] for t in texts], padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(_device)
    with autocast():
        feats = model.get_text_features(**inputs)
    return torch.nn.functional.normalize(_as_tensor(feats).float(), dim=-1).cpu()


@torch.inference_mode()
def embed_images(images: list[Image.Image]) -> torch.Tensor:
    proc, model = _load()
    inputs = proc(images=[im.convert("RGB") for im in images], return_tensors="pt").to(_device)
    with autocast():
        feats = model.get_image_features(**inputs)
    return torch.nn.functional.normalize(_as_tensor(feats).float(), dim=-1).cpu()


def embed_region(crop: Image.Image | None, text: str) -> list[float]:
    return embed_regions([(crop, text)], batch_size=1)[0]


def embed_regions(
    regions: list[tuple[Image.Image | None, str]], batch_size: int = 16
) -> list[list[float]]:
    """Embed a page of regions in batches while preserving region order.

    Model inference has a noticeable fixed cost. Running one forward pass per
    region made CPU parsing needlessly slow; batching produces the same
    normalised text/image vectors with far fewer model calls.
    """
    if not regions:
        return []
    batch_size = max(1, batch_size)

    text_parts = [
        embed_text([text or " " for _, text in regions[start:start + batch_size]])
        for start in range(0, len(regions), batch_size)
    ]
    text_vectors = torch.cat(text_parts, dim=0)

    image_positions = [index for index, (crop, _) in enumerate(regions) if crop is not None]
    image_vectors: dict[int, torch.Tensor] = {}
    for start in range(0, len(image_positions), batch_size):
        positions = image_positions[start:start + batch_size]
        crops = [regions[index][0] for index in positions]
        vectors = embed_images([crop for crop in crops if crop is not None])
        image_vectors.update(zip(positions, vectors))

    output: list[list[float]] = []
    for index, text_vector in enumerate(text_vectors):
        image_vector = image_vectors.get(index)
        vector = (
            text_vector
            if image_vector is None
            else torch.nn.functional.normalize(0.5 * (text_vector + image_vector), dim=-1)
        )
        output.append(vector.tolist())
    return output


def dim() -> int:
    _, model = _load()
    return int(model.config.projection_dim if hasattr(model.config, "projection_dim") else model.config.text_config.hidden_size)
