"""Region embeddings: SigLIP 2 for what a region looks like, a text embedder for what it says.

* visual (SigLIP 2): each region is the L2-normalised mean of its image-crop embedding and
  its text embedding, so a question can match the look of a chart as well as its values.
  SigLIP's text tower reads only 64 tokens.
* text (arctic-embed-s, settings.text_embed_model): the region's full text (512 tokens), so
  paragraphs are matched on meaning. Questions get the model's retrieval prefix.
"""
from functools import lru_cache

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from ..config import settings
from ..device import DEVICE, autocast

# The worker embeds regions on the GPU. The API only embeds one short question at a time,
# so it runs SigLIP on the CPU (a fraction of a second) and leaves the GPU memory to the
# worker: on a 4 GB card two GPU copies of SigLIP 2 overflowed into shared memory and made
# every stage ~10x slower. The API calls use_cpu() before anything is loaded.
_device = DEVICE


def use_cpu() -> None:
    global _device
    if _load.cache_info().currsize or _load_text.cache_info().currsize:
        raise RuntimeError("use_cpu() must be called before the models are loaded")
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


# ── text embedder ────────────────────────────────────────────────────────────
def text_enabled() -> bool:
    return bool(settings.text_embed_model.strip())


@lru_cache(maxsize=1)
def _load_text():
    tok = AutoTokenizer.from_pretrained(settings.text_embed_model)
    model = AutoModel.from_pretrained(settings.text_embed_model, add_pooling_layer=False).to(_device).eval()
    return tok, model


@torch.inference_mode()
def _encode_text(texts: list[str], batch_size: int = 16) -> torch.Tensor:
    tok, model = _load_text()
    parts = []
    for start in range(0, len(texts), batch_size):
        inputs = tok([t or " " for t in texts[start:start + batch_size]], padding=True, truncation=True,
                     max_length=512, return_tensors="pt").to(_device)
        cls = model(**inputs).last_hidden_state[:, 0]  # arctic-embed / bge use the [CLS] vector
        parts.append(torch.nn.functional.normalize(cls.float(), dim=-1).cpu())
    return torch.cat(parts) if parts else torch.empty(0)


def embed_passages(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """Text vectors for region contents (no prefix)."""
    return _encode_text(texts, batch_size).tolist() if texts else []


def embed_query(question: str) -> list[float]:
    """Text vector for a question (with the model's retrieval prefix)."""
    return _encode_text([settings.text_query_prefix + question])[0].tolist()


def text_dim() -> int:
    _, model = _load_text()
    return int(model.config.hidden_size)


def dim() -> int:
    _, model = _load()
    return int(model.config.projection_dim if hasattr(model.config, "projection_dim") else model.config.text_config.hidden_size)
