"""Download every model the application uses into the shared Hugging Face cache.

Run once during setup (or before a demo) so that the first parse does not wait for
roughly 2 GB of downloads:

    python -m scripts.download_models

The cache lives in ~/.cache/huggingface and is shared by every copy of the
project on the machine, so later setups and runs reuse it.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download  # noqa: E402

from app.config import settings  # noqa: E402

_OTHER_FRAMEWORKS = ["*.msgpack", "*.h5", "*.onnx", "*.ot", "tf_model*", "flax_model*", "rust_model*", "*.md"]


def _snapshot(repo: str) -> str:
    """Download a PyTorch model repo, skipping duplicate weights for other frameworks."""
    ignore = list(_OTHER_FRAMEWORKS)
    if any(name.endswith(".safetensors") for name in list_repo_files(repo)):
        ignore.append("*.bin")  # transformers loads safetensors; the .bin copy would double the download
    return snapshot_download(repo, ignore_patterns=ignore)


def main() -> None:
    jobs = [
        ("layout (DocLayout-YOLO)", lambda: hf_hub_download(settings.layout_repo, settings.layout_weights)),
        ("embeddings (SigLIP)", lambda: _snapshot(settings.siglip_model)),
        *([("text embeddings", lambda: _snapshot(settings.text_embed_model))] if settings.text_embed_model.strip() else []),
        ("tables (Table Transformer)", lambda: _snapshot(settings.table_model)),
        ("charts (DePlot)", lambda: _snapshot(settings.deplot_model)),
    ]
    total = time.perf_counter()
    for name, job in jobs:
        start = time.perf_counter()
        print(f"· {name} ...", flush=True)
        job()
        print(f"  ready in {time.perf_counter() - start:.1f}s", flush=True)
    print(f"All models are cached ({time.perf_counter() - total:.1f}s).")


if __name__ == "__main__":
    main()
