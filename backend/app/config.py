import json
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Every path is derived from this file's location (<project>/backend/app/config.py),
# so the project folder can be renamed or moved without changing any code.
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent

# The product name lives in one place: <project>/brand.json. The API title, the
# health endpoint, the frontend and the model prompt all read it from there.
_BRAND_FILE = PROJECT_DIR / "brand.json"
_BRAND_DEFAULTS = {"name": "DocLens", "tagline": "Grounded answers from visual PDFs", "logo_letter": "D"}
try:
    BRAND: dict[str, str] = {**_BRAND_DEFAULTS, **json.loads(_BRAND_FILE.read_text(encoding="utf-8"))}
except (OSError, ValueError):
    BRAND = dict(_BRAND_DEFAULTS)
APP_NAME = BRAND["name"]

# Prefer backend/.env if it exists.
# Otherwise, use <project>/.env.
BACKEND_ENV_FILE = BACKEND_DIR / ".env"
PROJECT_ENV_FILE = PROJECT_DIR / ".env"

if BACKEND_ENV_FILE.exists():
    ENV_FILE = BACKEND_ENV_FILE
elif PROJECT_ENV_FILE.exists():
    ENV_FILE = PROJECT_ENV_FILE
else:
    ENV_FILE = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Database and queues.
    # The "docqa" identifiers below are internal storage keys (database, queue,
    # collection). They are deliberately neutral and independent of the product name in
    # brand.json: renaming the product must never change them, or stored data is orphaned.
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/docqa"
    )
    redis_url: str = "redis://127.0.0.1:6379/0"
    parse_queue: str = "docqa:parse"

    # Vector database
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "regions"

    # Gemini
    gemini_api_key: str = ""
    # Chosen from the free-tier limits on AI Studio's Rate Limit page (13 Nov 2026):
    #   3.5 Flash-Lite 15 req/min, 500/day  -> primary (~2-6 s)
    #   3.1 Flash-Lite 15 req/min, 500/day  -> first fallback: its own quota, read our chart correctly (1-3 s)
    #   3.6 Flash       5 req/min,  20/day  -> last resort (3.8 Flash has the same 20/day limit and stalled)
    # "gemini-3.5-flash" has no quota row of its own, so it is not used. A model that returns
    # "not found" (retired) or 429 (quota used up) is skipped automatically.
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_fallback_model: str = "gemini-3.1-flash-lite,gemini-3.6-flash"
    # thinking level sent to the model ("minimal", "low", "medium", "high"; empty = model default)
    gemini_thinking: str = "minimal"
    # wall-clock limit per attempt (request + hedged backup), tries per model, total per call (s)
    gemini_timeout_s: float = 40
    gemini_max_attempts: int = 2
    gemini_budget_s: float = 90
    # send a duplicate request when the first has not answered after this many seconds (0 = off)
    # Off by default: on the free tier every backup request counts against the per-minute limit
    # and caused 429s. Turn on (e.g. 6) once billing is enabled on the Gemini project.
    gemini_hedge_after_s: float = 0

    # Storage
    storage_dir: str = "./data"

    # Device
    # auto prefers CUDA, then Apple MPS, then CPU.
    torch_device: str = "auto"
    # float16 autocast for the encoders when running on an NVIDIA GPU
    cuda_fp16: bool = True
    # load every model when the worker starts, so the first upload is not slowed by model loading
    preload_models: bool = True
    # the API embeds questions on the CPU so the GPU memory stays with the worker
    api_embed_on_cpu: bool = True

    # AI models
    # DocLayout-YOLO DocStructBench weights, Jan 2025 release trained at 1280 px. On the sample
    # paper it found the same tables/charts/equations as the Oct 2024 1024-px weights with higher
    # confidence and half the duplicate boxes, for +65 ms per page.
    layout_repo: str = "juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501"
    layout_weights: str = "doclayout_yolo_docstructbench_imgsz1280_2501.pt"
    layout_imgsz: int = 1280
    # SigLIP 2 (Feb 2025): same size and API as SigLIP; 10/16 vs 8/16 correct regions in the
    # top 6 on our retrieval test (hybrid with BM25).
    siglip_model: str = "google/siglip2-base-patch16-224"
    table_model: str = (
        "microsoft/table-transformer-structure-recognition-v1.1-all"
    )
    deplot_model: str = "google/deplot"

    # PDF/image processing
    render_dpi: int = 144
    embedding_batch_size: int = 16

    # Retrieval and verification
    grounding_threshold: float = 0.35
    top_k_retrieve: int = 20
    top_k_context: int = 6
    max_verify_attempts: int = 2
    # hybrid retrieval: weight of the lexical (BM25) ranking fused with SigLIP vectors
    lexical_weight: float = 1.0
    # ask Gemini to score the retrieved regions before answering (one extra API call per question)
    # Off by default: the local hybrid SigLIP+BM25 scoring ranks well and this halves Gemini calls.
    gemini_rerank: bool = False

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_upload_mb: int = 25

    def resolve_project_path(self, value: str | Path) -> Path:
        """Resolve configured and persisted relative paths from the project root."""
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()

    # Derived storage paths
    @property
    def data_dir(self) -> Path:
        path = self.resolve_project_path(self.storage_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def upload_dir(self) -> Path:
        path = self.data_dir / "uploads"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def pages_dir(self) -> Path:
        path = self.data_dir / "pages"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def crops_dir(self) -> Path:
        path = self.data_dir / "crops"
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
