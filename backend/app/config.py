from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# config.py is located at:
# DocLens/backend/app/config.py
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent

# Prefer backend/.env if it exists.
# Otherwise, use DocLens/.env.
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

    # Database and queues
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/figuremind"
    )
    redis_url: str = "redis://localhost:6379/0"
    parse_queue: str = "figuremind:parse"

    # Vector database
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "figuremind_regions"

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    gemini_fallback_model: str = "gemini-3.5-flash-lite"

    # Storage
    storage_dir: str = "./data"

    # Device
    # auto prefers CUDA, then Apple MPS, then CPU.
    torch_device: str = "auto"

    # AI models
    layout_repo: str = "juliozhao/DocLayout-YOLO-DocStructBench"
    layout_weights: str = "doclayout_yolo_docstructbench_imgsz1024.pt"
    siglip_model: str = "google/siglip-base-patch16-224"
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

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173"
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
