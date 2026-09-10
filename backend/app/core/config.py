from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Application identity ---
    app_name: str = "CSTutorAI"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True

    # --- Retrieval / RAG configuration ---
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 3

    # --- Data / index locations (relative to the repository root) ---
    raw_data_dir: str = "data/raw"
    index_dir: str = "data/index"

    # --- LLM configuration (local-first, free) ---
    # provider: "transformers" (local HF model) or "mock" (offline tests)
    llm_provider: str = "transformers"
    llm_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    llm_max_new_tokens: int = 256

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def raw_data_path(self) -> Path:
        return Path(self.raw_data_dir)

    @property
    def index_path(self) -> Path:
        return Path(self.index_dir)


settings = Settings()