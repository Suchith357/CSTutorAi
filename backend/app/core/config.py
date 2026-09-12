from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Application identity ---
    app_name: str = "OSTutorAI"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True

    # --- Retrieval / RAG configuration ---
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 3
    # Grounding: minimum cosine similarity (normalized embeddings + inner
    # product) for the top retrieved chunk. This is an engineering starting
    # point for the current toy corpus, NOT a probability or a calibrated
    # constant - it will be tuned later with an evaluation dataset.
    retrieval_min_score: float = 0.42

    # --- Data / index locations (relative to the repository root) ---
    raw_data_dir: str = "data/raw"
    index_dir: str = "data/index"

    # --- Security / authentication configuration ---
    # Secret used to sign JWTs. NEVER commit a real secret. The dev default is
    # refused in any non-development environment (validated at startup).
    auth_secret_key: str = "dev-insecure-secret-change-me-4f2b8c1a9d7e3f6b0a5c8d2e1f4b7a9c"
    auth_algorithm: str = "HS256"
    auth_access_token_expire_minutes: int = 60
    # bcrypt work factor (cost). 12 is a sane 2026-era default.
    auth_bcrypt_rounds: int = 12

    # --- Login rate limiting (brute-force protection) ---
    auth_rate_limit_max_failures: int = 5
    auth_rate_limit_window_seconds: int = 300
    auth_rate_limit_cooldown_seconds: int = 600

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

    def validate_security_settings(self) -> list[str]:
        """Validate security-critical settings; return a list of problems.

        Called by the FastAPI lifespan at startup. An empty list means the
        configuration is acceptable. In non-development environments the
        in-development auth secret is a hard error (fail closed).
        """
        problems: list[str] = []
        if self.auth_algorithm not in ("HS256", "HS384", "HS512"):
            problems.append(
                f"auth_algorithm {self.auth_algorithm!r} is not a supported HMAC algorithm"
            )
        if self.auth_access_token_expire_minutes < 1:
            problems.append("auth_access_token_expire_minutes must be >= 1")
        if not (4 <= self.auth_bcrypt_rounds <= 15):
            problems.append("auth_bcrypt_rounds must be between 4 and 15")
        if self.environment != "development" and self.auth_secret_key.startswith(
            "dev-insecure"
        ):
            problems.append(
                "auth_secret_key is the development default; set a real "
                "AUTH_SECRET_KEY before running outside development"
            )
        return problems


settings = Settings()