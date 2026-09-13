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

    # --- Tutor Engine / adaptive teaching ---
    # Topic mastery update deltas (mastery stays clamped to [0, 1]).
    tutor_mastery_gain_correct: float = 0.10
    tutor_mastery_drop_incorrect: float = 0.05
    tutor_mastery_gain_hint: float = 0.0  # hints are tracked, not scored
    # Difficulty scale 1..5: 1 beginner, 2 basic, 3 intermediate,
    # 4 advanced, 5 exam/viva. Mapping mastery -> difficulty thresholds:
    # < 0.25 -> 1, < 0.45 -> 2, < 0.70 -> 3, < 0.90 -> 4, else 5.
    tutor_difficulty_low_threshold: float = 0.25
    tutor_difficulty_medium_threshold: float = 0.45
    tutor_difficulty_high_threshold: float = 0.70
    tutor_difficulty_exam_threshold: float = 0.90
    # Student model per-student topic history cap (transparent in-memory model).
    tutor_recent_performance_limit: int = 10
    # Default student id when a tutoring request omits one.
    tutor_default_student_id: str = "demo"
    # Teaching answers need more room than QA answers (definitions + points +
    # check question). Separate budget so /query stays fast.
    tutor_max_new_tokens: int = 512
    # Lower generation cap for beginner difficulty (1-2): prompt-level
    # conciseness constraints do the real work; this bounds runaway output.
    tutor_max_new_tokens_beginner: int = 320

    # --- LLM configuration (local-first, free) ---
    # provider: "transformers" (local HF model) or "mock" (offline tests)
    llm_provider: str = "transformers"
    # Instruction-tuned model. 3B chosen for teaching quality; fits 6 GB VRAM
    # only with quantization, so default device is CPU (fp32, ~12.4 GB RAM).
    # Set LLM_DEVICE=cuda to try GPU first; the client falls back to CPU if
    # CUDA is unavailable or the model cannot be placed on it.
    llm_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    # "auto": try CUDA 4-bit (NF4) when torch+CUDA+bitsandbytes are all
    # available (3B fp32 does NOT fit 6 GB VRAM; 4-bit needs ~2-3 GB);
    # otherwise fall back to CPU float32. Explicit "cuda"/"cpu" force the
    # device (CPU fallback still applies if placement fails).
    llm_device: str = "auto"
    # "auto": bfloat16 on GPU with 4-bit quantization, float32 on CPU.
    llm_torch_dtype: str = "auto"
    # 4-bit quantization settings (bitsandbytes NF4) used when device=auto/cuda
    # and CUDA is available. Double quantization + compute in bfloat16.
    llm_load_in_4bit: bool = True
    llm_4bit_quant_type: str = "nf4"
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