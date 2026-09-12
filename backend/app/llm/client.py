"""LLM client abstraction and the first local provider (HuggingFace transformers).

OSTutorAI depends only on the LLMClient interface; the concrete provider is
selected through settings (llm_provider), so the model can be replaced later
without touching application code.

Providers:
- TransformersLLMClient: local open-weight instruction model (free, no API key).
  Chosen model: Qwen/Qwen2.5-0.5B-Instruct - small enough for CPU / 6 GB VRAM,
  loaded lazily, decoded greedily for determinism.
- MockLLMClient: deterministic offline client used by the automated tests, so
  pytest never downloads or loads a real model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from backend.app.core.config import settings


class LLMClient(ABC):
    """Interface for grounded text generation (replaceable per provider)."""

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Generate a completion for the given system + user messages."""


class MockLLMClient(LLMClient):
    """Deterministic offline client for tests. No model download."""

    def generate(self, system: str, user: str) -> str:
        sources = [ln for ln in user.splitlines() if ln.startswith("Source [")]
        if not sources or "do not contain enough information" in user:
            return (
                "The available sources do not contain enough information "
                "to answer this question."
            )
        return f"Mock grounded answer citing {len(sources)} source(s). {sources[0]}"


class TransformersLLMClient(LLMClient):
    """Local instruction-tuned model via HuggingFace transformers.

    The model loads lazily on first generate() so constructing the client
    (e.g. at app startup) stays cheap. Greedy decoding (do_sample=False) is
    deterministic and more coherent for small models.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_new_tokens: int | None = None,
    ):
        self.model_name = model_name or settings.llm_model_name
        self.max_new_tokens = max_new_tokens or settings.llm_max_new_tokens
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32,
        )
        self._model.eval()

    def generate(self, system: str, user: str) -> str:
        self._ensure_loaded()
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


def create_llm_client(provider: str | None = None) -> LLMClient:
    """Factory: build the LLM client selected in settings."""
    provider = provider or settings.llm_provider
    if provider == "mock":
        return MockLLMClient()
    if provider == "transformers":
        return TransformersLLMClient()
    raise ValueError(f"Unknown LLM provider: {provider!r}")
