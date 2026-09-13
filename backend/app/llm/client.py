"""LLM client abstraction and the local providers (HuggingFace transformers).

OSTutorAI depends only on the LLMClient interface; the concrete provider is
selected through settings (llm_provider), so the model can be replaced later
without touching application code.

Providers:
- TransformersLLMClient: local open-weight instruction model (free, no API key).
  Default model: Qwen/Qwen2.5-3B-Instruct - chosen for teaching quality.
  Loading strategy (llm_device="auto", the default):
    * CUDA + bitsandbytes available -> 4-bit NF4 quantized on the GPU
      (~2-3 GB VRAM; the 3B model does NOT fit 6 GB VRAM in fp32/fp16).
    * anything unavailable -> reliable CPU float32 fallback (never crashes).
  Greedy decoding (do_sample=False) is deterministic and coherent.
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
    (e.g. at app startup) stays cheap. See the module docstring for the
    device/quantization strategy and CPU fallback.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_new_tokens: int | None = None,
        device: str | None = None,
        torch_dtype: str | None = None,
    ):
        self.model_name = model_name or settings.llm_model_name
        self.max_new_tokens = max_new_tokens or settings.llm_max_new_tokens
        self.device = device or settings.llm_device
        self.torch_dtype = torch_dtype or settings.llm_torch_dtype
        self._model = None
        self._tokenizer = None
        self._quantization = "none"
        self._cpu_fallback_used = False
        self._cuda_available = False
        self._dtype_requested = self.torch_dtype
        self._dtype_effective: str | None = None

    # -- loading -----------------------------------------------------------

    @staticmethod
    def _cuda_and_bnb_available() -> tuple[bool, bool]:
        """Check CUDA and bitsandbytes availability without importing bnb."""
        import importlib.util

        import torch

        cuda = torch.cuda.is_available()
        bnb = importlib.util.find_spec("bitsandbytes") is not None
        return cuda, (bnb if cuda else False)

    def _load_gpu_4bit(self):
        """Load quantized (NF4, double-quant) on the CUDA device."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=settings.llm_load_in_4bit,
            bnb_4bit_quant_type=settings.llm_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
        )
        self._quantization = (
            f"4-bit {settings.llm_4bit_quant_type} (double-quant, compute bfloat16)"
        )
        self._dtype_effective = "bfloat16 (compute, 4-bit weights)"
        model.eval()
        return model

    def _load_cpu(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": torch.float32,  # CPU auto resolves to float32
        }.get(self.torch_dtype, torch.float32)
        self._dtype_effective = str(dtype).replace("torch.", "")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
        )
        self._quantization = "none"
        model.eval()
        return model.to("cpu")

    def _ensure_loaded(self):
        if self._model is not None:
            return

        cuda, bnb = self._cuda_and_bnb_available()
        self._cuda_available = cuda

        device = self.device
        if device == "auto":
            if cuda and bnb and settings.llm_load_in_4bit:
                device = "cuda"
                loader = self._load_gpu_4bit
            else:
                device = "cpu"
                loader = self._load_cpu
                if self.torch_dtype == "auto":
                    self._dtype_requested = "float32"
        elif device.startswith("cuda"):
            if not cuda:
                print(
                    f"LLM device {device!r} requested but CUDA is unavailable; "
                    "falling back to CPU"
                )
                device = "cpu"
                loader = self._load_cpu
            elif bnb and settings.llm_load_in_4bit:
                loader = self._load_gpu_4bit
            else:
                loader = self._load_cpu
        else:
            loader = self._load_cpu

        try:
            self._model = loader()
        except Exception as exc:  # noqa: BLE001 - any GPU failure -> CPU
            if device == "cpu":
                raise
            print(f"LLM failed to load on {device!r} ({exc}); falling back to CPU")
            device = "cpu"
            self._cpu_fallback_used = True
            self._model = self._load_cpu()

        self._effective_device = device
        self._cpu_fallback_used = self._cpu_fallback_used or (
            self.device != "cpu" and device == "cpu"
        )
        print(
            f"LLM loaded: {self.model_name} | device={device} | "
            f"{self._quantization} | dtype={self._dtype_effective}"
        )

    # -- inference -----------------------------------------------------------

    def generate(self, system: str, user: str, max_new_tokens: int | None = None) -> str:
        self._ensure_loaded()
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(
            getattr(self, "_effective_device", "cpu")
        )
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    # -- diagnostics -----------------------------------------------------------

    def describe(self) -> dict:
        """Report exactly how the model is (or will be) loaded.

        Exposes: configured model, requested vs effective device, dtype,
        quantization, whether CUDA was available in the installed torch
        build, and whether a CPU fallback happened. Intended for startup
        logs and diagnostics.
        """
        return {
            "model": self.model_name,
            "requested_device": self.device,
            "effective_device": getattr(self, "_effective_device", None),
            "torch_dtype": self._dtype_effective or self._dtype_requested,
            "quantization": self._quantization,
            "cuda_available": self._cuda_available,
            "cpu_fallback_used": self._cpu_fallback_used,
            "loaded": self._model is not None,
        }


def create_llm_client(provider: str | None = None) -> LLMClient:
    """Factory: build the LLM client selected in settings."""
    provider = provider or settings.llm_provider
    if provider == "mock":
        return MockLLMClient()
    if provider == "transformers":
        return TransformersLLMClient()
    raise ValueError(f"Unknown LLM provider: {provider!r}")
