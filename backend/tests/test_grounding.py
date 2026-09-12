"""Grounding / relevance-control tests.

The key behavioral proof required by the project plan:

    WEAK RETRIEVAL -> NO LLM GENERATION

Tests use synthetic RetrievedChunk objects (no model downloads) and a
spy/mock LLM client so the control flow itself is asserted, not just the
answer text. The mock provider keeps the suite offline and fast.
"""

import pytest

from backend.app.llm.client import LLMClient
from backend.app.llm.prompts import SYSTEM_INSTRUCTION, build_grounded_prompt
from backend.app.rag.grounding import (
    INSUFFICIENT_CONTEXT_ANSWER,
    InsufficientContextError,
    evaluate_retrieval,
)
from backend.app.rag.schemas import DocumentMetadata, RetrievedChunk


class SpyLLM(LLMClient):
    """Records whether generate() was called (control-flow proof)."""

    def __init__(self, answer: str = "spy answer"):
        self.calls: list[tuple[str, str]] = []
        self.answer = answer

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.answer


@pytest.fixture
def spy_llm():
    return SpyLLM()


def _chunk(text: str, score: float, title: str = "Toy Chunk") -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=score,
        metadata=DocumentMetadata(
            title=title, source="toy_corpus.md", topic="operating-systems"
        ),
    )


STRONG_CHUNK = _chunk(
    "Deadlock is a situation where a set of processes are permanently "
    "waiting for resources held by each other.",
    score=0.69,
    title="Deadlock",
)

WEAK_CHUNK = _chunk(
    "TCP is a reliable connection-oriented transport layer protocol.",
    score=0.287,
    title="Networking",
)


class TestEvaluateRetrieval:
    def test_no_chunks_raises(self):
        with pytest.raises(InsufficientContextError):
            evaluate_retrieval([], min_score=0.42)

    def test_below_threshold_raises(self):
        with pytest.raises(InsufficientContextError) as excinfo:
            evaluate_retrieval([WEAK_CHUNK], min_score=0.42)
        # Insufficiency reason is exposed for debugging
        assert "below the configured minimum" in excinfo.value.reason

    def test_at_threshold_boundary_passes(self):
        # Exactly at the threshold is sufficient (strictly-below rejects)
        evaluate_retrieval([WEAK_CHUNK], min_score=0.287)

    def test_above_threshold_passes(self):
        evaluate_retrieval([STRONG_CHUNK], min_score=0.42)

    def test_default_threshold_comes_from_settings(self, monkeypatch):
        from backend.app.core import config

        monkeypatch.setattr(config.settings, "retrieval_min_score", 0.10)
        evaluate_retrieval([WEAK_CHUNK])  # must NOT raise (0.287 >= 0.10)

        monkeypatch.setattr(config.settings, "retrieval_min_score", 0.50)
        with pytest.raises(InsufficientContextError):
            evaluate_retrieval([WEAK_CHUNK])  # must raise (0.287 < 0.50)

    def test_error_carries_retrieved_chunks_for_debugging(self):
        with pytest.raises(InsufficientContextError) as excinfo:
            evaluate_retrieval([WEAK_CHUNK], min_score=0.9)
        assert excinfo.value.retrieved == [WEAK_CHUNK]


class TestAnswerWithGroundingUnit:
    """Unit tests for the decision function the route delegates to."""

    def test_insufficient_retrieval_skips_llm(self, spy_llm, monkeypatch):
        from backend.app import main

        monkeypatch.setattr(main.state, "llm", spy_llm)

        with pytest.raises(InsufficientContextError):
            main._answer_with_grounding("Explain TCP vs UDP", [WEAK_CHUNK])
        assert spy_llm.calls == []

    def test_empty_retrieval_skips_llm(self, spy_llm, monkeypatch):
        from backend.app import main

        monkeypatch.setattr(main.state, "llm", spy_llm)

        with pytest.raises(InsufficientContextError):
            main._answer_with_grounding("Any question", [])
        assert spy_llm.calls == []

    def test_sufficient_retrieval_reaches_llm(self, spy_llm, monkeypatch):
        from backend.app import main

        monkeypatch.setattr(main.state, "llm", spy_llm)

        answer = main._answer_with_grounding("What is deadlock?", [STRONG_CHUNK])
        assert answer == "spy answer"
        assert len(spy_llm.calls) == 1


class TestQueryEndpointGrounding:
    """Prove the /query control flow, with the LLM replaced by a spy."""

    @pytest.fixture
    def grounded_client(self, ingested_index, monkeypatch):
        """TestClient on the toy index whose LLM is a SpyLLM.

        The threshold is pinned to 0.65 so decisions are deterministic: the
        fixture's short, topic-pure chunks score high for OS questions and
        around 0.55 for TCP/UDP (in the real 512-token-chunk index the TCP
        score dilutes to ~0.29 - the toy fixture is not score-calibrated).
        """
        from fastapi.testclient import TestClient

        from backend.app import main
        from backend.app.core import config

        monkeypatch.setattr(config.settings, "retrieval_min_score", 0.65)
        spy = SpyLLM(answer="grounded OS explanation [1]")
        monkeypatch.setattr(config.settings, "llm_provider", "mock")
        with TestClient(main.app) as client:
            monkeypatch.setattr(main.state, "llm", spy)
            yield client, spy

    def test_strong_retrieval_calls_llm_and_is_grounded(self, grounded_client):
        client, spy = grounded_client
        body = client.post(
            "/query", json={"question": "What is deadlock in operating systems?"}
        ).json()

        assert body["grounded"] is True
        assert body["answer"] == "grounded OS explanation [1]"
        assert len(spy.calls) == 1

    def test_weak_retrieval_does_not_call_llm(self, grounded_client):
        """THE key grounding test: TCP/UDP must not reach generation."""
        client, spy = grounded_client
        body = client.post(
            "/query",
            json={"question": "Explain the difference between TCP and UDP."},
        ).json()

        assert body["grounded"] is False
        assert body["answer"] == INSUFFICIENT_CONTEXT_ANSWER
        assert spy.calls == []
        # Scores and provenance stay visible for debugging even on fallback
        assert len(body["retrieved"]) >= 1
        assert isinstance(body["retrieved"][0]["score"], float)
        assert body["retrieved"][0]["metadata"]["source"]

    def test_threshold_change_flips_decision(self, grounded_client, monkeypatch):
        client, spy = grounded_client
        from backend.app.core import config

        # Loosen the threshold so the same TCP/UDP question becomes "grounded"
        monkeypatch.setattr(config.settings, "retrieval_min_score", 0.40)
        body = client.post(
            "/query",
            json={"question": "Explain the difference between TCP and UDP."},
        ).json()
        assert body["grounded"] is True
        assert len(spy.calls) == 1

        # Tighten again and confirm the decision flips back
        monkeypatch.setattr(config.settings, "retrieval_min_score", 0.99)
        body = client.post(
            "/query",
            json={"question": "What is deadlock in operating systems?"},
        ).json()
        assert body["grounded"] is False
        assert body["answer"] == INSUFFICIENT_CONTEXT_ANSWER
        assert len(spy.calls) == 1  # unchanged: the tightened query hit the fallback

    def test_threshold_configured_via_env(self, monkeypatch):
        """RETRIEVAL_MIN_SCORE must be honored through the settings system."""
        import importlib

        from backend.app.core import config

        original = config.settings.retrieval_min_score
        try:
            monkeypatch.setenv("RETRIEVAL_MIN_SCORE", "0.77")
            fresh = importlib.reload(config)
            assert fresh.settings.retrieval_min_score == 0.77
        finally:
            monkeypatch.setenv("RETRIEVAL_MIN_SCORE", str(original))
            importlib.reload(config)


class TestGroundedPromptStrengthening:
    def test_system_instruction_is_os_specific_and_grounding_focused(self):
        assert "Operating Systems" in SYSTEM_INSTRUCTION
        assert "authoritative" in SYSTEM_INSTRUCTION
        assert "Do not fabricate citations" in SYSTEM_INSTRUCTION

    def test_prompt_below_threshold_gets_insufficiency_instruction(self):
        system, user = build_grounded_prompt(
            "Explain TCP vs UDP", [WEAK_CHUNK], retrieval_min_score=0.42
        )
        assert "do not contain enough information" in user
        assert "Do not invent an answer" in user

    def test_prompt_above_threshold_gets_context(self):
        system, user = build_grounded_prompt(
            "What is deadlock?", [STRONG_CHUNK], retrieval_min_score=0.42
        )
        assert "Source [1]:" in user
        assert "What is deadlock?" in user
