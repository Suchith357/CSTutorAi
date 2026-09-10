"""LLM layer tests: prompts, client abstraction, mocked /query integration.

No real model is downloaded or loaded here - the mock provider is used so the
suite runs offline and fast.
"""

import pytest

from backend.app.llm.client import (
    LLMClient,
    MockLLMClient,
    create_llm_client,
)
from backend.app.llm.prompts import (
    SYSTEM_INSTRUCTION,
    build_grounded_prompt,
    format_context,
)


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient()


class TestGroundedPrompt:
    def test_system_instruction_contains_grounding_rules(self):
        assert "ONLY the retrieved sources" in SYSTEM_INSTRUCTION
        assert "Never invent facts" in SYSTEM_INSTRUCTION
        assert "do not contain enough information" in SYSTEM_INSTRUCTION
        assert "Teach, don't just answer" in SYSTEM_INSTRUCTION
        assert "cite it as [1], [2]" in SYSTEM_INSTRUCTION

    def test_format_context_numbers_sources(self, toy_chunks):
        context = format_context(toy_chunks[:2])
        assert "Source [1]:" in context
        assert "Source [2]:" in context
        assert "toy_corpus.md" in context
        assert toy_chunks[0].text in context

    def test_format_context_empty(self):
        assert format_context([]) == ""

    def test_prompt_with_context_includes_question_and_rules(self, toy_chunks):
        system, user = build_grounded_prompt("What is deadlock?", toy_chunks[:2])
        assert system == SYSTEM_INSTRUCTION
        assert "What is deadlock?" in user
        assert "Source [1]:" in user
        assert "Cite sources as [1], [2]" in user
        # chunk text included as grounding context
        assert toy_chunks[0].text in user

    def test_prompt_without_context_demands_insufficiency_statement(self):
        system, user = build_grounded_prompt("What is quantum complexity theory?", [])
        assert "do not contain enough information" in user
        assert "Do not invent an answer" in user


class TestLLMClients:
    def test_mock_and_transformers_satisfy_interface(self):
        assert isinstance(MockLLMClient(), LLMClient)
        from backend.app.llm.client import TransformersLLMClient
        assert isinstance(TransformersLLMClient(), LLMClient)

    def test_mock_generation_with_sources(self, mock_llm, toy_chunks):
        system, user = build_grounded_prompt("What is deadlock?", toy_chunks[:2])
        answer = mock_llm.generate(system, user)
        assert "Mock grounded answer" in answer
        assert "Source [1]" in answer

    def test_mock_generation_without_sources(self, mock_llm):
        system, user = build_grounded_prompt("Obscure question", [])
        answer = mock_llm.generate(system, user)
        assert "do not contain enough information" in answer

    def test_factory_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_client(provider="does-not-exist")

    def test_factory_mock_provider(self):
        assert isinstance(create_llm_client(provider="mock"), MockLLMClient)


class TestQueryEndpointWithLLM:
    @pytest.fixture
    def mock_provider_client(self, monkeypatch):
        """TestClient whose lifespan builds a MockLLMClient (no model download).

        The retriever stays real (loads the persisted toy index) so this is a
        true end-to-end test with only the LLM mocked.
        """
        from fastapi.testclient import TestClient

        from backend.app import main
        from backend.app.core import config

        monkeypatch.setattr(config.settings, "llm_provider", "mock")
        with TestClient(main.app) as client:
            yield client

    def test_query_returns_answer_and_retrieved(self, mock_provider_client):
        """End-to-end /query with mocked LLM: structure + grounding contract."""
        response = mock_provider_client.post(
            "/query", json={"question": "What is deadlock?"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["question"] == "What is deadlock?"
        assert isinstance(body["answer"], str) and body["answer"]
        assert len(body["retrieved"]) >= 1
        assert body["retrieved"][0]["metadata"]["source"]
        assert isinstance(body["retrieved"][0]["score"], float)

    def test_query_without_llm_returns_503(self, mock_provider_client, monkeypatch):
        from backend.app import main

        monkeypatch.setattr(main.state, "llm", None)
        response = mock_provider_client.post("/query", json={"question": "hi"})

        assert response.status_code == 503
        assert "LLM" in response.json()["detail"]
