"""The Tutor Engine: one reusable pipeline behind all teaching modes.

Flow (grounding gate stays authoritative, exactly as in /query):

    student request
        -> retrieval (existing Retriever)
        -> grounding gate (existing evaluate_retrieval, threshold unchanged)
        -> [insufficient] safe tutoring refusal, LLM NOT called
        -> [sufficient]   topic from retrieval metadata
                          difficulty from the student model
                          mode-specific prompt -> local LLM
                          labeled sections parsed into a structured TutorResponse

The engine never lowers the threshold and never bypasses the gate.

Structured output: the prompt asks the model to answer in clearly labeled
sections (Explanation:, Key points:, Example:, Check question:, ...). The
engine parses those labels into the matching TutorResponse fields so each
piece of content lives in its own structured field instead of one giant
explanation string. When the model ignores the labels, the previous
mode-driven heuristic is used as a fallback (honest, no fabrication).
"""

from __future__ import annotations

import re

from backend.app.core.config import settings
from backend.app.llm.client import LLMClient
from backend.app.rag.grounding import (
    INSUFFICIENT_CONTEXT_ANSWER,
    InsufficientContextError,
    evaluate_retrieval,
)
from backend.app.rag.retriever import Retriever
from backend.app.rag.schemas import RetrievedChunk
from backend.app.tutoring.modes import ModeStrategy, get_strategy
from backend.app.tutoring.prompts import build_tutor_prompt
from backend.app.tutoring.schemas import TutorMode, TutorResponse
from backend.app.tutoring.student_model import StudentModel

# Labels the structured-output prompt asks for. Order matters only for
# readability; matching is case-insensitive and allows **bold** markers.
_LABEL_NAMES = (
    "explanation",
    "simplified explanation",
    "example",
    "key points",
    "hint",
    "common mistake",
    "check question",
    "practice question",
)
_LABEL_RE = re.compile(
    r"^\s*(?:\*\*)?\[?(" + "|".join(_LABEL_NAMES) + r")\]?\s*(?:\*\*)?\s*:\s*(?:\*\*)?\s*(.*)$",
    re.IGNORECASE,
)


def detect_topic(chunks: list[RetrievedChunk]) -> str:
    """Pick the dominant OS topic from retrieval metadata (no classifier).

    Chunks are already ranked by similarity, so the first chunk's topic is
    the strongest signal; ties among the top topics are broken by frequency
    then first-appearance order.
    """
    if not chunks:
        return "unknown"
    return chunks[0].metadata.topic


def parse_sections(answer: str) -> tuple[dict[str, str], str]:
    """Parse labeled sections out of an LLM answer.

    Returns:
        (sections, preamble) where sections maps a lowercase label name
        (e.g. "check question") to its text and preamble is any prose that
        appeared before the first label (usually empty when the model
        follows the format).
    """
    sections: dict[str, list[str]] = {}
    preamble: list[str] = []
    current: str | None = None

    for line in answer.splitlines():
        match = _LABEL_RE.match(line)
        if match:
            current = match.group(1).lower()
            rest = match.group(2).strip()
            sections.setdefault(current, [rest] if rest else [])
        elif current is not None:
            sections[current].append(line)
        else:
            preamble.append(line)

    cleaned = {
        name: "\n".join(lines).strip().replace("**", "").strip()
        for name, lines in sections.items()
    }
    return {k: v for k, v in cleaned.items() if v}, "\n".join(preamble).strip()


class TutorEngine:
    """Facade binding retriever + LLM + student model into tutoring turns."""

    def __init__(self, retriever: Retriever, llm: LLMClient) -> None:
        self.retriever = retriever
        self.llm = llm
        # Teaching answers need more room than plain QA (definition + points +
        # check question); transformers clients honor this per-call override.
        self.max_new_tokens = settings.tutor_max_new_tokens

    def _token_budget(self, difficulty: int) -> int:
        """Generation budget per difficulty: beginner answers must stay short.

        Prompt-level constraints do the real work; this is an appropriate
        generation limit so a runaway generation cannot produce essay-length
        beginner text. No post-hoc truncation (which would cut sentences).
        """
        if difficulty <= 2:
            return min(
                settings.tutor_max_new_tokens_beginner, settings.tutor_max_new_tokens
            )
        return settings.tutor_max_new_tokens

    def _generate(self, system: str, user: str, difficulty: int) -> str:
        """Call the LLM with the difficulty-appropriate token budget."""
        from backend.app.llm.client import TransformersLLMClient

        if isinstance(self.llm, TransformersLLMClient):
            return self.llm.generate(
                system, user, max_new_tokens=self._token_budget(difficulty)
            )
        return self.llm.generate(system, user)

    # ------------------------------------------------------------------
    # Tutoring turn
    # ------------------------------------------------------------------
    def tutor(
        self,
        question: str,
        mode: TutorMode,
        student: StudentModel,
    ) -> TutorResponse:
        strategy: ModeStrategy = get_strategy(mode)
        retrieved = self.retriever.retrieve(question)

        # Grounding gate FIRST - identical policy to /query.
        try:
            evaluate_retrieval(retrieved)
        except InsufficientContextError as exc:
            return TutorResponse(
                question=question,
                mode=mode,
                topic=detect_topic(exc.retrieved),
                difficulty=student.difficulty(detect_topic(exc.retrieved)),
                sources=exc.retrieved,
                grounded=False,
                refusal_reason=exc.reason,
                explanation=INSUFFICIENT_CONTEXT_ANSWER,
            )

        topic = detect_topic(retrieved)
        difficulty = student.difficulty(topic)
        system, user = build_tutor_prompt(question, retrieved, strategy, difficulty)
        answer = self._generate(system, user, difficulty)

        response = TutorResponse(
            question=question,
            mode=mode,
            topic=topic,
            difficulty=difficulty,
            sources=retrieved,
            grounded=True,
        )
        return strategy_fields(response, answer, strategy)

    # ------------------------------------------------------------------
    # Learning interaction
    # ------------------------------------------------------------------
    def evaluate_and_update(
        self,
        student: StudentModel,
        topic: str,
        question: str,
        student_answer: str,
        reference_points: list[str],
        hint_used: bool = False,
    ) -> tuple:
        """Evaluate a student answer, update mastery, return (evaluation, snapshot)."""
        from backend.app.tutoring.evaluator import evaluate_answer
        from backend.app.tutoring.schemas import EvaluateAnswerRequest

        request = EvaluateAnswerRequest(
            student_id=student.student_id,
            topic=topic,
            question=question,
            student_answer=student_answer,
            reference_points=reference_points,
            hint_used=hint_used,
        )
        evaluation = evaluate_answer(request, self.llm)

        record = student.topic(topic)
        record.apply_result(evaluation.verdict, hint_used)
        return evaluation, record.snapshot()


# Text fields that can serve as a mode's "main" content, in priority order.
_MAIN_TEXT_FIELDS = ("explanation", "simplified_explanation", "hint", "practice_question")

# Label aliases: acceptable alternative labels for a field, tried only when the
# field's canonical label is absent. Used so SIMPLIFY responses still populate
# the mode's primary field when the model writes "Explanation:" instead of
# "Simplified explanation:" (never leak: HINT declares neither field).
_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "simplified_explanation": ("simplified explanation", "explanation"),
}


def strategy_fields(
    response: TutorResponse,
    answer: str,
    strategy: ModeStrategy,
) -> TutorResponse:
    """Fold the LLM's answer into the structured response fields.

    Preferred path: the answer follows the labeled-section format requested by
    the prompt, and each label maps to its own TutorResponse field (no content
    duplicated across fields, no nulls when the model provided the section).
    Fallback path (model ignored the labels): the mode-driven heuristic below.
    """
    result = response.model_copy()
    fields = strategy.fields
    sections, preamble = parse_sections(answer)

    if not sections:
        return _heuristic_fields(result, answer, fields)

    # Primary content: declared text field with a matching section (canonical
    # label first, then aliases), else the preamble (e.g. a model that wrote
    # one intro sentence before "Key points:").
    primary = next((f for f in fields if f in _MAIN_TEXT_FIELDS), None)
    if primary is not None:
        primary_labels = (primary.replace("_", " "),) + _LABEL_ALIASES.get(primary, ())
        label_hit = next((lab for lab in primary_labels if lab in sections), None)
        if label_hit:
            setattr(result, primary, sections[label_hit])
        elif preamble:
            setattr(result, primary, preamble)

    # Remaining declared fields.
    for field_name in fields:
        if field_name == primary or field_name == "key_points":
            continue
        labels = (field_name.replace("_", " "),) + _LABEL_ALIASES.get(field_name, ())
        label_hit = next((lab for lab in labels if lab in sections), None)
        if label_hit:
            setattr(result, field_name, sections[label_hit])

    if "key_points" in fields and "key points" in sections:
        points = _extract_points(sections["key points"])
        result.key_points = points or [sections["key points"]]
    return result


def _heuristic_fields(
    result: TutorResponse,
    answer: str,
    fields: tuple[str, ...],
) -> TutorResponse:
    """Legacy fold: assign purposeful slices without labeled sections.

    key_points are extracted from bullet/numbered lines when present. The
    first configured text field carries the main answer text; question-asking
    modes (practice/viva) treat the whole answer as the question.
    """
    if "key_points" in fields:
        points = _extract_points(answer)
        result.key_points = points or None
        prose = _strip_points(answer)
    else:
        prose = answer

    if "practice_question" in fields:
        result.practice_question = prose.strip() or None
    else:
        text_fields = [f for f in fields if f in _MAIN_TEXT_FIELDS]
        if text_fields:
            setattr(result, text_fields[0], prose.strip() or None)
    return result


def _extract_points(answer: str) -> list[str]:
    """Extract bullet/numbered key points from an LLM answer, if any."""
    points: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        match = re.match(r"^(?:[-*•]|\d+[.)])\s+(.{8,})$", stripped)
        if match:
            points.append(match.group(1).strip())
    return points


def _strip_points(answer: str) -> str:
    """Answer text with extracted bullet/numbered lines removed."""
    kept = [
        line
        for line in answer.splitlines()
        if not re.match(r"^(?:[-*•]|\d+[.)])\s+", line.strip())
    ]
    return "\n".join(kept).strip()
