"""Lightweight answer evaluation: deterministic rules first, LLM only if needed.

Honesty notes:
- Rule-based evaluation checks the student answer for the reference points'
  key vocabulary. It is intentionally conservative: when the rules cannot
  decide, the verdict is partially_correct and the LLM judge is consulted.
- LLM evaluation is NOT perfect; its verdict is parsed strictly and any
  malformed output falls back to the rule-based result. The feedback is meant
  to teach what was missing rather than just say "wrong".
"""

from __future__ import annotations

import re

from backend.app.llm.client import LLMClient
from backend.app.tutoring.prompts import build_evaluation_prompt
from backend.app.tutoring.schemas import (
    AnswerEvaluation,
    EvaluateAnswerRequest,
    EvaluationVerdict,
)

# Words shorter than this are ignored when matching reference vocabulary.
_MIN_TOKEN_LEN = 4
_STOPWORDS = {
    "that", "this", "with", "from", "they", "them", "have", "been",
    "were", "when", "which", "what", "each", "into", "than", "then",
    "must", "such", "also", "some", "only", "more", "most", "very",
    "over", "under", "between", "because", "while", "where", "there",
}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {
        w
        for w in words
        if len(w) >= _MIN_TOKEN_LEN and w not in _STOPWORDS
    }


def _rule_based_verdict(
    student_answer: str,
    reference_points: list[str],
) -> tuple[EvaluationVerdict, list[str]]:
    """Return (verdict, list of reference points considered missed).

    A reference point is 'covered' when enough of its distinctive vocabulary
    appears in the student answer. Conservative: ambiguous cases resolve to
    partially_correct.
    """
    answer_tokens = _tokens(student_answer)
    if not answer_tokens or not reference_points:
        return EvaluationVerdict.PARTIALLY_CORRECT, list(reference_points)

    covered: list[str] = []
    missed: list[str] = []
    for point in reference_points:
        point_tokens = _tokens(point)
        if not point_tokens:
            covered.append(point)  # nothing distinctive to check
            continue
        overlap = point_tokens & answer_tokens
        # The point counts as covered when at least half of its distinctive
        # vocabulary (and at least one token) appears in the answer.
        if overlap and len(overlap) >= max(1, len(point_tokens) // 2):
            covered.append(point)
        else:
            missed.append(point)

    if not missed:
        return EvaluationVerdict.CORRECT, []
    if covered:
        return EvaluationVerdict.PARTIALLY_CORRECT, missed
    return EvaluationVerdict.INCORRECT, missed


def _parse_llm_evaluation(raw: str) -> tuple[EvaluationVerdict, str] | None:
    """Parse 'VERDICT: ...\\nFEEDBACK: ...'; None if malformed."""
    verdict_match = re.search(
        r"VERDICT:\s*(correct|partially_correct|incorrect)", raw, re.IGNORECASE
    )
    if not verdict_match:
        return None
    verdict = EvaluationVerdict(verdict_match.group(1).lower())
    feedback_match = re.search(r"FEEDBACK:\s*(.+)", raw, re.DOTALL | re.IGNORECASE)
    feedback = feedback_match.group(1).strip() if feedback_match else ""
    return verdict, feedback


def evaluate_answer(
    request: EvaluateAnswerRequest,
    llm: LLMClient,
) -> AnswerEvaluation:
    """Evaluate a student's answer and return a teaching-oriented verdict.

    Flow: rule-based check first; if it yields a definite verdict on all
    points, that result stands (deterministic, explainable). Otherwise the
    LLM judge is consulted; malformed LLM output falls back to the rule-based
    verdict. The mastery delta is computed by the caller (engine) via the
    student model; here we only report the rule that WILL apply.
    """
    verdict, missed = _rule_based_verdict(request.student_answer, request.reference_points)
    feedback_parts: list[str] = []

    if verdict is EvaluationVerdict.PARTIALLY_CORRECT:
        # Ask the LLM judge for a sharper decision; fall back silently.
        system, user = build_evaluation_prompt(
            request.question, request.student_answer, request.reference_points
        )
        try:
            parsed = _parse_llm_evaluation(llm.generate(system, user))
        except Exception:  # noqa: BLE001 - evaluation must never break the API
            parsed = None
        if parsed is not None:
            verdict, feedback = parsed
            missed = [] if verdict is EvaluationVerdict.CORRECT else missed
            feedback_parts.append(feedback)

    if not feedback_parts:
        if verdict is EvaluationVerdict.CORRECT:
            feedback_parts.append(
                "Correct - you covered the key ideas for this question."
            )
        elif missed:
            expected = "; ".join(missed[:2])
            feedback_parts.append(
                "Your answer is missing at least one key idea: consider "
                f"{expected}. Review the sources and try rephrasing."
            )
        else:
            feedback_parts.append(
                "Not correct yet. Re-read the explanation and focus on the "
                "core definition before trying again."
            )

    # Mastery delta the student model will apply for this verdict (mirrors
    # TopicMastery.apply_result so the API response is transparent).
    from backend.app.core.config import settings

    if verdict is EvaluationVerdict.CORRECT:
        delta = settings.tutor_mastery_gain_correct
    elif verdict is EvaluationVerdict.INCORRECT:
        delta = -settings.tutor_mastery_drop_incorrect
    else:
        delta = 0.0

    return AnswerEvaluation(
        verdict=verdict,
        feedback=" ".join(part for part in feedback_parts if part),
        detected_topic=request.topic,
        mastery_update=round(delta, 4),
    )
