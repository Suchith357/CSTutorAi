"""Pydantic schemas for the Tutor Engine.

Design notes:
- Every tutoring response is structured, not one giant string. Which fields
  are populated depends on the teaching mode; optional fields default to None.
- Sources are always the actual retrieved chunks (scores + provenance), so the
  student can always see WHY the tutor answered and from where.
- Difficulty is an integer scale 1..5 (1 beginner .. 5 exam/viva).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from backend.app.rag.schemas import RetrievedChunk

# Difficulty scale, mirrored in settings thresholds.
DIFFICULTY_BEGINNER = 1
DIFFICULTY_BASIC = 2
DIFFICULTY_INTERMEDIATE = 3
DIFFICULTY_ADVANCED = 4
DIFFICULTY_EXAM = 5
DIFFICULTY_MIN = DIFFICULTY_BEGINNER
DIFFICULTY_MAX = DIFFICULTY_EXAM


class TutorMode(str, Enum):
    """Teaching modes supported by the Tutor Engine."""

    EXPLAIN = "explain"
    SIMPLIFY = "simplify"
    EXAMPLE = "example"
    HINT = "hint"
    PRACTICE = "practice"
    VIVA = "viva"
    EXAM = "exam"


class EvaluationVerdict(str, Enum):
    """Outcome of evaluating a student's answer."""

    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"


class TutorRequest(BaseModel):
    """POST /tutor request body."""

    question: str
    mode: TutorMode = TutorMode.EXPLAIN
    # Simple identifier selecting the in-memory student model. NOT auth.
    student_id: str = "demo"

    @field_validator("question")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        return value


class MasterySnapshot(BaseModel):
    """Per-topic learning state for one student."""

    topic: str
    questions_attempted: int = 0
    questions_correct: int = 0
    questions_incorrect: int = 0
    hints_used: int = 0
    current_difficulty: int = DIFFICULTY_BEGINNER
    estimated_mastery: float = Field(ge=0.0, le=1.0, default=0.0)
    recent_performance: list[EvaluationVerdict] = Field(default_factory=list)


class StudentProgressResponse(BaseModel):
    """GET /tutor/progress/{student_id} response."""

    student_id: str
    topics: list[MasterySnapshot]
    overall_attempts: int = 0
    overall_correct: int = 0


class TutorResponse(BaseModel):
    """Structured tutoring response (fields populated per mode).

    Always present: question, mode, topic, difficulty, sources, grounded.
    Mode-dependent: explanation, simplified_explanation, example, key_points,
    hint, check_question, common_mistake, practice_question, evaluation.
    """

    question: str
    mode: TutorMode
    # OS topic inferred from retrieval metadata (never an ML classifier).
    topic: str
    difficulty: int = Field(ge=DIFFICULTY_MIN, le=DIFFICULTY_MAX)
    sources: list[RetrievedChunk]
    grounded: bool

    # Teaching content (mode-dependent).
    explanation: str | None = None
    simplified_explanation: str | None = None
    example: str | None = None
    key_points: list[str] | None = None
    hint: str | None = None
    check_question: str | None = None
    common_mistake: str | None = None
    practice_question: str | None = None

    # Populated only by the evaluate-answer flow.
    evaluation: AnswerEvaluation | None = None

    # Only populated when grounded=False (safe refusal text).
    refusal_reason: str | None = None


class AnswerEvaluation(BaseModel):
    """Result of evaluating a student's answer to a check/practice question."""

    verdict: EvaluationVerdict
    # Feedback that TEACHES: says what was missing, not just "wrong".
    feedback: str
    detected_topic: str
    # Mastery delta that was applied to the student model (transparent).
    mastery_update: float


class EvaluateAnswerRequest(BaseModel):
    """POST /tutor/evaluate request body."""

    student_id: str = "demo"
    topic: str
    question: str
    student_answer: str = Field(min_length=1)
    # Optional retrieved context used to judge the answer (from the original
    # tutoring turn). When omitted, evaluation falls back to rule-based checks.
    reference_points: list[str] = Field(default_factory=list)
    hint_used: bool = False

    @field_validator("student_answer")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("student_answer must not be empty")
        return value
