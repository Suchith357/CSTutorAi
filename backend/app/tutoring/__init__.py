"""Tutor Engine and adaptive teaching for OSTutorAI.

Public surface:
- TutorEngine: grounding-gated tutoring turns in 7 teaching modes.
- STUDENT_REGISTRY: in-memory student models (no database by design).
- evaluate_answer: rule-first, LLM-assisted answer evaluation.
"""

from backend.app.tutoring.engine import TutorEngine, detect_topic
from backend.app.tutoring.student_model import STUDENT_REGISTRY, StudentModelRegistry
from backend.app.tutoring.schemas import (
    TutorMode,
    TutorRequest,
    TutorResponse,
    StudentProgressResponse,
)

__all__ = [
    "STUDENT_REGISTRY",
    "StudentModelRegistry",
    "TutorEngine",
    "TutorMode",
    "TutorRequest",
    "TutorResponse",
    "StudentProgressResponse",
    "detect_topic",
]
