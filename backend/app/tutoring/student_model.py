"""Lightweight in-memory Student Model (no database, per project scope).

Design:
- One StudentModel per student_id, held in a process-wide registry.
- Per-topic mastery in [0, 1], updated by a transparent rule:
    correct   -> mastery += settings.tutor_mastery_gain_correct
    incorrect -> mastery -= settings.tutor_mastery_drop_incorrect
    hint used -> tracked (hints_used counter), never scored by itself
- Difficulty is a pure function of mastery via explicit, configurable
  thresholds (no randomness, no ML).
- Thread-safe via a lock, since FastAPI may serve requests concurrently.

This is deliberately explainable and academically defensible: every mastery
value in the progress endpoint can be traced to counted events.
"""

from __future__ import annotations

import threading

from backend.app.core.config import settings
from backend.app.tutoring.schemas import (
    DIFFICULTY_BEGINNER,
    DIFFICULTY_BASIC,
    DIFFICULTY_INTERMEDIATE,
    DIFFICULTY_ADVANCED,
    DIFFICULTY_EXAM,
    EvaluationVerdict,
    MasterySnapshot,
)


def difficulty_for_mastery(mastery: float) -> int:
    """Map mastery [0,1] to difficulty 1..5 using explicit thresholds.

    Thresholds come from settings so they are configurable and documented:
      mastery < low          -> 1 (beginner)
      mastery < medium       -> 2 (basic)
      mastery < high         -> 3 (intermediate)
      mastery < exam         -> 4 (advanced)
      otherwise              -> 5 (exam/viva)
    """
    if mastery < settings.tutor_difficulty_low_threshold:
        return DIFFICULTY_BEGINNER
    if mastery < settings.tutor_difficulty_medium_threshold:
        return DIFFICULTY_BASIC
    if mastery < settings.tutor_difficulty_high_threshold:
        return DIFFICULTY_INTERMEDIATE
    if mastery < settings.tutor_difficulty_exam_threshold:
        return DIFFICULTY_ADVANCED
    return DIFFICULTY_EXAM


class TopicMastery:
    """Learning state for one student on one OS topic."""

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.questions_attempted = 0
        self.questions_correct = 0
        self.questions_incorrect = 0
        self.hints_used = 0
        self.estimated_mastery = 0.0
        self.recent_performance: list[EvaluationVerdict] = []

    def apply_result(
        self,
        verdict: EvaluationVerdict,
        hint_used: bool,
    ) -> float:
        """Apply one evaluated attempt; return the mastery delta applied."""
        self.questions_attempted += 1
        if hint_used:
            self.hints_used += 1

        delta = 0.0
        if verdict == EvaluationVerdict.CORRECT:
            self.questions_correct += 1
            delta = settings.tutor_mastery_gain_correct
        elif verdict == EvaluationVerdict.INCORRECT:
            self.questions_incorrect += 1
            delta = -settings.tutor_mastery_drop_incorrect
        # partially_correct: attempted and tracked, mastery unchanged.

        self.estimated_mastery = min(1.0, max(0.0, self.estimated_mastery + delta))
        self.recent_performance.append(verdict)
        limit = settings.tutor_recent_performance_limit
        if len(self.recent_performance) > limit:
            self.recent_performance = self.recent_performance[-limit:]
        return delta

    def current_difficulty(self) -> int:
        return difficulty_for_mastery(self.estimated_mastery)

    def snapshot(self) -> MasterySnapshot:
        return MasterySnapshot(
            topic=self.topic,
            questions_attempted=self.questions_attempted,
            questions_correct=self.questions_correct,
            questions_incorrect=self.questions_incorrect,
            hints_used=self.hints_used,
            current_difficulty=self.current_difficulty(),
            estimated_mastery=round(self.estimated_mastery, 4),
            recent_performance=list(self.recent_performance),
        )


class StudentModel:
    """All topic mastery for one student."""

    def __init__(self, student_id: str) -> None:
        self.student_id = student_id
        self._topics: dict[str, TopicMastery] = {}
        self._lock = threading.Lock()

    def topic(self, topic: str) -> TopicMastery:
        """Get-or-create the mastery record for a topic."""
        with self._lock:
            return self._topics.setdefault(topic, TopicMastery(topic))

    def get(self, topic: str) -> TopicMastery | None:
        with self._lock:
            return self._topics.get(topic)

    def difficulty(self, topic: str) -> int:
        """Current difficulty for a topic (beginner if never attempted)."""
        record = self.get(topic)
        if record is None:
            return DIFFICULTY_BEGINNER
        return record.current_difficulty()

    def progress(self) -> list[MasterySnapshot]:
        with self._lock:
            records = list(self._topics.values())
        return [r.snapshot() for r in records]

    def totals(self) -> tuple[int, int]:
        """(overall_attempts, overall_correct) across all topics."""
        with self._lock:
            attempted = sum(r.questions_attempted for r in self._topics.values())
            correct = sum(r.questions_correct for r in self._topics.values())
        return attempted, correct


class StudentModelRegistry:
    """Process-wide registry: student_id -> StudentModel."""

    def __init__(self) -> None:
        self._students: dict[str, StudentModel] = {}
        self._lock = threading.Lock()

    def get_or_create(self, student_id: str) -> StudentModel:
        with self._lock:
            model = self._students.get(student_id)
            if model is None:
                model = StudentModel(student_id)
                self._students[student_id] = model
            return model

    def reset(self, student_id: str | None = None) -> None:
        """Clear one student or the whole registry (used by tests)."""
        with self._lock:
            if student_id is None:
                self._students.clear()
            else:
                self._students.pop(student_id, None)


# Process-wide registry used by the API layer.
STUDENT_REGISTRY = StudentModelRegistry()
