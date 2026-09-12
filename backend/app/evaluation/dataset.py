"""Labeled evaluation dataset for OSTutorAI retrieval + grounding decisions.

Schema (also documented in data/evaluation/os_retrieval_eval.json):

    id                  unique question id
    question            the student question text
    topic               OS topic or "non-os"
    expected_relevance  "relevant" | "borderline" | "irrelevant"
                        OS-domain judgment, stable across corpora
    corpus_support      "covered" | "partial" | "absent"
                        whether the CURRENT corpus contains supporting content
    expected_source     optional; source file expected to contain the answer
    note                free-text labeling rationale (ground-truth provenance)

Ground-truth policy (no fake ground truth):
- A question is a reliable POSITIVE only when relevant AND corpus_support=covered.
- A question is a reliable NEGATIVE only when expected_relevance=irrelevant.
- borderline questions and relevant+absent questions (corpus coverage gaps)
  are excluded from the confusion matrix and reported separately. They are
  deliberately NOT counted as threshold failures because the corpus simply
  lacks the content.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

# Default dataset location, relative to the repository root.
DATASET_PATH = Path("data") / "evaluation" / "os_retrieval_eval.json"

Relevance = Literal["relevant", "borderline", "irrelevant"]
CorpusSupport = Literal["covered", "partial", "absent"]


class EvalQuestion(BaseModel):
    """One labeled evaluation question."""

    id: str
    question: str
    topic: str
    expected_relevance: Relevance
    corpus_support: CorpusSupport
    expected_source: str | None = None
    note: str | None = None

    @field_validator("id", "question", "topic")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()


class EvalDataset(BaseModel):
    """A labeled dataset of evaluation questions with documented semantics."""

    schema_version: int
    description: str
    label_definitions: dict = {}
    questions: list[EvalQuestion]

    @field_validator("questions")
    @classmethod
    def _validate_questions(cls, value: list[EvalQuestion]) -> list[EvalQuestion]:
        if not value:
            raise ValueError("evaluation dataset must contain at least one question")
        ids = [q.id for q in value]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate question ids in dataset: {sorted(duplicates)}")
        return value

    # ---- ground-truth grouping (documented policy, no silent counting) ----

    @property
    def reliable_positives(self) -> list[EvalQuestion]:
        """Questions the system SHOULD accept: relevant and actually covered."""
        return [
            q
            for q in self.questions
            if q.expected_relevance == "relevant" and q.corpus_support == "covered"
        ]

    @property
    def reliable_negatives(self) -> list[EvalQuestion]:
        """Questions the system SHOULD reject: clearly outside OS scope."""
        return [q for q in self.questions if q.expected_relevance == "irrelevant"]

    @property
    def excluded(self) -> list[EvalQuestion]:
        """Questions excluded from P/R/F1: borderline + relevant-but-uncovered."""
        return [
            q
            for q in self.questions
            if q not in self.reliable_positives and q not in self.reliable_negatives
        ]


def load_dataset(path: str | Path | None = None) -> EvalDataset:
    """Load and validate the evaluation dataset from disk."""
    dataset_path = Path(path) if path else DATASET_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    try:
        return EvalDataset.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"Invalid evaluation dataset at {dataset_path}: {exc}") from exc
