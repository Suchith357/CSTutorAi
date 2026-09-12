"""Retrieval/grounding decision metrics for OSTutorAI evaluation.

These are RETRIEVAL decision metrics, not "LLM accuracy". The prediction is
the grounding gate's decision at a given threshold: accept (score >= t and at
least one chunk retrieved) or reject.

All rates return 0.0 when their denominators are zero (well-defined for an
empty split), so sweeps over tiny controlled examples never crash.
"""

from dataclasses import dataclass


@dataclass
class ConfusionMatrix:
    """Binary decision matrix for the grounding gate.

    Prediction = accept the retrieval as sufficient.
    Positive class = questions that should be answered (reliable positives).
    """

    tp: int = 0  # should answer, gate accepts
    fn: int = 0  # should answer, gate rejects
    fp: int = 0  # should reject, gate accepts
    tn: int = 0  # should reject, gate rejects

    @property
    def total(self) -> int:
        return self.tp + self.fn + self.fp + self.tn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    @classmethod
    def from_decisions(
        cls,
        should_accept: list[bool],
        accepted: list[bool],
    ) -> "ConfusionMatrix":
        """Build the matrix from paired ground truth / gate decisions."""
        if len(should_accept) != len(accepted):
            raise ValueError("decision lists must have equal length")
        matrix = cls()
        for truth, predicted in zip(should_accept, accepted):
            if truth and predicted:
                matrix.tp += 1
            elif truth and not predicted:
                matrix.fn += 1
            elif not truth and predicted:
                matrix.fp += 1
            else:
                matrix.tn += 1
        return matrix


@dataclass
class SweepRow:
    """One threshold's outcome in the sweep."""

    threshold: float
    matrix: ConfusionMatrix

    @property
    def precision(self) -> float:
        return self.matrix.precision

    @property
    def recall(self) -> float:
        return self.matrix.recall

    @property
    def f1(self) -> float:
        return self.matrix.f1


def sweep_thresholds(
    scores: dict[str, float],
    should_accept: dict[str, bool],
    thresholds: list[float],
) -> list[SweepRow]:
    """Evaluate the gate decision for each threshold.

    scores: question id -> best retrieved score (empty retrieval => -inf so
            the gate rejects it at every threshold).
    should_accept: question id -> ground truth ("should be answered").
    """
    rows: list[SweepRow] = []
    for threshold in thresholds:
        accepted = {
            qid: score >= threshold for qid, score in scores.items()
        }
        matrix = ConfusionMatrix.from_decisions(
            [should_accept[qid] for qid in scores],
            [accepted[qid] for qid in scores],
        )
        rows.append(SweepRow(threshold=threshold, matrix=matrix))
    return rows


def recommend_threshold(
    rows: list[SweepRow],
    fallback: float,
    current_matrix: ConfusionMatrix | None = None,
) -> tuple[float, str]:
    """Pick the best threshold from a sweep and explain the choice.

    Selection policy (tutoring gate: a false ACCEPT is the worst failure
    mode, because it produces confident answers without source support):

    1. Prefer thresholds with PERFECT precision (no false accepts). Among
       them, take the best F1; ties go to the loosest (highest recall).
    2. If no threshold reaches precision 1.0, fall back to the best F1 among
       thresholds with precision >= 0.9.
    3. If none reach 0.9, propose the best-F1 value but flag that more
       labeled data is required.
    4. If the resulting candidate's confusion matrix is IDENTICAL to the
       currently configured threshold's matrix, keep the current value:
       configuration should not churn without a measurable improvement.

    Returns (threshold, reason).
    """
    if not rows:
        return fallback, "no sweep rows available"

    best_f1 = max(rows, key=lambda r: (round(r.f1, 6), r.matrix.tp))

    perfect = [r for r in rows if r.matrix.precision >= 1.0 and r.matrix.f1 > 0]
    trusted = [r for r in rows if r.matrix.precision >= 0.9 and r.matrix.f1 > 0]

    if perfect:
        candidate = max(perfect, key=lambda r: (round(r.f1, 6), r.recall))
        if best_f1.matrix.precision < 1.0:
            context = (
                f"highest-F1 sweep value {best_f1.threshold:.2f} "
                f"(F1={best_f1.f1:.3f}) has precision {best_f1.precision:.3f}, "
                "i.e. it accepts at least one irrelevant question; the selected "
                "value keeps precision at 1.0"
            )
        else:
            context = (
                f"highest-F1 sweep value {best_f1.threshold:.2f} also has "
                "precision 1.0; the selected value prefers the loosest "
                "(highest-recall) threshold that keeps precision at 1.0"
            )
    elif trusted:
        candidate = max(trusted, key=lambda r: (round(r.f1, 6), r.recall))
        context = (
            f"no threshold reaches precision 1.0; selected the best F1 among "
            f"precision >= 0.9 values"
        )
    else:
        if current_matrix is not None:
            return (
                fallback,
                f"no threshold on this dataset reaches precision 0.9 (best F1 "
                f"{best_f1.f1:.3f} at {best_f1.threshold:.2f} with precision "
                f"{best_f1.precision:.3f}); more labeled data is required before "
                "changing the configured threshold",
            )
        return (
            best_f1.threshold,
            f"best F1 ({best_f1.f1:.3f}) but precision {best_f1.precision:.3f} "
            "(< 0.9); more labeled data is required",
        )

    # Policy 4: do not churn the configuration without measurable improvement.
    if current_matrix is not None and (
        (candidate.matrix.tp, candidate.matrix.fn, candidate.matrix.fp, candidate.matrix.tn)
        == (current_matrix.tp, current_matrix.fn, current_matrix.fp, current_matrix.tn)
    ):
        return (
            fallback,
            f"sweep candidate {candidate.threshold:.2f} makes exactly the same "
            f"decisions as the current {fallback:.2f} "
            f"(TP={current_matrix.tp}, FN={current_matrix.fn}, "
            f"FP={current_matrix.fp}, TN={current_matrix.tn}); {context}; "
            "no strictly better matrix exists in the sweep, so the current "
            "threshold is kept (more labeled data required before recalibration)",
        )

    return candidate.threshold, (
        f"{context} (P={candidate.precision:.3f}, R={candidate.recall:.3f}, "
        f"F1={candidate.f1:.3f})"
    )
