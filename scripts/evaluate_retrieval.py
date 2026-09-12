"""OSTutorAI retrieval + grounding-decision evaluation (offline, no LLM calls).

Runs the EXISTING persisted FAISS index through the EXISTING retriever over
the labeled OS evaluation dataset, records top-1 cosine scores, and evaluates
the grounding threshold decision against the dataset labels.

What this measures: retrieval quality and the grounding gate decision.
What this does NOT measure: LLM answer quality (no LLM is called).

Usage (from the repository root, after building the index):
    .venv/Scripts/python.exe scripts/evaluate_retrieval.py            # report only
    .venv/Scripts/python.exe scripts/evaluate_retrieval.py --json out.json

The threshold sweep and recommendation are proposals only. RETRIEVAL_MIN_SCORE
is not changed by this script; configuration changes are made deliberately
with documentation and tests.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.config import settings  # noqa: E402
from backend.app.evaluation.dataset import DATASET_PATH, load_dataset  # noqa: E402
from backend.app.evaluation.metrics import (  # noqa: E402
    ConfusionMatrix,
    SweepRow,
    recommend_threshold,
    sweep_thresholds,
)
from backend.app.rag.retriever import Retriever  # noqa: E402

# Sweep range chosen around the observed score distribution of the toy corpus
# (positives ~0.34-0.69, negatives <= 0.29): it spans well below the weakest
# expected positive score to well above the strongest expected negative.
DEFAULT_SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def collect_scores(
    retriever: Retriever,
    questions,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Retrieve for every question and record top-1 score + provenance.

    Returns (top_scores, per-question debug info). The retrieval algorithm is
    NOT duplicated - the existing Retriever is the only retrieval path.
    """
    scores: dict[str, float] = {}
    details: dict[str, dict] = {}
    for q in questions:
        retrieved = retriever.retrieve(q.question)
        if retrieved:
            top = retrieved[0]
            scores[q.id] = float(top.score)
            details[q.id] = {
                "top_score": float(top.score),
                "top_source": top.metadata.source,
                "top_title": top.metadata.title,
                "all_scores": [round(float(r.score), 4) for r in retrieved],
                "decision_at_current_threshold": (
                    "accept"
                    if not math.isinf(float(top.score))
                    and float(top.score) >= settings.retrieval_min_score
                    else "reject"
                ),
            }
        else:
            # No chunks retrieved: the gate rejects at every threshold.
            scores[q.id] = float("-inf")
            details[q.id] = {
                "top_score": None,
                "top_source": None,
                "top_title": None,
                "all_scores": [],
                "decision_at_current_threshold": "reject",
            }
    return scores, details


def build_report(dataset, scores: dict[str, float], current_threshold: float):
    """Assemble every number the report needs (pure, unit-testable).

    Only reliable pairs enter the confusion matrix and sweep: reliable
    positives (relevant + covered) and reliable negatives (irrelevant).
    Borderline and relevant-but-uncovered questions are excluded by policy;
    they are reported separately with their scores.
    """
    reliable_ids = [q.id for q in dataset.reliable_positives] + [
        q.id for q in dataset.reliable_negatives
    ]
    reliable_scores = {qid: scores[qid] for qid in reliable_ids if qid in scores}
    should_accept = {
        **{q.id: True for q in dataset.reliable_positives},
        **{q.id: False for q in dataset.reliable_negatives},
    }

    current_matrix = ConfusionMatrix.from_decisions(
        [should_accept[qid] for qid in reliable_scores],
        [reliable_scores[qid] >= current_threshold for qid in reliable_scores],
    )

    sweep = sweep_thresholds(reliable_scores, should_accept, DEFAULT_SWEEP)
    recommended, reason = recommend_threshold(
        sweep, fallback=current_threshold, current_matrix=current_matrix
    )

    coverage_gaps = [
        q
        for q in dataset.reliable_positives
        if q.corpus_support == "absent"
    ]
    return {
        "num_questions": len(dataset.questions),
        "reliable_positives": len(dataset.reliable_positives),
        "reliable_negatives": len(dataset.reliable_negatives),
        "excluded_count": len(dataset.excluded),
        "coverage_gaps": [q.id for q in coverage_gaps],
        "current_threshold": current_threshold,
        "current_matrix": current_matrix,
        "sweep": sweep,
        "recommended": recommended,
        "reason": reason,
        "scores": scores,
    }


def print_report(dataset, report: dict) -> None:
    """Human-readable terminal output (Part 13 format)."""
    matrix: ConfusionMatrix = report["current_matrix"]
    current: float = report["current_threshold"]

    print("=" * 64)
    print("OSTutorAI Retrieval Evaluation")
    print("=" * 64)
    print()
    print("Dataset:      OS retrieval evaluation dataset (data/evaluation/os_retrieval_eval.json)")
    print(f"Questions:    {report['num_questions']}")
    print(
        f"  reliable positives (relevant + corpus-covered): {report['reliable_positives']}"
    )
    print(f"  reliable negatives (irrelevant):                {report['reliable_negatives']}")
    print(
        f"  excluded from P/R/F1 (borderline + uncovered):  {report['excluded_count']}"
    )
    print()
    print("Note: retrieval scores are cosine similarity in [-1, 1] -")
    print("      NOT percentages, probabilities, or confidence values.")
    print()
    print(f"Current threshold: {current}")
    print()
    print("Confusion Matrix (at current threshold):")
    print(f"  TP: {matrix.tp}   (should answer, accepted)")
    print(f"  FN: {matrix.fn}   (should answer, rejected)")
    print(f"  FP: {matrix.fp}   (should reject, accepted)")
    print(f"  TN: {matrix.tn}   (should reject, rejected)")
    print()
    print(f"Precision: {matrix.precision:.3f}")
    print(f"Recall:    {matrix.recall:.3f}")
    print(f"F1:        {matrix.f1:.3f}")
    print()
    print("Threshold Sweep (same labeled questions):")
    print("threshold | precision | recall | f1")
    print("-" * 40)
    for row in report["sweep"]:
        print(
            f"{row.threshold:>9.2f} | {row.precision:>9.3f} | "
            f"{row.recall:>6.3f} | {row.f1:>5.3f}"
        )
    print()
    print(f"Recommended baseline: {report['recommended']:.2f}")
    print(f"Reason: {report['reason']}")
    print()
    print(
        "This is an engineering baseline on the CURRENT project-authored OS "
    )
    print(
        "knowledge base (data/raw/*.md). Recalibrate after any corpus change."
    )
    print()
    print("Excluded questions (not counted in the matrix):")
    for q in dataset.excluded:
        score = report["scores"].get(q.id)
        score_str = f"{score:.4f}" if score is not None and not math.isinf(score) else "n/a"
        print(
            f"  [{q.expected_relevance:>10}/{q.corpus_support:>7}] "
            f"{q.id:<14} score={score_str:<7} {q.question[:52]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate OSTutorAI retrieval and grounding threshold decisions."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=f"Path to the evaluation dataset (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        default=None,
        help="Optional path to write the full results as JSON",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    retriever = Retriever.from_prebuilt(settings.index_path)
    print(f"Loaded index: {retriever.vector_store.size} chunks from {settings.index_path}")

    scores, details = collect_scores(retriever, dataset.questions)
    report = build_report(dataset, scores, settings.retrieval_min_score)

    print_report(dataset, report)

    if args.json_out:
        payload = {
            "schema_version": 1,
            "dataset": args.dataset or str(DATASET_PATH),
            "index_dir": str(settings.index_path),
            "embedding_model": settings.embedding_model_name,
            "current_threshold": report["current_threshold"],
            "num_questions": report["num_questions"],
            "reliable_positives": report["reliable_positives"],
            "reliable_negatives": report["reliable_negatives"],
            "excluded_count": report["excluded_count"],
            "confusion_matrix": {
                "tp": report["current_matrix"].tp,
                "fn": report["current_matrix"].fn,
                "fp": report["current_matrix"].fp,
                "tn": report["current_matrix"].tn,
                "precision": report["current_matrix"].precision,
                "recall": report["current_matrix"].recall,
                "f1": report["current_matrix"].f1,
            },
            "threshold_sweep": [
                {
                    "threshold": row.threshold,
                    "tp": row.matrix.tp,
                    "fn": row.matrix.fn,
                    "fp": row.matrix.fp,
                    "tn": row.matrix.tn,
                    "precision": row.precision,
                    "recall": row.recall,
                    "f1": row.f1,
                }
                for row in report["sweep"]
            ],
            "recommended_threshold": report["recommended"],
            "recommendation_reason": report["reason"],
            "per_question": {},
        }
        # Attach per-question details with their labels (explicit loop).
        for q in dataset.questions:
            payload["per_question"][q.id] = {
                **details[q.id],
                "expected_relevance": q.expected_relevance,
                "corpus_support": q.corpus_support,
                "note": q.note,
            }
        Path(args.json_out).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Full results written to {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
