"""Tutoring benchmark: deterministic checks over the Tutor Engine.

What this measures (and what it deliberately does NOT claim):
- grounding decision accuracy vs the dataset's should_ground labels;
- deterministic phrase checks (should_contain / must_not_contain) on generated
  teaching text - must_not_contain encodes the known Qwen Coffman-conditions
  confusion so regressions are caught mechanically;
- optional rule-based reference-point coverage of generated explanations
  (vocabulary-overlap proxy for content coverage; requires --llm).

What this does NOT measure automatically: correctness, instructional
usefulness, teaching quality. The script prints full transcripts so a human
can review them; those dimensions are intentionally not automated.

Usage:
    python scripts/evaluate_tutoring.py            # grounding + refusal checks (no LLM needed for refusals; mock LLM used for grounded answers)
    python scripts/evaluate_tutoring.py --llm      # real local Qwen generation (slower)
    python scripts/evaluate_tutoring.py --json     # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.app.llm.client import LLMClient, create_llm_client  # noqa: E402
from backend.app.rag.retriever import Retriever  # noqa: E402
from backend.app.tutoring import TutorEngine, TutorMode  # noqa: E402
from backend.app.tutoring.evaluator import _tokens  # noqa: E402
from backend.app.tutoring.student_model import STUDENT_REGISTRY  # noqa: E402

BENCHMARK_PATH = REPO_ROOT / "data" / "evaluation" / "os_tutoring_benchmark.json"


@dataclass
class CaseResult:
    case_id: str
    topic: str
    mode: str
    grounded: bool
    expected_grounded: bool
    grounding_ok: bool
    phrase_hits: list[str] = field(default_factory=list)
    phrase_misses: list[str] = field(default_factory=list)
    forbidden_hits: list[str] = field(default_factory=list)  # must be empty
    covered_points: int = 0
    total_points: int = 0
    transcript: str = ""


def load_cases() -> list[dict]:
    data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    return data["cases"]


def evaluate_case(
    case: dict,
    engine: TutorEngine,
    student_id: str = "benchmark",
) -> CaseResult:
    student = STUDENT_REGISTRY.get_or_create(student_id)
    mode = TutorMode(case["mode"])
    response = engine.tutor(case["question"], mode, student)

    text_parts = [
        part
        for part in (
            response.explanation,
            response.simplified_explanation,
            response.hint,
            response.practice_question,
            " ".join(response.key_points or []),
        )
        if part
    ]
    generated = "\n".join(text_parts).lower()

    grounding_ok = response.grounded == case["should_ground"]
    phrase_hits = [p for p in case.get("should_contain", []) if p.lower() in generated]
    phrase_misses = [p for p in case.get("should_contain", []) if p.lower() not in generated]
    forbidden_hits = [p for p in case.get("must_not_contain", []) if p.lower() in generated]

    covered = total = 0
    if case["should_ground"] and response.grounded and case.get("reference_points"):
        answer_tokens = _tokens(generated)
        for point in case["reference_points"]:
            total += 1
            point_tokens = _tokens(point)
            overlap = point_tokens & answer_tokens
            if overlap and len(overlap) >= max(1, len(point_tokens) // 2):
                covered += 1

    transcript_lines = [
        f"[{case['id']}] mode={mode.value} topic={response.topic} "
        f"grounded={response.grounded} difficulty={response.difficulty}",
        f"Q: {case['question']}",
    ]
    for part in text_parts:
        transcript_lines.append(f"A: {part[:400]}")
    if response.refusal_reason:
        transcript_lines.append(f"refusal: {response.refusal_reason}")
    transcript_lines.append(
        "sources: " + ", ".join(f"{s.metadata.topic} ({s.score:.3f})" for s in response.sources)
    )

    return CaseResult(
        case_id=case["id"],
        topic=case["topic"],
        mode=case["mode"],
        grounded=response.grounded,
        expected_grounded=case["should_ground"],
        grounding_ok=grounding_ok,
        phrase_hits=phrase_hits,
        phrase_misses=phrase_misses,
        forbidden_hits=forbidden_hits,
        covered_points=covered,
        total_points=total,
        transcript="\n".join(transcript_lines),
    )


def build_report(results: list[CaseResult]) -> dict:
    total = len(results)
    grounding_ok = sum(r.grounding_ok for r in results)
    grounded_cases = [r for r in results if r.expected_grounded]
    refusal_cases = [r for r in results if not r.expected_grounded]
    forbidden_violations = [r for r in results if r.forbidden_hits]
    point_total = sum(r.total_points for r in results)
    point_covered = sum(r.covered_points for r in results)

    return {
        "total_cases": total,
        "grounding_decisions_ok": grounding_ok,
        "grounding_accuracy": round(grounding_ok / total, 4) if total else 0.0,
        "grounded_cases": len(grounded_cases),
        "refusal_cases": len(refusal_cases),
        "refusals_correct": sum(1 for r in refusal_cases if r.grounding_ok),
        "phrase_checks_passed": sum(
            1 for r in results if not r.phrase_misses and not r.forbidden_hits
        ),
        "forbidden_phrase_violations": len(forbidden_violations),
        "reference_point_coverage": round(point_covered / point_total, 4)
        if point_total
        else None,
        "reference_points_total": point_total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true", help="use the real local LLM")
    parser.add_argument("--json", action="store_true", help="print JSON report")
    parser.add_argument("--verbose", action="store_true", help="print transcripts")
    args = parser.parse_args()

    cases = load_cases()
    retriever = Retriever.from_prebuilt()
    llm: LLMClient = create_llm_client("transformers" if args.llm else "mock")
    engine = TutorEngine(retriever, llm)

    results = [evaluate_case(case, engine) for case in cases]
    report = build_report(results)

    if args.json:
        print(
            json.dumps(
                {
                    "report": report,
                    "cases": [
                        {
                            "id": r.case_id,
                            "topic": r.topic,
                            "grounded": r.grounded,
                            "expected_grounded": r.expected_grounded,
                            "grounding_ok": r.grounding_ok,
                            "phrase_misses": r.phrase_misses,
                            "forbidden_hits": r.forbidden_hits,
                            "covered_points": r.covered_points,
                            "total_points": r.total_points,
                        }
                        for r in results
                    ],
                },
                indent=2,
            )
        )
    else:
        print("OSTutorAI Tutoring Benchmark")
        print("============================")
        print(f"Cases: {report['total_cases']}")
        print(
            "Grounding decisions: "
            f"{report['grounding_decisions_ok']}/{report['total_cases']} "
            f"(accuracy {report['grounding_accuracy']:.3f})"
        )
        print(f"  grounded (should answer): {report['grounded_cases']}")
        print(
            f"  refusals (should refuse): {report['refusals_correct']}/"
            f"{report['refusal_cases']} correct"
        )
        print(f"Phrase checks passed: {report['phrase_checks_passed']}/{report['total_cases']}")
        print(
            "Forbidden-phrase violations (e.g. Coffman confusion): "
            f"{report['forbidden_phrase_violations']}"
        )
        if report["reference_point_coverage"] is not None:
            print(
                "Reference-point coverage (vocabulary proxy, NOT correctness): "
                f"{report['reference_point_coverage']:.3f} "
                f"({report['reference_points_total']} points)"
            )
        print()
        print("Note: correctness / usefulness / teaching quality require human")
        print("review of the transcripts below; they are not automated.")

    if args.verbose or not args.json:
        print()
        print("-" * 60)
        for r in results:
            print(r.transcript)
            print("-" * 60)

    return 0 if report["forbidden_phrase_violations"] == 0 and report["grounding_accuracy"] >= 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
