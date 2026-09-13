"""Live verification of the upgraded local LLM (Qwen2.5-3B-Instruct) against
the real retrieval + Tutor Engine pipeline.

Exercises:
- the 7 required quality questions (process, deadlock, FCFS, threads,
  semaphore, virtual memory, Banker's algorithm),
- all seven tutoring modes once each,
- one complete adaptive-learning interaction (incorrect -> correct),
- the progress endpoint state.

Usage:
    .venv/Scripts/python.exe -X utf8 scripts/live_model_check.py
    .venv/Scripts/python.exe -X utf8 scripts/live_model_check.py --modes-only
    .venv/Scripts/python.exe -X utf8 scripts/live_model_check.py --adaptive-only

Reads data/evaluation/os_tutoring_benchmark.json for reference points used by
the evaluator in the adaptive interaction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.llm.client import TransformersLLMClient, create_llm_client  # noqa: E402
from backend.app.rag.retriever import Retriever  # noqa: E402
from backend.app.tutoring.engine import TutorEngine  # noqa: E402
from backend.app.tutoring.schemas import TutorMode  # noqa: E402
from backend.app.tutoring.student_model import STUDENT_REGISTRY  # noqa: E402

# The 7 required quality questions. `checks` are substring guards for known
# failure modes (case-insensitive); `expect_grounded` gates on the grounding
# gate, not on generation.
QUALITY_QUESTIONS = [
    {
        "name": "PROCESS",
        "question": "What is a process in an operating system? Explain it to me like a beginner.",
        "mode": "explain",
        "expect_grounded": True,
        "forbidden": ["CFG", "control-flow graph", "control flow graph",
                      "passively executable file", "passive executable"],
        # Concept-level: program-in-execution idea (any correct phrasing).
        "checks": ["program"],
    },
    {
        "name": "DEADLOCK",
        "question": "Explain deadlock to me like I am a beginner.",
        "mode": "explain",
        "expect_grounded": True,
        # The invalid example pattern: each process holding its own copy.
        "forbidden": ["separate copy", "separate copies", "own copy"],
        "checks": [],
    },
    {
        "name": "FCFS",
        "question": "Explain FCFS CPU scheduling with a simple example.",
        "mode": "example",
        "expect_grounded": True,
        "forbidden": [
            "geometric distribution", "probability", "random",
            "shortest job", "SJF",
        ],
        "checks": ["arrival", "first"],
    },
    {
        "name": "THREADS",
        "question": "What is the difference between a process and a thread? Explain simply.",
        "mode": "explain",
        "expect_grounded": True,
        # The specific measured contradiction is forbidden; sharing must appear.
        "forbidden": ["every thread has its own address space",
                      "each thread has its own address space"],
        "checks": ["thread", "share"],
    },
    {
        "name": "SEMAPHORE",
        "question": "What is a semaphore? Explain it like I am a beginner.",
        "mode": "simplify",
        "expect_grounded": True,
        "forbidden": ["disk scheduling", "SCAN", "geometric"],
        "checks": [],
    },
    {
        "name": "VIRTUAL MEMORY",
        "question": "Explain virtual memory using a simple real-world analogy.",
        "mode": "example",
        "expect_grounded": True,
        "forbidden": ["deadlock"],
        "checks": [],
    },
    {
        "name": "BANKER'S ALGORITHM",
        "question": "Explain Banker's Algorithm in operating systems with a small example.",
        "mode": "explain",
        "expect_grounded": True,
        "forbidden": ["geometric distribution", "probability"],
        "checks": ["safe"],
    },
]

ALL_MODES = [
    ("explain", "What is CPU scheduling?"),
    # Long-form phrasing measured above the grounding gate (short forms like
    # "What is thrashing?" legitimately fall under 0.42).
    ("simplify", "What is thrashing in operating systems?"),
    ("example", "Give an example of a race condition in a critical section."),
    ("hint", "What makes a deadlock possible?"),
    ("practice", "Topic: page replacement algorithms."),
    ("viva", "Topic: process states."),
    ("exam", "What is virtual memory?"),
]


def _show(resp_dict: dict, verbose: bool) -> None:
    mode = resp_dict.get("mode")
    print(
        f"  grounded={resp_dict['grounded']} topic={resp_dict['topic']} "
        f"difficulty={resp_dict['difficulty']} mode={mode}"
    )
    explanation = resp_dict.get("explanation") or resp_dict.get("simplified_explanation") or ""
    print(f"  main text: {len(explanation)} chars, {len(explanation.split())} words")
    for field in ("key_points", "example", "hint", "check_question",
                  "practice_question", "common_mistake"):
        value = resp_dict.get(field)
        if value:
            shown = value if isinstance(value, str) else "; ".join(value)
            print(f"  {field}: {shown[:160]}{'...' if len(shown) > 160 else ''}")
    if verbose:
        print("  --- full main text ---")
        print(explanation)
    if resp_dict.get("sources"):
        top = resp_dict["sources"][0]
        print(f"  top source: {top['metadata']['title']} (score {top['score']:.4f})")


def run_quality_questions(engine, verbose: bool) -> bool:
    print("=" * 70)
    print("PART A - 7 QUALITY QUESTIONS (real model)")
    print("=" * 70)
    student = STUDENT_REGISTRY.get_or_create("live-quality")
    all_ok = True
    for spec in QUALITY_QUESTIONS:
        print(f"\n[{spec['name']}] {spec['question']}")
        resp = engine.tutor(
            spec["question"], TutorMode(spec["mode"]), student
        ).model_dump()
        _show(resp, verbose)

        problems = []
        if spec["expect_grounded"] and not resp["grounded"]:
            problems.append("expected grounded=True but gate refused")
        blob = json.dumps(
            {k: v for k, v in resp.items() if k != "sources"},
            ensure_ascii=False,
        ).lower()
        for phrase in spec["forbidden"]:
            if phrase.lower() in blob:
                problems.append(f"forbidden phrase present: {phrase!r}")
        for phrase in spec["checks"]:
            if phrase.lower() not in blob:
                problems.append(f"expected phrase missing: {phrase!r}")
        if problems:
            all_ok = False
            for p in problems:
                print(f"  !! {p}")
        else:
            print("  OK")
    return all_ok


def run_all_modes(engine, verbose: bool) -> bool:
    print("\n" + "=" * 70)
    print("PART B - ALL 7 MODES (real model)")
    print("=" * 70)
    student = STUDENT_REGISTRY.get_or_create("live-modes")
    all_ok = True
    for mode_name, question in ALL_MODES:
        print(f"\n[{mode_name.upper()}] {question}")
        resp = engine.tutor(question, TutorMode(mode_name), student).model_dump()
        _show(resp, verbose)
        if not resp["grounded"]:
            problems = ["gate refused (grounded=False)"]
            all_ok = False
            for p in problems:
                print(f"  !! {p}")
            continue
        # Mode-respect: the mode's primary field must be populated.
        primary = {
            "explain": "explanation",
            "simplify": "simplified_explanation",
            "example": "example",
            "hint": "hint",
            "practice": "practice_question",
            "viva": "practice_question",
            "exam": "explanation",
        }[mode_name]
        if not resp.get(primary):
            print(f"  !! primary field '{primary}' is empty")
            all_ok = False
        else:
            print(f"  OK (primary field '{primary}' populated)")
    return all_ok


def run_adaptive(engine, benchmark_path: Path, verbose: bool) -> bool:
    print("\n" + "=" * 70)
    print("PART C - ADAPTIVE LEARNING INTERACTION (real model)")
    print("=" * 70)
    from backend.app.tutoring.schemas import EvaluateAnswerRequest
    from backend.app.tutoring.evaluator import evaluate_answer

    student = STUDENT_REGISTRY.get_or_create("live-adaptive")

    # Reference points from the tutoring benchmark (deadlock case).
    points = [
        "a set of processes are permanently blocked",
        "each process waits for a resource held by another process",
    ]

    # Step 1: tutor explains.
    resp = engine.tutor(
        "Explain deadlock to me like I am a beginner.",
        TutorMode.EXPLAIN,
        student,
    ).model_dump()
    check_q = resp.get("check_question") or "What is deadlock?"
    print(f"\n  tutor check question: {check_q[:140]}")
    record = student.topic("deadlocks")
    print(f"  before: mastery={record.estimated_mastery:.2f} "
          f"difficulty={student.difficulty('deadlocks')}")

    # Step 2: student answers INCORRECTLY.
    bad = evaluate_answer(
        EvaluateAnswerRequest(
            student_id="live-adaptive",
            topic="deadlocks",
            question=check_q,
            student_answer="Deadlock is a CPU scheduling algorithm that picks "
            "the shortest job first.",
            reference_points=points,
        ),
        engine.llm,
    )
    print(f"  incorrect answer -> verdict={bad.verdict.value} "
          f"delta={bad.mastery_update:+.2f} feedback={bad.feedback[:110]}")
    record.apply_result(bad.verdict, hint_used=False)
    print(f"  after:  mastery={record.estimated_mastery:.2f} "
          f"difficulty={student.difficulty('deadlocks')}")

    # Step 3: student answers CORRECTLY (twice for visible mastery growth).
    good = evaluate_answer(
        EvaluateAnswerRequest(
            student_id="live-adaptive",
            topic="deadlocks",
            question=check_q,
            student_answer="A deadlock is when a set of processes are "
            "permanently blocked because each process waits for a resource "
            "held by another process, so none can proceed.",
            reference_points=points,
        ),
        engine.llm,
    )
    print(f"  correct answer   -> verdict={good.verdict.value} "
          f"delta={good.mastery_update:+.2f} feedback={good.feedback[:110]}")
    record.apply_result(good.verdict, hint_used=False)
    record.apply_result(good.verdict, hint_used=False)
    print(f"  final:  mastery={record.estimated_mastery:.2f} "
          f"difficulty={student.difficulty('deadlocks')}")
    print(f"  progress totals: attempted={student.totals()[0]} "
          f"correct={student.totals()[1]}")

    ok = (
        bad.verdict.value in ("incorrect", "partially_correct")
        and good.verdict.value == "correct"
        and record.estimated_mastery > 0
    )
    print(f"  adaptive interaction: {'OK' if ok else 'PROBLEM'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes-only", action="store_true")
    parser.add_argument("--adaptive-only", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="print full texts")
    args = parser.parse_args()

    llm = create_llm_client()
    if isinstance(llm, TransformersLLMClient):
        print(f"Model: {llm.model_name} | requested device: {llm.device} "
              f"| dtype: {llm.torch_dtype}")
    retriever = Retriever.from_prebuilt()
    engine = TutorEngine(retriever, llm)
    print(f"Index: {retriever.vector_store.size} chunks loaded\n")

    benchmark = Path("data/evaluation/os_tutoring_benchmark.json")
    results = {}
    if not args.modes_only and not args.adaptive_only:
        results["quality"] = run_quality_questions(engine, args.verbose)
    if not args.adaptive_only:
        results["modes"] = run_all_modes(engine, args.verbose)
    if not args.modes_only:
        results["adaptive"] = run_adaptive(engine, benchmark, args.verbose)

    print("\n" + "=" * 70)
    print("SUMMARY:", json.dumps(results))
    ok = all(results.values()) if results else False
    print("OVERALL:", "PASS" if ok else "ISSUES FOUND (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
