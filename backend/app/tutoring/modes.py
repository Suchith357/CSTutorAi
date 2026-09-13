"""Mode-specific teaching behavior for the Tutor Engine.

One reusable engine + a small strategy object per mode (not seven separate
implementations). Each ModeStrategy defines:
- how to phrase the tutoring instruction for the LLM,
- which response fields it populates,
- how to fold the LLM's free-text output into the structured TutorResponse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.app.tutoring.schemas import TutorMode


@dataclass(frozen=True)
class ModeStrategy:
    """Per-mode customization of the reusable tutoring pipeline."""

    mode: TutorMode
    # Instruction injected into the tutoring prompt for this mode.
    instruction: str
    # Which structured fields this mode populates.
    fields: tuple[str, ...]
    # Modes that generate a question for the student to answer.
    asks_question: bool = False
    # Extra prompt constraints appended verbatim (e.g. exam framing).
    extra_constraints: tuple[str, ...] = field(default_factory=tuple)


EXPLAIN = ModeStrategy(
    mode=TutorMode.EXPLAIN,
    instruction=(
        "Explain the concept clearly to an Operating Systems student: start with "
        "a precise definition from the sources, then build the explanation step "
        "by step. Use the labels Explanation:, Key points: (2-4 short lines), "
        "Example: (only when a source-supported example helps), and Check "
        "question: (one short question)."
    ),
    fields=("explanation", "key_points", "example", "check_question"),
)

SIMPLIFY = ModeStrategy(
    mode=TutorMode.SIMPLIFY,
    instruction=(
        "Simplify the concept for a student who is seeing it for the first time: "
        "use plain language and a concrete analogy or example drawn from the "
        "sources. Add no new facts. Keep the technical term in parentheses the "
        "first time it appears. Use the labels Simplified explanation:, Example:, "
        "and Check question:."
    ),
    fields=("simplified_explanation", "example", "check_question"),
)

EXAMPLE = ModeStrategy(
    mode=TutorMode.EXAMPLE,
    instruction=(
        "Give a concrete worked example of the concept in an Operating Systems "
        "context (e.g. specific processes, resources, or numbers exactly as the "
        "algorithm's rules dictate), briefly restating the definition first. "
        "The example is the centerpiece: keep any explanation short and make "
        "sure the example follows the mechanism the sources describe. Do not "
        "turn the example into unrelated mathematics (no probabilities, "
        "distributions, or random-arrival calculations) unless the student "
        "explicitly asked for calculations. Use the labels Explanation:, "
        "Example:, and Check question:."
    ),
    fields=("explanation", "example", "check_question"),
)

HINT = ModeStrategy(
    mode=TutorMode.HINT,
    instruction=(
        "Do NOT give the full answer. Provide one graduated hint that points the "
        "student toward the key idea without stating it, phrased as a guiding "
        "question or nudge. Keep it to 1-3 sentences under the label Hint:."
    ),
    fields=("hint",),
)

PRACTICE = ModeStrategy(
    mode=TutorMode.PRACTICE,
    instruction=(
        "Generate ONE practice question about this topic at the indicated "
        "difficulty, answerable from the sources. Output only the practice "
        "question (no answer) under the label Practice question:, so the "
        "student can attempt it first. Do not solve it."
    ),
    fields=("practice_question",),
    asks_question=True,
)

VIVA = ModeStrategy(
    mode=TutorMode.VIVA,
    instruction=(
        "Conduct an oral-examination style interaction: ask ONE probing viva "
        "question about this topic at the indicated difficulty, suitable for a "
        "spoken answer. Output only the question (no answer) under the label "
        "Practice question:."
    ),
    fields=("practice_question",),
    asks_question=True,
)

EXAM = ModeStrategy(
    mode=TutorMode.EXAM,
    instruction=(
        "Answer in exam-preparation style: a crisp definition, the key points an "
        "examiner expects, and the most common mistake students make about this "
        "topic. Use the labels Explanation:, Key points:, Common mistake:, and "
        "Check question:."
    ),
    fields=("explanation", "key_points", "common_mistake", "check_question"),
    extra_constraints=(
        "Distinguish carefully between a concept's DEFINITION and related "
        "mechanisms: e.g. necessary conditions enable a phenomenon, while "
        "prevention is the strategy of breaking at least one of them - never "
        "conflate the two.",
    ),
)


STRATEGIES: dict[TutorMode, ModeStrategy] = {
    strategy.mode: strategy
    for strategy in (EXPLAIN, SIMPLIFY, EXAMPLE, HINT, PRACTICE, VIVA, EXAM)
}


def get_strategy(mode: TutorMode) -> ModeStrategy:
    """Return the strategy for a teaching mode."""
    return STRATEGIES[mode]
