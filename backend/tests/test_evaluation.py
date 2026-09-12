"""Evaluation framework tests.

Unit tests use small, controlled mocked retrieval results (no FAISS, no
models) so metric logic is verified exactly. The real-retrieval integration
test runs against the ingested toy index via the existing fixtures.
"""

import pytest

from backend.app.evaluation.dataset import (
    DATASET_PATH,
    EvalDataset,
    EvalQuestion,
    load_dataset,
)
from backend.app.evaluation.metrics import (
    ConfusionMatrix,
    recommend_threshold,
    sweep_thresholds,
)


# ---------------------------------------------------------------- dataset ---

class TestDatasetSchema:
    def test_real_dataset_loads(self):
        dataset = load_dataset()  # default path must exist and validate
        assert len(dataset.questions) >= 25
        # v1 = labels relative to the toy corpus (fixture);
        # v2 = labels re-validated against the 15-document OS knowledge base.
        # See the dataset's "label_history" for the documented history.
        assert dataset.schema_version == 2

    def test_real_dataset_documents_its_schema_history(self):
        """The current schema contract: version must match the documented history."""
        import json

        raw = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        versions = [h["schema_version"] for h in raw.get("label_history", [])]
        assert raw["schema_version"] == max(versions), (
            "dataset schema_version must be the latest entry in label_history"
        )

    def test_real_dataset_grouping_is_complete(self):
        dataset = load_dataset()
        # Every question must land in exactly one of: positive / negative / excluded
        pos = {q.id for q in dataset.reliable_positives}
        neg = {q.id for q in dataset.reliable_negatives}
        exc = {q.id for q in dataset.excluded}
        all_ids = {q.id for q in dataset.questions}
        assert pos | neg | exc == all_ids
        assert not (pos & neg) and not (pos & exc) and not (neg & exc)
        # No fake ground truth: positives require actual corpus coverage
        assert all(q.corpus_support == "covered" for q in dataset.reliable_positives)

    def test_real_dataset_topics_cover_os_syllabus(self):
        dataset = load_dataset()
        topics = {q.topic for q in dataset.questions}
        for expected in (
            "processes",
            "threads",
            "cpu-scheduling",
            "synchronization",
            "deadlocks",
            "memory-management",
            "virtual-memory",
            "file-systems",
            "io",
            "os-fundamentals",
            "non-os",
        ):
            assert expected in topics

    def test_invalid_entry_is_detected(self, tmp_path):
        bad = {
            "schema_version": 1,
            "description": "bad",
            "questions": [
                {
                    "id": "x1",
                    "question": "What is a process?",
                    "topic": "processes",
                    "expected_relevance": "definitely-not-a-label",  # invalid
                    "corpus_support": "covered",
                }
            ],
        }
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            load_dataset(path)

    def test_duplicate_ids_detected(self, tmp_path):
        q = {
            "id": "dup",
            "question": "q?",
            "topic": "t",
            "expected_relevance": "relevant",
            "corpus_support": "covered",
        }
        bad = {
            "schema_version": 1,
            "description": "dups",
            "questions": [q, dict(q)],
        }
        path = tmp_path / "dups.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate"):
            load_dataset(path)

    def test_empty_questions_rejected(self, tmp_path):
        bad = {"schema_version": 1, "description": "empty", "questions": []}
        path = tmp_path / "empty.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            load_dataset(path)

    def test_missing_dataset_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nope.json")


import json  # noqa: E402  (kept next to usage for clarity of fixtures above)


# ---------------------------------------------------------------- metrics ---

class TestConfusionMatrix:
    def test_counts_all_four_quadrants(self):
        matrix = ConfusionMatrix.from_decisions(
            should_accept=[True, True, False, False, True],
            accepted=[True, False, True, False, True],
        )
        assert (matrix.tp, matrix.fn, matrix.fp, matrix.tn) == (2, 1, 1, 1)
        assert matrix.total == 5

    def test_precision_recall_f1_known_values(self):
        # 8 TP, 2 FN, 1 FP, 4 TN
        matrix = ConfusionMatrix(tp=8, fn=2, fp=1, tn=4)
        assert matrix.precision == pytest.approx(8 / 9)
        assert matrix.recall == pytest.approx(8 / 10)
        expected_f1 = 2 * (8 / 9) * (8 / 10) / ((8 / 9) + (8 / 10))
        assert matrix.f1 == pytest.approx(expected_f1)

    def test_zero_denominators_do_not_crash(self):
        empty = ConfusionMatrix(tp=0, fn=0, fp=0, tn=0)
        assert empty.precision == 0.0
        assert empty.recall == 0.0
        assert empty.f1 == 0.0

        no_positives = ConfusionMatrix(tp=0, fn=0, fp=3, tn=4)
        assert no_positives.recall == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            ConfusionMatrix.from_decisions([True, False], [True])


class TestThresholdSweep:
    @pytest.fixture
    def controlled_scores(self):
        """Tiny controlled example with a known score separation."""
        scores = {
            "pos_high": 0.70,
            "pos_mid": 0.45,
            "pos_low": 0.30,   # hard positive
            "neg_high": 0.28,  # strong negative (vocabulary overlap)
            "neg_low": 0.05,
        }
        truth = {
            "pos_high": True,
            "pos_mid": True,
            "pos_low": True,
            "neg_high": False,
            "neg_low": False,
        }
        return scores, truth

    def test_sweep_produces_expected_rows(self, controlled_scores):
        scores, truth = controlled_scores
        thresholds = [0.20, 0.35, 0.50, 0.65]
        rows = sweep_thresholds(scores, truth, thresholds)
        assert [r.threshold for r in rows] == thresholds
        # t=0.35: pos_high/mid accepted, pos_low(0.30) rejected, both negs rejected
        at_035 = rows[1]
        assert (at_035.matrix.tp, at_035.matrix.fn, at_035.matrix.fp, at_035.matrix.tn) == (2, 1, 0, 2)
        # t=0.20: all positives accepted, but neg_high (0.28) leaks in -> FP
        at_020 = rows[0]
        assert (at_020.matrix.tp, at_020.matrix.fn, at_020.matrix.fp, at_020.matrix.tn) == (3, 0, 1, 1)
        # t=0.50: only pos_high (0.70) accepted; pos_mid (0.45) now rejected too
        at_050 = rows[2]
        assert (at_050.matrix.tp, at_050.matrix.fn, at_050.matrix.fp, at_050.matrix.tn) == (1, 2, 0, 2)

    def test_boundary_accepts_at_exact_threshold(self, controlled_scores):
        scores, truth = controlled_scores
        rows = sweep_thresholds(scores, truth, [0.45])
        # pos_mid (score exactly 0.45) must be ACCEPTED (>= comparison).
        # tp=2 (pos_high + pos_mid), fn=1 (pos_low). If the comparison were
        # strict (>), pos_mid would be rejected and tp would be 1.
        assert (rows[0].matrix.tp, rows[0].matrix.fn, rows[0].matrix.fp, rows[0].matrix.tn) == (2, 1, 0, 2)

    def test_recommendation_prefers_perfect_precision(self, controlled_scores):
        scores, truth = controlled_scores
        rows = sweep_thresholds(scores, truth, [0.20, 0.35, 0.50, 0.65])
        threshold, reason = recommend_threshold(rows, fallback=0.42)
        # t=0.20 has the best F1 (0.857) but lets a negative in (P=0.75);
        # policy prefers perfect precision: t=0.35 (P=1.0, R=0.667, F1=0.8).
        assert threshold == 0.35
        assert "precision" in reason
        assert "0.20" in reason  # explains why the best-F1 value was not chosen

    def test_recommendation_no_churn_for_identical_matrix(self, controlled_scores):
        scores, truth = controlled_scores
        rows = sweep_thresholds(scores, truth, [0.35, 0.40, 0.42, 0.50])
        current = ConfusionMatrix.from_decisions(
            list(truth.values()), [s >= 0.42 for s in scores.values()]
        )
        threshold, reason = recommend_threshold(
            rows, fallback=0.42, current_matrix=current
        )
        # t=0.35 and t=0.40 make exactly the same decisions as the current
        # 0.42 -> the current value must be kept instead of churning config.
        assert threshold == 0.42
        assert "same decisions" in reason

    def test_recommendation_flags_low_precision_datasets(self):
        # Interleaved scores: every threshold that accepts all positives also
        # accepts a negative, so no threshold can reach precision 0.9.
        scores = {"p1": 0.60, "n1": 0.58, "p2": 0.55, "n2": 0.50}
        truth = {"p1": True, "n1": False, "p2": True, "n2": False}
        rows = sweep_thresholds(scores, truth, [0.49, 0.52, 0.57, 0.61])
        threshold, reason = recommend_threshold(rows, fallback=0.42)
        # Best F1 is 0.8 at t=0.52 (P=0.667, R=1.0) < 0.9; with no
        # current_matrix supplied, the best-F1 value is returned but flagged.
        assert threshold == 0.52
        assert "more labeled data" in reason

    def test_empty_sweep_returns_fallback(self):
        threshold, reason = recommend_threshold([], fallback=0.42)
        assert threshold == 0.42
        assert "no sweep rows" in reason


class TestCollectScores:
    def test_collect_scores_uses_existing_retriever(self, ingested_index, monkeypatch):
        """Integration: the evaluation path calls the real Retriever."""
        from backend.app.core import config
        from backend.app.rag.retriever import Retriever
        from scripts.evaluate_retrieval import collect_scores

        retriever = Retriever.from_prebuilt()
        monkeypatch.setattr(config.settings, "index_dir", str(ingested_index))

        questions = [
            EvalQuestion(
                id="p1",
                question="What is a deadlock in operating systems?",
                topic="deadlocks",
                expected_relevance="relevant",
                corpus_support="covered",
            ),
            EvalQuestion(
                id="n1",
                question="What is the capital of France?",
                topic="non-os",
                expected_relevance="irrelevant",
                corpus_support="absent",
            ),
        ]
        scores, details = collect_scores(retriever, questions)

        assert scores["p1"] > scores["n1"]
        assert details["p1"]["top_source"] == "toy_corpus.md"
        assert details["p1"]["decision_at_current_threshold"] in {"accept", "reject"}
        assert isinstance(details["n1"]["all_scores"], list)
