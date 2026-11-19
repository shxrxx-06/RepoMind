import json
from pathlib import Path

import pytest

from evaluation.evaluate import (
    EvaluationInputError,
    evaluate,
    validate_dataset,
    validate_predictions,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _small_dataset(root):
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "a.py").write_text("def foo(): pass\n", encoding="utf-8")
    (root / "src" / "b.py").write_text("def bar(): pass\n", encoding="utf-8")
    (root / "tests" / "test_a.py").write_text("def test_foo(): pass\n", encoding="utf-8")
    return {
        "schema_version": 1,
        "questions": [
            {
                "question_id": "Q-1",
                "question": "What does foo do?",
                "category": "factual_codebase",
                "difficulty": "easy",
                "expected_answer": "foo exists.",
                "key_claims": ["foo is defined"],
                "expected_evidence": [{"file_path": "src/a.py", "symbols": ["foo"]}],
                "expected_relationships": [
                    {
                        "relationship_type": "IMPORTS",
                        "source_file_path": "src/a.py",
                        "target_file_path": "src/b.py",
                        "target_symbol": "bar",
                    }
                ],
                "expected_uncertainty": "verified",
                "expected_test_mappings": [
                    {
                        "test_file_path": "tests/test_a.py",
                        "test_name": "test_foo",
                        "source_file_path": "src/a.py",
                        "source_symbol": "foo",
                    }
                ],
            },
            {
                "question_id": "Q-2",
                "question": "What happened in an unavailable external run?",
                "category": "insufficient_evidence",
                "difficulty": "easy",
                "expected_answer": "Unknown from this repository.",
                "key_claims": [],
                "expected_evidence": [],
                "expected_relationships": [],
                "expected_uncertainty": "unknown",
                "expected_test_mappings": [],
            },
        ],
    }


def _prediction(question_id="Q-1", **overrides):
    prediction = {
        "question_id": question_id,
        "answer": "",
        "evidence": [],
        "uncertainty": "verified",
        "relationships": [],
        "test_mappings": [],
    }
    prediction.update(overrides)
    return prediction


def test_bundled_benchmark_has_30_valid_questions_and_expected_categories():
    payload = json.loads((REPOSITORY_ROOT / "evaluation" / "dataset.json").read_text(encoding="utf-8"))
    questions = validate_dataset(payload, REPOSITORY_ROOT)

    assert len(questions) == 30
    assert len({item["question_id"] for item in questions}) == 30
    assert {
        category: sum(item["category"] == category for item in questions)
        for category in {item["category"] for item in questions}
    } == {
        "factual_codebase": 8,
        "dependency_tracing": 6,
        "change_impact": 5,
        "test_discovery": 6,
        "insufficient_evidence": 5,
    }


def test_dataset_validation_rejects_duplicate_ids_and_unsafe_paths(tmp_path):
    dataset = _small_dataset(tmp_path)
    dataset["questions"][1]["question_id"] = "Q-1"
    with pytest.raises(EvaluationInputError, match="Duplicate"):
        validate_dataset(dataset, tmp_path)

    dataset = _small_dataset(tmp_path / "second")
    dataset["questions"][0]["expected_evidence"][0]["file_path"] = "../outside.py"
    with pytest.raises(EvaluationInputError, match="unsafe"):
        validate_dataset(dataset, tmp_path / "second")


def test_evaluator_reports_deterministic_metrics_and_human_answer_labels(tmp_path):
    questions = validate_dataset(_small_dataset(tmp_path), tmp_path)
    predictions = [
        {
            "question_id": "Q-1",
            "answer": "foo exists and imports bar",
            "evidence": [
                {"citation_id": "E-" + "a" * 24, "file_path": "src/a.py", "symbol": "foo"},
                {"citation_id": "bad-id", "file_path": "src/b.py", "symbol": "bar"},
                {"citation_id": "E-" + "b" * 24, "file_path": "../outside.py", "symbol": "foo"},
                {"citation_id": "E-" + "c" * 24, "file_path": "missing.py", "symbol": "foo"},
            ],
            "uncertainty": "verified",
            "relationships": [
                {
                    "relationship_type": "IMPORTS",
                    "source_file_path": "src/a.py",
                    "target_file_path": "src/b.py",
                    "target_symbol": "bar",
                },
                {
                    "relationship_type": "CALLS",
                    "source_file_path": "src/b.py",
                    "target_file_path": "src/a.py",
                },
            ],
            "test_mappings": [
                {
                    "test_file_path": "tests/test_a.py",
                    "test_name": "test_foo",
                    "source_file_path": "src/a.py",
                    "source_symbol": "foo",
                },
                {
                    "test_file_path": "tests/test_a.py",
                    "test_name": "other_test",
                    "source_file_path": "src/a.py",
                    "source_symbol": "foo",
                },
            ],
            "human_answer_correct": True,
        },
        {
            "question_id": "Q-2",
            "answer": "Not reported.",
            "evidence": [],
            "uncertainty": "hypothesis",
            "relationships": [],
            "test_mappings": [],
            "human_answer_correct": False,
        },
    ]
    validated_predictions = validate_predictions(
        {"schema_version": 1, "predictions": predictions}, questions
    )

    report = evaluate(questions, validated_predictions, tmp_path)

    assert report["answer_correctness_human"] == {"labeled_count": 2, "accuracy": 0.5}
    assert report["automated_metrics"] == {
        "evidence_precision": 0.25,
        "evidence_recall": 1.0,
        "citation_path_validity": 0.25,
        "uncertainty_accuracy": 0.5,
        "dependency_relationship_precision": 0.5,
        "dependency_relationship_recall": 1.0,
        "dependency_relationship_accuracy": 0.0,
        "test_mapping_precision": 0.5,
        "test_mapping_recall": 1.0,
    }


def test_prediction_validation_rejects_duplicate_and_unknown_question_ids(tmp_path):
    questions = validate_dataset(_small_dataset(tmp_path), tmp_path)
    item = {
        "question_id": "Q-1",
        "answer": "",
        "evidence": [],
        "uncertainty": "verified",
        "relationships": [],
        "test_mappings": [],
    }
    with pytest.raises(EvaluationInputError, match="Duplicate"):
        validate_predictions({"schema_version": 1, "predictions": [item, item]}, questions)

    item["question_id"] = "UNKNOWN"
    with pytest.raises(EvaluationInputError, match="unknown"):
        validate_predictions({"schema_version": 1, "predictions": [item]}, questions)


def test_path_only_expected_evidence_matches_any_symbol_in_that_file(tmp_path):
    dataset = _small_dataset(tmp_path)
    dataset["questions"][0]["expected_evidence"][0]["symbols"] = []
    questions = validate_dataset(dataset, tmp_path)
    predictions = [
        {
            "question_id": "Q-1",
            "answer": "foo exists",
            "evidence": [
                {
                    "citation_id": "E-" + "a" * 24,
                    "file_path": "src/a.py",
                    "symbol": "foo",
                }
            ],
            "uncertainty": "verified",
            "relationships": [],
            "test_mappings": [],
        }
    ]

    report = evaluate(questions, predictions, tmp_path)

    assert report["automated_metrics"]["evidence_precision"] == 1.0
    assert report["automated_metrics"]["evidence_recall"] == 1.0


def test_path_only_expected_evidence_matches_only_one_predicted_symbol(tmp_path):
    dataset = _small_dataset(tmp_path)
    dataset["questions"][0]["expected_evidence"][0]["symbols"] = []
    questions = validate_dataset(dataset, tmp_path)
    prediction = _prediction(
        evidence=[
            {"citation_id": "E-" + "a" * 24, "file_path": "src/a.py", "symbol": "foo"},
            {"citation_id": "E-" + "b" * 24, "file_path": "src/a.py", "symbol": "unrelated"},
        ]
    )

    report = evaluate(questions, [prediction], tmp_path)

    assert report["automated_metrics"]["evidence_precision"] == 0.5
    assert report["automated_metrics"]["evidence_recall"] == 1.0


def test_missing_predictions_reduce_recall_and_uncertainty_accuracy(tmp_path):
    questions = validate_dataset(_small_dataset(tmp_path), tmp_path)

    metrics = evaluate(questions, [], tmp_path)["automated_metrics"]

    assert metrics["evidence_precision"] is None
    assert metrics["evidence_recall"] == 0.0
    assert metrics["dependency_relationship_precision"] is None
    assert metrics["dependency_relationship_recall"] == 0.0
    assert metrics["test_mapping_precision"] is None
    assert metrics["test_mapping_recall"] == 0.0
    assert metrics["uncertainty_accuracy"] == 0.0


def test_empty_expected_and_predicted_sets_have_undefined_precision_and_recall():
    questions = []

    metrics = evaluate(questions, [], REPOSITORY_ROOT)["automated_metrics"]

    assert metrics["evidence_precision"] is None
    assert metrics["evidence_recall"] is None
    assert metrics["dependency_relationship_precision"] is None
    assert metrics["dependency_relationship_recall"] is None
    assert metrics["dependency_relationship_accuracy"] is None
    assert metrics["test_mapping_precision"] is None
    assert metrics["test_mapping_recall"] is None


@pytest.mark.parametrize(
    ("field", "records", "message"),
    [
        (
            "evidence",
            [
                {"citation_id": "E-" + "a" * 24, "file_path": "src/a.py", "symbol": "foo"},
                {"citation_id": "E-" + "a" * 24, "file_path": "src/a.py", "symbol": "foo"},
            ],
            "duplicate evidence",
        ),
        (
            "relationships",
            [
                {"relationship_type": "IMPORTS", "source_file_path": "src/a.py", "target_file_path": "src/b.py"},
                {"relationship_type": "IMPORTS", "source_file_path": "src/a.py", "target_file_path": "src/b.py"},
            ],
            "duplicate dependency relationships",
        ),
        (
            "test_mappings",
            [
                {"test_file_path": "tests/test_a.py", "test_name": "test_foo", "source_file_path": "src/a.py", "source_symbol": "foo"},
                {"test_file_path": "tests/test_a.py", "test_name": "test_foo", "source_file_path": "src/a.py", "source_symbol": "foo"},
            ],
            "duplicate test mappings",
        ),
    ],
)
def test_prediction_validation_rejects_duplicate_metric_records(
    tmp_path, field, records, message
):
    questions = validate_dataset(_small_dataset(tmp_path), tmp_path)
    prediction = _prediction(**{field: records})

    with pytest.raises(EvaluationInputError, match=message):
        validate_predictions(
            {"schema_version": 1, "predictions": [prediction]}, questions
        )


def test_expected_symbol_validation_supports_qualified_test_and_source_names(tmp_path):
    dataset = _small_dataset(tmp_path)
    (tmp_path / "src" / "a.py").write_text(
        "class Foo:\n    def run(self):\n        return True\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "test_a.py").write_text(
        "class TestFoo:\n    def test_run(self):\n        assert Foo().run()\n",
        encoding="utf-8",
    )
    item = dataset["questions"][0]
    item["expected_evidence"] = [{"file_path": "src/a.py", "symbols": ["Foo.run"]}]
    item["expected_test_mappings"] = [
        {
            "test_file_path": "tests/test_a.py",
            "test_name": "TestFoo.test_run",
            "source_file_path": "src/a.py",
            "source_symbol": "Foo.run",
        }
    ]

    assert validate_dataset(dataset, tmp_path)[0]["question_id"] == "Q-1"

    item["expected_test_mappings"][0]["test_name"] = "test_run"
    with pytest.raises(EvaluationInputError, match="not a declared test"):
        validate_dataset(dataset, tmp_path)
