"""Deterministic metrics for RepoMind investigation benchmark predictions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from backend.app.parsing.python_parser import PythonParser  # noqa: E402

_CATEGORIES = {
    "factual_codebase",
    "dependency_tracing",
    "change_impact",
    "test_discovery",
    "insufficient_evidence",
}
_DIFFICULTIES = {"easy", "medium", "hard"}
_UNCERTAINTY = {"verified", "partially_supported", "hypothesis", "unknown"}
_CITATION_ID = re.compile(r"^E-[0-9a-f]{24}$", re.IGNORECASE)


class EvaluationInputError(ValueError):
    """Raised when the dataset or prediction file does not match its contract."""


def _safe_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    return bool(
        not normalized.startswith("/")
        and not (len(normalized) >= 2 and normalized[1] == ":")
        and all(part not in {"", ".", ".."} for part in parts)
        and PurePosixPath(normalized).as_posix() == normalized
    )


def _repository_file(repository_root: Path, file_path: str) -> bool:
    if not _safe_path(file_path):
        return False
    normalized = file_path.replace("\\", "/")
    source_path = repository_root.joinpath(*normalized.split("/"))
    try:
        source_path.resolve(strict=True).relative_to(repository_root.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return source_path.is_file()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationInputError(f"Unable to read valid JSON from {path}.") from exc


def validate_dataset(dataset: Any, repository_root: Path) -> list[dict[str, Any]]:
    """Validate benchmark structure, identifiers, labels, and repository references."""
    if not isinstance(dataset, dict) or dataset.get("schema_version") != 1:
        raise EvaluationInputError("Dataset must be an object with schema_version 1.")
    questions = dataset.get("questions")
    if not isinstance(questions, list):
        raise EvaluationInputError("Dataset questions must be a list.")
    identifiers: set[str] = set()
    declared_symbols: dict[str, dict[str, Any]] = {}

    def symbols_for(file_path: str) -> dict[str, Any]:
        normalized_path = file_path.replace("\\", "/")
        if normalized_path not in declared_symbols:
            parsed = PythonParser().parse_file(
                repository_root.joinpath(*normalized_path.split("/"))
            )
            if parsed.errors:
                raise EvaluationInputError(
                    f"Expected symbols cannot be validated in {normalized_path}."
                )
            declared_symbols[normalized_path] = {
                symbol.qualified_name: symbol for symbol in parsed.symbols
            }
        return declared_symbols[normalized_path]

    for index, item in enumerate(questions):
        prefix = f"questions[{index}]"
        if not isinstance(item, dict):
            raise EvaluationInputError(f"{prefix} must be an object.")
        for field in ("question_id", "question", "expected_answer"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise EvaluationInputError(f"{prefix}.{field} must be a non-empty string.")
        question_id = item["question_id"]
        if question_id in identifiers:
            raise EvaluationInputError(f"Duplicate question_id: {question_id}.")
        identifiers.add(question_id)
        if item.get("category") not in _CATEGORIES:
            raise EvaluationInputError(f"{prefix}.category is not supported.")
        if item.get("difficulty") not in _DIFFICULTIES:
            raise EvaluationInputError(f"{prefix}.difficulty is not supported.")
        if item.get("expected_uncertainty") not in _UNCERTAINTY:
            raise EvaluationInputError(f"{prefix}.expected_uncertainty is not supported.")
        for field in ("key_claims", "expected_evidence", "expected_relationships", "expected_test_mappings"):
            if not isinstance(item.get(field), list):
                raise EvaluationInputError(f"{prefix}.{field} must be a list.")
        expected_evidence_keys: set[tuple[str, str]] = set()
        for evidence in item["expected_evidence"]:
            if not isinstance(evidence, dict) or not _safe_path(evidence.get("file_path")):
                raise EvaluationInputError(f"{prefix} contains an unsafe expected evidence path.")
            if not _repository_file(repository_root, evidence["file_path"]):
                raise EvaluationInputError(
                    f"{prefix} expected evidence path is missing or outside the repository."
                )
            if not isinstance(evidence.get("symbols", []), list) or any(
                not isinstance(symbol, str) for symbol in evidence.get("symbols", [])
            ):
                raise EvaluationInputError(f"{prefix}.expected_evidence symbols must be strings.")
            for symbol_name in evidence.get("symbols") or [""]:
                evidence_key = (_path(evidence["file_path"]), symbol_name.casefold())
                if evidence_key in expected_evidence_keys:
                    raise EvaluationInputError(f"{prefix} contains duplicate expected evidence records.")
                expected_evidence_keys.add(evidence_key)
                if symbol_name and symbol_name not in symbols_for(evidence["file_path"]):
                    raise EvaluationInputError(
                        f"{prefix} expected symbol {symbol_name!r} is not declared in {evidence['file_path']}."
                    )
        for relation in item["expected_relationships"]:
            if not isinstance(relation, dict) or relation.get("relationship_type") not in {
                "IMPORTS", "CALLS", "CONTAINS", "DEFINES"
            }:
                raise EvaluationInputError(f"{prefix} contains a malformed expected relationship.")
            for field in ("source_file_path", "target_file_path"):
                if not _safe_path(relation.get(field)):
                    raise EvaluationInputError(f"{prefix} contains an unsafe relationship path.")
                if not _repository_file(repository_root, relation[field]):
                    raise EvaluationInputError(f"{prefix} relationship references a missing source file.")
            if any(
                key in relation and not isinstance(relation[key], str)
                for key in ("source_symbol", "target_symbol")
            ):
                raise EvaluationInputError(f"{prefix} relationship symbols must be strings.")
            for field, path_field in (
                ("source_symbol", "source_file_path"),
                ("target_symbol", "target_file_path"),
            ):
                symbol_name = relation.get(field)
                if symbol_name and symbol_name not in symbols_for(relation[path_field]):
                    raise EvaluationInputError(
                        f"{prefix} relationship symbol {symbol_name!r} is not declared in {relation[path_field]}."
                    )
        for mapping in item["expected_test_mappings"]:
            if not isinstance(mapping, dict):
                raise EvaluationInputError(f"{prefix} contains a malformed expected test mapping.")
            for field in ("test_file_path", "source_file_path"):
                if not _safe_path(mapping.get(field)):
                    raise EvaluationInputError(f"{prefix} contains an unsafe test mapping path.")
            if not _repository_file(repository_root, mapping["test_file_path"]) or not _repository_file(
                repository_root, mapping["source_file_path"]
            ):
                raise EvaluationInputError(f"{prefix} references a missing test file.")
            if not isinstance(mapping.get("test_name"), str) or not isinstance(
                mapping.get("source_symbol"), str
            ):
                raise EvaluationInputError(f"{prefix} test mapping names must be strings.")
            test_symbol = symbols_for(mapping["test_file_path"]).get(mapping["test_name"])
            is_test_declaration = test_symbol is not None and (
                test_symbol.name.startswith("test_")
                or (
                    test_symbol.symbol_type == "class"
                    and test_symbol.name.startswith("Test")
                )
            )
            if not is_test_declaration:
                raise EvaluationInputError(
                    f"{prefix} test name {mapping['test_name']!r} is not a declared test function or class."
                )
            if mapping["source_symbol"] not in symbols_for(mapping["source_file_path"]):
                raise EvaluationInputError(
                    f"{prefix} mapped source symbol {mapping['source_symbol']!r} is not declared in {mapping['source_file_path']}."
                )
    return questions


def validate_predictions(
    payload: Any,
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate prediction records and restrict them to known benchmark IDs."""
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise EvaluationInputError("Predictions must be an object with schema_version 1.")
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise EvaluationInputError("Predictions must contain a predictions list.")
    known_ids = {item["question_id"] for item in questions}
    seen: set[str] = set()
    for index, item in enumerate(predictions):
        prefix = f"predictions[{index}]"
        if not isinstance(item, dict) or item.get("question_id") not in known_ids:
            raise EvaluationInputError(f"{prefix} references an unknown question_id.")
        question_id = item["question_id"]
        if question_id in seen:
            raise EvaluationInputError(f"Duplicate prediction for question_id: {question_id}.")
        seen.add(question_id)
        for field in ("answer", "evidence", "relationships", "test_mappings"):
            if field not in item:
                raise EvaluationInputError(f"{prefix}.{field} is required.")
        if not isinstance(item["answer"], str):
            raise EvaluationInputError(f"{prefix}.answer must be a string.")
        for field in ("evidence", "relationships", "test_mappings"):
            if not isinstance(item[field], list):
                raise EvaluationInputError(f"{prefix}.{field} must be a list.")
        if item.get("uncertainty") not in _UNCERTAINTY:
            raise EvaluationInputError(f"{prefix}.uncertainty is not supported.")
        answer_label = item.get("human_answer_correct")
        if answer_label is not None and not isinstance(answer_label, bool):
            raise EvaluationInputError(f"{prefix}.human_answer_correct must be boolean or null.")
        evidence_keys: set[tuple[str, str, str]] = set()
        for evidence in item["evidence"]:
            if not isinstance(evidence, dict) or not isinstance(evidence.get("citation_id"), str):
                raise EvaluationInputError(f"{prefix} contains a malformed evidence citation.")
            if not isinstance(evidence.get("file_path"), str):
                raise EvaluationInputError(f"{prefix} evidence file_path must be a string.")
            if evidence.get("symbol") is not None and not isinstance(evidence["symbol"], str):
                raise EvaluationInputError(f"{prefix} evidence symbols must be strings or null.")
            evidence_key = (
                evidence["citation_id"].casefold(),
                _path(evidence["file_path"]),
                (evidence.get("symbol") or "").casefold(),
            )
            if evidence_key in evidence_keys:
                raise EvaluationInputError(f"{prefix} contains duplicate evidence records.")
            evidence_keys.add(evidence_key)
        relationship_keys: set[tuple[str, ...]] = set()
        for relation in item["relationships"]:
            if not isinstance(relation, dict) or relation.get("relationship_type") not in {
                "IMPORTS", "CALLS", "CONTAINS", "DEFINES"
            }:
                raise EvaluationInputError(f"{prefix} contains a malformed relationship.")
            if not isinstance(relation.get("source_file_path"), str) or not isinstance(
                relation.get("target_file_path"), str
            ):
                raise EvaluationInputError(f"{prefix} relationship paths must be strings.")
            if any(
                key in relation and not isinstance(relation[key], str)
                for key in ("source_symbol", "target_symbol")
            ):
                raise EvaluationInputError(f"{prefix} relationship symbols must be strings.")
            relationship_key = (
                relation["relationship_type"],
                _path(relation["source_file_path"]),
                (relation.get("source_symbol") or "").casefold(),
                _path(relation["target_file_path"]),
                (relation.get("target_symbol") or "").casefold(),
            )
            if relationship_key in relationship_keys:
                raise EvaluationInputError(f"{prefix} contains duplicate dependency relationships.")
            relationship_keys.add(relationship_key)
        mapping_keys: set[tuple[str, ...]] = set()
        for mapping in item["test_mappings"]:
            if not isinstance(mapping, dict) or not isinstance(mapping.get("test_file_path"), str):
                raise EvaluationInputError(f"{prefix} contains a malformed test mapping.")
            if not isinstance(mapping.get("source_file_path"), str):
                raise EvaluationInputError(f"{prefix} mapped source path must be a string.")
            if any(
                key in mapping and not isinstance(mapping[key], str)
                for key in ("test_name", "source_symbol")
            ):
                raise EvaluationInputError(f"{prefix} test mapping names must be strings.")
            mapping_key = (
                _path(mapping["test_file_path"]),
                (mapping.get("test_name") or "").casefold(),
                _path(mapping["source_file_path"]),
                (mapping.get("source_symbol") or "").casefold(),
            )
            if mapping_key in mapping_keys:
                raise EvaluationInputError(f"{prefix} contains duplicate test mappings.")
            mapping_keys.add(mapping_key)
    return predictions


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _path(value: str) -> str:
    return value.replace("\\", "/").casefold()


def _expected_evidence(questions: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    expected: list[tuple[str, str, str]] = []
    for question in questions:
        for item in question["expected_evidence"]:
            symbols = item.get("symbols") or [""]
            expected.extend(
                (question["question_id"], _path(item["file_path"]), symbol.casefold())
                for symbol in symbols
            )
    return expected


def _predicted_evidence(
    predictions: list[dict[str, Any]],
) -> list[tuple[str, str, str, str]]:
    return [
        (
            item["question_id"],
            _path(evidence["file_path"]),
            (evidence.get("symbol") or "").casefold(),
            evidence["citation_id"].casefold(),
        )
        for item in predictions
        for evidence in item["evidence"]
    ]


def _evidence_match(
    predicted: tuple[str, str, str, str],
    expected: tuple[str, str, str],
) -> bool:
    return predicted[:2] == expected[:2] and (not expected[2] or predicted[2] == expected[2])


def _match_evidence_records(
    expected: list[tuple[str, str, str]],
    predicted: list[tuple[str, str, str, str]],
) -> int:
    """Return a deterministic one-to-one match count, preferring exact symbols."""
    ordered_expected = sorted(
        expected,
        key=lambda item: (item[0], item[1], not bool(item[2]), item[2]),
    )
    ordered_predicted = sorted(predicted, key=lambda item: (*item[:3], item[3]))
    available = set(range(len(ordered_predicted)))
    match_count = 0
    for expected_item in ordered_expected:
        match_index = next(
            (
                index
                for index in sorted(available)
                if _evidence_match(ordered_predicted[index], expected_item)
            ),
            None,
        )
        if match_index is not None:
            available.remove(match_index)
            match_count += 1
    return match_count


def _relationship_records(
    records: list[dict[str, Any]],
    id_field: str,
    list_field: str,
) -> set[tuple[str, str, str, str, str, str]]:
    return {
        (
            item[id_field],
            record["relationship_type"],
            _path(record["source_file_path"]),
            record.get("source_symbol", "").casefold(),
            _path(record["target_file_path"]),
            record.get("target_symbol", "").casefold(),
        )
        for item in records
        for record in item[list_field]
    }


def _test_mapping_records(records: list[dict[str, Any]], list_field: str) -> set[tuple[str, str, str, str, str]]:
    return {
        (
            item["question_id"],
            _path(mapping["test_file_path"]),
            mapping.get("test_name", "").casefold(),
            _path(mapping["source_file_path"]),
            mapping.get("source_symbol", "").casefold(),
        )
        for item in records
        for mapping in item[list_field]
    }


def _relationship_accuracy(
    expected: set[tuple[str, str, str, str, str, str]],
    predicted: set[tuple[str, str, str, str, str, str]],
) -> float | None:
    expected_by_question: dict[str, set[tuple[str, str, str, str, str, str]]] = {}
    predicted_by_question: dict[str, set[tuple[str, str, str, str, str, str]]] = {}
    for relationship in expected:
        expected_by_question.setdefault(relationship[0], set()).add(relationship)
    for relationship in predicted:
        predicted_by_question.setdefault(relationship[0], set()).add(relationship)
    evaluated_ids = set(expected_by_question) | set(predicted_by_question)
    if not evaluated_ids:
        return None
    exact_matches = sum(
        expected_by_question.get(question_id, set())
        == predicted_by_question.get(question_id, set())
        for question_id in evaluated_ids
    )
    return round(exact_matches / len(evaluated_ids), 6)


def evaluate(
    questions: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    repository_root: Path,
) -> dict[str, Any]:
    """Compute set-based source/graph metrics and human-labeled answer accuracy."""
    predictions = validate_predictions(
        {"schema_version": 1, "predictions": predictions}, questions
    )
    question_by_id = {item["question_id"]: item for item in questions}
    prediction_by_id = {item["question_id"]: item for item in predictions}
    expected_evidence = _expected_evidence(questions)
    predicted_evidence = _predicted_evidence(predictions)
    evidence_true_positives = _match_evidence_records(expected_evidence, predicted_evidence)

    evidence_entries = [
        (item["question_id"], evidence)
        for item in predictions
        for evidence in item["evidence"]
    ]
    valid_evidence_entries = 0
    for _question_id, evidence in evidence_entries:
        path_value = evidence["file_path"]
        path_is_valid = _repository_file(repository_root, path_value)
        if _CITATION_ID.fullmatch(evidence["citation_id"]) and path_is_valid:
            valid_evidence_entries += 1

    expected_relationships = _relationship_records(questions, "question_id", "expected_relationships")
    predicted_relationships = _relationship_records(predictions, "question_id", "relationships")
    expected_test_mappings = _test_mapping_records(questions, "expected_test_mappings")
    predicted_test_mappings = _test_mapping_records(predictions, "test_mappings")
    labeled_answers = [
        prediction["human_answer_correct"]
        for prediction in predictions
        if prediction.get("human_answer_correct") is not None
    ]
    uncertainty_correct = sum(
        prediction_by_id[question_id].get("uncertainty") == question["expected_uncertainty"]
        for question_id, question in question_by_id.items()
        if question_id in prediction_by_id
    )

    return {
        "benchmark_question_count": len(questions),
        "prediction_count": len(predictions),
        "answer_correctness_human": {
            "labeled_count": len(labeled_answers),
            "accuracy": _ratio(sum(labeled_answers), len(labeled_answers)),
        },
        "automated_metrics": {
            "evidence_precision": _ratio(evidence_true_positives, len(predicted_evidence)),
            "evidence_recall": _ratio(evidence_true_positives, len(expected_evidence)),
            "citation_path_validity": _ratio(valid_evidence_entries, len(evidence_entries)),
            "uncertainty_accuracy": _ratio(uncertainty_correct, len(questions)),
            "dependency_relationship_precision": _ratio(
                len(expected_relationships & predicted_relationships), len(predicted_relationships)
            ),
            "dependency_relationship_recall": _ratio(
                len(expected_relationships & predicted_relationships), len(expected_relationships)
            ),
            "dependency_relationship_accuracy": _relationship_accuracy(
                expected_relationships, predicted_relationships
            ),
            "test_mapping_precision": _ratio(
                len(expected_test_mappings & predicted_test_mappings), len(predicted_test_mappings)
            ),
            "test_mapping_recall": _ratio(
                len(expected_test_mappings & predicted_test_mappings), len(expected_test_mappings)
            ),
        },
        "limitations": [
            "Automated evidence, path, relationship, mapping, and uncertainty metrics compare structured labels only.",
            "Answer correctness requires human labels; this evaluator does not infer semantic correctness from text similarity.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    evaluation_dir = Path(__file__).resolve().parent
    repository_root = _REPOSITORY_ROOT
    parser.add_argument("--dataset", type=Path, default=evaluation_dir / "dataset.json")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=repository_root)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    questions = validate_dataset(_load_json(args.dataset), args.repo_root)
    predictions = validate_predictions(_load_json(args.predictions), questions)
    report = evaluate(questions, predictions, args.repo_root)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
