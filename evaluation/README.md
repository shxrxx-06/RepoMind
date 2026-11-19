# RepoMind investigation benchmark

This directory contains a small, source-grounded benchmark and a deterministic evaluator. The benchmark describes questions about the checked-out RepoMind source tree; it does not invoke the investigation service or any LLM.

## Dataset format

`dataset.json` is UTF-8 JSON with `schema_version: 1` and a `questions` array. Each entry has:

- `question_id`: stable unique ID.
- `question`, `category`, and `difficulty`.
- `expected_answer` and `key_claims`: human-readable answer guidance, not string-match scoring targets.
- `expected_evidence`: repository-relative `file_path` values with relevant `symbols`.
- `expected_relationships`: directed relationship type, source and target paths, and optional source/target symbols.
- `expected_uncertainty`: one of `verified`, `partially_supported`, `hypothesis`, or `unknown`.
- `expected_test_mappings`: test file/name and source file/symbol references.

Categories are factual codebase questions, dependency tracing, change impact, test discovery, and insufficient evidence. Expected source paths and symbol declarations are checked against the repository when the evaluator loads the dataset. Symbol names use the existing Python parser's qualified names; mapped test names must exactly identify a declared `test_` function/method or `Test*` class.

## Prediction format and running

Copy `predictions.example.json` and fill its `predictions` array. Each record is keyed by `question_id` and contains the answer, evidence citations (`citation_id`, `file_path`, optional `symbol`), uncertainty label, dependency `relationships`, and `test_mappings`. `human_answer_correct` is an optional human-supplied boolean label; use `null` until reviewed.

From the repository root, run:

```powershell
python evaluation/evaluate.py --predictions evaluation/predictions.json --output evaluation/results.json
```

Use `--dataset`, `--repo-root`, or omit `--output` to customize paths or print the report to stdout. The output is sorted JSON and rounded to six decimal places for repeatable diffs.

## Metrics

Automated metrics compare structured records, not free-form answer meaning:

- **Evidence precision/recall:** one-to-one match predicted `(question ID, path, symbol)` citations against expected evidence. A path-only gold item is a path wildcard, but can match at most one prediction; exact-symbol expectations are matched first. Thus one path-only gold item cannot credit multiple unrelated symbols from that file.
- **Citation/path validity:** fraction of predicted evidence entries whose citation ID matches the `E-` plus 24 hexadecimal character format and whose safe repository-relative file path exists inside the selected repository.
- **Uncertainty accuracy:** exact match of predicted labels to expected labels over the full benchmark; omitted questions count as incorrect.
- **Dependency relationship precision/recall:** exact match of directed relationship type, source/target paths, and optional symbols. **Relationship accuracy** is the fraction of questions containing expected or predicted relationships whose complete relationship sets match exactly.
- **Test mapping precision/recall:** exact match of test path/name and mapped source path/symbol.
- **Answer correctness (human):** accuracy over predictions with a non-null `human_answer_correct` label. Reviewers should apply the expected answer and key claims, checking cited source and uncertainty. No automated string-similarity score is used as a substitute for semantic judgment.

For a metric with a zero denominator, the result is `null`. Structured predictions should be complete for reliable comparison; missing questions contribute no evidence/relationship/test-map recall and count incorrect for uncertainty. Duplicate or unknown question IDs are rejected. Within a prediction, exact duplicate evidence records (case-normalized citation ID, path, and symbol), relationships, and test mappings are rejected during validation instead of being silently collapsed or improving precision. Distinct citation IDs for the same path/symbol remain separate predictions, but one expected evidence item still credits at most one. Re-run human answer review when the benchmark source or rubric changes.

## Limitations

The initial 30 questions are manually authored against this repository snapshot. File paths and expected graph/test relationships can become stale when source code changes. The evaluator verifies declared Python symbols and qualified test names with the existing AST parser, but that only proves a declaration exists; expected answers and graph/test relationships still require human source review. Citation validity checks only citation shape and repository path existence; it does not prove that a citation points to the claimed lines or semantically supports an answer. Automated relationship and test mapping metrics assess exact metadata matches and do not measure runtime behavior or test execution. Answer correctness remains a human judgment.
