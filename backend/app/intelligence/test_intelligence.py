"""AST and graph based inventory of repository tests and source references."""

import ast
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import ClassVar

from backend.app.graph.models import DependencyInfo, GraphNode, RepositoryAnalysisResult
from backend.app.graph.repository_graph import file_node_id, symbol_node_id
from backend.app.intelligence.impact_analyzer import (
    ChangeImpactAnalyzer,
)
from backend.app.intelligence.impact_models import ImpactAnalysisRequest
from backend.app.intelligence.test_intelligence_models import (
    DiscoveredTestFile,
    InferredTestSuggestion,
    TestCaseInfo,
    TestCodeMapping,
    TestIntelligenceRequest,
    TestIntelligenceResponse,
    TestMappingEvidence,
    UncoveredComponent,
)
from backend.app.parsing.models import CodeSymbol, ImportInfo, ParsedFile, ParseError


def is_python_test_path(file_path: str) -> bool:
    """Return whether a Python file follows a common pytest file convention."""
    name = PurePosixPath(file_path).name.casefold()
    return name.startswith("test_") and name.endswith(".py") or (
        name.endswith("_test.py")
    )


class TestIntelligenceAnalyzer:
    """Build conservative mappings using parsed tests and repository graph evidence."""

    _CERTAINTY_RANK: ClassVar[dict[str, int]] = {"low": 0, "medium": 1, "high": 2}

    def analyze(
        self,
        repository_id: str,
        repository_root: str | Path,
        parsed_files: list[ParsedFile],
        analysis: RepositoryAnalysisResult,
        request: TestIntelligenceRequest,
    ) -> TestIntelligenceResponse:
        root = Path(repository_root).resolve()
        parsed_by_path = {
            PurePosixPath(item.file_path.replace("\\", "/")).as_posix(): item
            for item in parsed_files
        }
        for file_path in parsed_by_path:
            if not self._safe_repository_path(file_path):
                raise ValueError("Parsed source path is not repository-relative")
            try:
                (root / PurePosixPath(file_path)).resolve().relative_to(root)
            except ValueError as exc:
                raise ValueError("Parsed source path escapes the repository root") from exc
        test_paths = sorted(path for path in parsed_by_path if is_python_test_path(path))
        test_symbols: dict[str, list[TestCaseInfo]] = {}
        test_files: list[DiscoveredTestFile] = []
        for path in test_paths:
            parsed = parsed_by_path[path]
            cases = self._test_cases(parsed.symbols)
            test_symbols[path] = cases
            test_files.append(
                DiscoveredTestFile(
                    file_path=path,
                    tests=cases,
                    imports=parsed.imports,
                    parse_errors=self._safe_parse_errors(parsed.errors, root),
                )
            )

        test_path_set = set(test_paths)
        nodes_by_id = {node.node_id: node for node in analysis.nodes}
        if len(nodes_by_id) != len(analysis.nodes):
            raise ValueError("Repository analysis contains duplicate graph node identifiers")
        for node in analysis.nodes:
            if not self._valid_graph_node(node):
                raise ValueError("Repository analysis contains an invalid graph node")
        node_ids = set(nodes_by_id)
        if any(
            relationship.source not in node_ids or relationship.target not in node_ids
            for relationship in analysis.relationships
        ):
            raise ValueError("Repository analysis relationships must reference graph nodes")
        symbols_by_file: dict[str, list[str]] = defaultdict(list)
        for node in analysis.nodes:
            if node.node_type != "File" and node.file_path:
                symbols_by_file[node.file_path].append(node.node_id)
        mappings: dict[tuple[str, str, str, str], dict[str, object]] = {}
        # Import context is scoped to the module or the test case containing the
        # import. Each binding retains its public name and imported target so
        # aliases can be compared without treating them as resolved calls.
        imported_bindings: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
        imported_targets_by_file: dict[str, set[str]] = defaultdict(set)
        source_symbols_by_leaf: dict[str, list[str]] = defaultdict(list)
        for node in analysis.nodes:
            if node.node_type != "File" and node.qualified_name and node.file_path not in test_path_set:
                source_symbols_by_leaf[node.qualified_name.rsplit(".", 1)[-1]].append(node.node_id)

        def add_mapping(
            test_path: str,
            test_name: str,
            source_node_id: str,
            certainty: str,
            evidence: TestMappingEvidence,
        ) -> None:
            source_node = nodes_by_id.get(source_node_id)
            if (
                source_node is None
                or source_node.file_path in test_path_set
                or source_node.file_path is None
            ):
                return
            classification = {
                "resolved_call": "direct_resolved_call",
                "resolved_call_outside_test": "uncertain_candidate",
                "imported_symbol": "import_only_reference",
                "imported_module": "import_only_reference",
                "unresolved_call_candidate": "uncertain_candidate",
            }[evidence.evidence_type]
            key = (test_path, test_name, source_node_id, classification)
            record = mappings.setdefault(
                key,
                {
                    "certainty": certainty,
                    "evidence_classification": classification,
                    "evidence": [],
                },
            )
            if self._CERTAINTY_RANK[certainty] > self._CERTAINTY_RANK[record["certainty"]]:
                record["certainty"] = certainty
            record["evidence"].append(evidence)

        for dependency in analysis.dependencies:
            source_path = PurePosixPath(dependency.source_file).as_posix()
            target_path = dependency.imported_file
            if source_path not in test_path_set or not target_path:
                continue
            if str(dependency.resolution) != "internal":
                continue
            scope = self._import_scope(
                source_path, dependency.line_number, test_symbols[source_path], root
            )
            test_name = scope if scope != "<non-test>" else "<module>"
            original_import = next(
                (
                    item
                    for item in parsed_by_path[source_path].imports
                    if item.line_number == dependency.line_number
                    and item.symbol_name == dependency.symbol_name
                    and item.alias == dependency.alias
                ),
                None,
            )
            evidence_expression = self._import_expression(dependency, original_import)
            if dependency.symbol_name:
                imported_target_id = symbol_node_id(target_path, dependency.symbol_name)
                if imported_target_id in nodes_by_id:
                    imported_targets_by_file[source_path].add(imported_target_id)
            else:
                imported_targets_by_file[source_path].update(
                    symbols_by_file.get(target_path, [])
                )
            if scope == "<non-test>":
                continue
            if dependency.symbol_name:
                target_id = symbol_node_id(target_path, dependency.symbol_name)
                if target_id in nodes_by_id:
                    binding_name = dependency.alias or dependency.symbol_name
                    imported_bindings[(source_path, scope)].append(
                        (binding_name, target_id, evidence_expression)
                    )
                    add_mapping(
                        source_path,
                        test_name,
                        target_id,
                        "medium",
                        TestMappingEvidence(
                            evidence_type="imported_symbol",
                            description="The test file imports this source symbol; import alone does not prove use or execution.",
                            line_number=dependency.line_number,
                            expression=evidence_expression,
                        ),
                    )
                    continue
            target_id = file_node_id(target_path)
            module_binding = dependency.alias or (dependency.module_name or "").rsplit(".", 1)[-1]
            for symbol_id in symbols_by_file.get(target_path, []):
                imported_bindings[(source_path, scope)].append(
                    (module_binding, symbol_id, evidence_expression)
                )
            add_mapping(
                source_path,
                test_name,
                target_id,
                "low",
                TestMappingEvidence(
                    evidence_type="imported_module",
                    description="The test file imports this source module; the import does not identify exercised symbols.",
                    line_number=dependency.line_number,
                    expression=evidence_expression,
                ),
            )

        for call in analysis.function_calls:
            source_path = PurePosixPath(call.source_file).as_posix()
            if source_path not in test_path_set:
                continue
            lexical_scope = self._import_scope(
                source_path, call.line_number, test_symbols[source_path], root
            )
            test_name = lexical_scope if lexical_scope != "<non-test>" else "<module>"
            scopes = ["<module>"]
            if test_name != "<module>":
                scopes.append(test_name)
            visible_bindings = [
                binding
                for scope in scopes
                for binding in imported_bindings.get((source_path, scope), [])
            ]
            target_id = (
                symbol_node_id(call.target_file, call.target_qualified_name)
                if call.target_file and call.target_qualified_name
                else None
            )
            imported_elsewhere = (
                target_id in imported_targets_by_file[source_path]
                and not any(binding_target == target_id for _name, binding_target, _expr in visible_bindings)
            )
            if (
                str(call.resolution_status) == "resolved"
                and call.target_file
                and call.target_qualified_name
                and not imported_elsewhere
            ):
                is_in_test_case = test_name != "<module>"
                add_mapping(
                    source_path,
                    test_name,
                    target_id,
                    "high" if is_in_test_case else "low",
                    TestMappingEvidence(
                        evidence_type=(
                            "resolved_call" if is_in_test_case else "resolved_call_outside_test"
                        ),
                        description=(
                            "Static call analysis resolved this call inside an identified test case; this does not prove the test was run or passed."
                            if is_in_test_case
                            else "The source call resolved, but it occurs outside an identified test case and is not counted as a direct test call."
                        ),
                        line_number=call.line_number,
                        expression=call.call_expression,
                    ),
                )
            elif str(call.resolution_status) in {"unresolved", "ambiguous"} or imported_elsewhere:
                called_leaf = call.called_name.rsplit(".", 1)[-1]
                imported_matches = [
                    (target_id, import_expression)
                    for bound_name, target_id, import_expression in visible_bindings
                    if (
                        bound_name == called_leaf
                        or (
                            call.called_name.rsplit(".", 1)[0].split(".", 1)[0] == bound_name
                            and nodes_by_id[target_id].qualified_name
                            and nodes_by_id[target_id].qualified_name.rsplit(".", 1)[-1] == called_leaf
                        )
                        or (
                            bound_name == nodes_by_id[target_id].qualified_name.rsplit(".", 1)[-1]
                            if nodes_by_id[target_id].qualified_name
                            else False
                        )
                    )
                ]
                if imported_matches:
                    # Keep duplicate aliases as separate candidates; downstream
                    # mapping deduplication preserves all source evidence.
                    targets = imported_matches
                elif visible_bindings:
                    targets = []
                else:
                    targets = [(target_id, "") for target_id in source_symbols_by_leaf.get(called_leaf, [])]
                status = "unresolved" if imported_elsewhere else str(call.resolution_status)
                for target_id, import_expression in targets:
                    import_context = bool(import_expression)
                    explanation = (
                        f"The {status} test call matches an imported binding ({import_expression}), "
                        "but the call target was not established."
                        if import_context
                        else f"The {status} test call matches this source name only; no import or module binding confirms the target."
                    )
                    add_mapping(
                        source_path,
                        test_name,
                        target_id,
                        "low",
                        TestMappingEvidence(
                            evidence_type="unresolved_call_candidate",
                            description=explanation,
                            line_number=call.line_number,
                            expression=(
                                f"{call.call_expression} (import: {import_expression})"
                                if import_expression
                                else call.call_expression
                            ),
                        ),
                    )

        impact_report = None
        warnings: list[str] = []
        result_limit_reached = False
        if request.changed_file_path:
            impact_request = ImpactAnalysisRequest(
                changed_file_path=request.changed_file_path,
                qualified_symbol_name=request.qualified_symbol_name,
                max_depth=request.max_depth,
                max_results=request.max_results,
            )
            impact_report = ChangeImpactAnalyzer().analyze(
                repository_id,
                root,
                analysis,
                impact_request,
            )
            warnings.extend(impact_report.warnings)
            result_limit_reached = impact_report.result_limit_reached
            relevant_components = [
                *impact_report.directly_affected,
                *impact_report.indirectly_affected,
            ]
            candidates = [
                *impact_report.changed_targets,
                *(item.node for item in relevant_components),
            ]
            unique_candidates = {node.node_id: node for node in candidates}
            target_nodes = [
                node
                for node_id, node in unique_candidates.items()
                if node.file_path not in test_path_set
            ]
            if len(target_nodes) > request.max_results:
                target_nodes = target_nodes[: request.max_results]
                result_limit_reached = True
                warnings.append(
                    "The test analysis component limit was reached; results cover only the first max_results components."
                )
        else:
            symbol_nodes = [
                node
                for node in analysis.nodes
                if node.node_type != "File"
                and node.file_path not in test_path_set
            ]
            referenced_files = {
                key[2]
                for key, record in mappings.items()
                if record["evidence_classification"] == "import_only_reference"
                and nodes_by_id[key[2]].node_type == "File"
            }
            file_nodes = [
                node
                for node in analysis.nodes
                if node.node_id in referenced_files and node.file_path not in test_path_set
            ]
            target_nodes = sorted(symbol_nodes, key=lambda node: node.node_id)
            target_nodes.extend(sorted(file_nodes, key=lambda node: node.node_id))
            if len(target_nodes) > request.max_results:
                target_nodes = target_nodes[: request.max_results]
                result_limit_reached = True
                warnings.append(
                    "The source component limit was reached; results cover only the first max_results components."
                )

        relevant_ids = {node.node_id for node in target_nodes}
        classifications_by_node: dict[str, set[str]] = defaultdict(set)
        for key, record in mappings.items():
            if key[2] in relevant_ids:
                classifications_by_node[key[2]].add(record["evidence_classification"])
        uncovered_nodes = []
        uncovered_classifications: dict[str, str] = {}
        for node in target_nodes:
            classifications = classifications_by_node.get(node.node_id, set())
            if "direct_resolved_call" in classifications:
                continue
            if "uncertain_candidate" in classifications:
                classification = "uncertain_candidate"
            elif "import_only_reference" in classifications:
                classification = "import_only_reference"
            else:
                classification = "no_discoverable_test_reference"
            uncovered_nodes.append(node)
            uncovered_classifications[node.node_id] = classification

        response_mappings = []
        for (test_path, test_name, target_id, classification), record in sorted(mappings.items()):
            if target_id not in relevant_ids:
                continue
            evidence = sorted(
                record["evidence"],
                key=lambda item: (item.line_number, item.evidence_type, item.expression or ""),
            )
            certainty = record["certainty"]
            response_mappings.append(
                TestCodeMapping(
                    test_file_path=test_path,
                    test_qualified_name=test_name,
                    source_component=nodes_by_id[target_id],
                    evidence_classification=classification,
                    certainty=certainty,
                    evidence=evidence,
                    explanation=(
                        "Static source references are shown as evidence, not as proof of runtime test coverage."
                    ),
                )
            )

        uncovered = [
            UncoveredComponent(
                source_component=node,
                evidence_classification=uncovered_classifications[node.node_id],
                explanation=self._uncovered_explanation(
                    uncovered_classifications[node.node_id]
                ),
            )
            for node in uncovered_nodes
        ]
        recommendations = [
            InferredTestSuggestion(
                source_component=item.source_component,
                evidence_classification=item.evidence_classification,
                explanation=self._recommendation_explanation(item),
            )
            for item in uncovered
        ]
        for test_file in test_files:
            if test_file.parse_errors:
                warnings.append(
                    f"Test file {test_file.file_path} could not be fully parsed; its test mappings may be incomplete."
                )
        if any(item.evidence_classification == "uncertain_candidate" for item in response_mappings):
            warnings.append("Some test mappings are uncertain candidates and require source review.")
        warnings.append(
            "Mappings describe static source references only; no test execution or runtime coverage was measured."
        )
        if not test_files:
            warnings.append("No Python test files matching test_*.py or *_test.py were discovered.")

        return TestIntelligenceResponse(
            repository_id=repository_id,
            changed_file_path=request.changed_file_path,
            qualified_symbol_name=request.qualified_symbol_name,
            discovered_test_files=test_files,
            mappings=response_mappings,
            uncovered_components=uncovered,
            inferred_recommendations=recommendations,
            impact_report=impact_report,
            max_results=request.max_results,
            analyzed_component_count=len(target_nodes),
            result_limit_reached=result_limit_reached,
            warnings=list(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _test_cases(symbols: list[CodeSymbol]) -> list[TestCaseInfo]:
        cases = []
        for symbol in symbols:
            kind = str(symbol.symbol_type)
            if kind == "class" and symbol.name.startswith("Test"):
                cases.append(
                    TestCaseInfo(
                        name=symbol.name,
                        qualified_name=symbol.qualified_name,
                        kind="class",
                        start_line=symbol.start_line,
                        end_line=symbol.end_line,
                    )
                )
            elif kind in {"function", "method"} and symbol.name.startswith("test_"):
                cases.append(
                    TestCaseInfo(
                        name=symbol.name,
                        qualified_name=symbol.qualified_name,
                        kind=kind,
                        start_line=symbol.start_line,
                        end_line=symbol.end_line,
                    )
                )
        return sorted(cases, key=lambda item: (item.start_line, item.end_line, item.qualified_name))

    @staticmethod
    def _case_for_line(cases: list[TestCaseInfo], line_number: int) -> str:
        containing = [
            item
            for item in cases
            if item.kind != "class" and item.start_line <= line_number <= item.end_line
        ]
        if not containing:
            return "<module>"
        return min(
            containing,
            key=lambda item: (item.end_line - item.start_line, item.qualified_name),
        ).qualified_name

    @staticmethod
    def _import_scope(
        source_path: str,
        line_number: int,
        cases: list[TestCaseInfo],
        repository_root: Path,
    ) -> str:
        """Return the nearest genuine test-function, module, or excluded scope."""
        try:
            source = (repository_root / PurePosixPath(source_path)).read_text(
                encoding="utf-8-sig"
            )
            tree = ast.parse(source)
        except (OSError, SyntaxError, UnicodeError):
            return "<module>"
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.lineno <= line_number <= getattr(node, "end_lineno", node.lineno)
        ]
        if functions:
            innermost = min(
                functions,
                key=lambda node: (
                    getattr(node, "end_lineno", node.lineno) - node.lineno,
                    -node.lineno,
                ),
            )
            case = next((item for item in cases if item.start_line == innermost.lineno), None)
            return case.qualified_name if case else "<non-test>"
        if any(
            isinstance(node, ast.ClassDef)
            and node.lineno <= line_number <= getattr(node, "end_lineno", node.lineno)
            for node in ast.walk(tree)
        ):
            return "<non-test>"
        return "<module>"

    @staticmethod
    def _import_expression(
        dependency: DependencyInfo, imported: ImportInfo | None
    ) -> str:
        module_name = imported.module_name if imported else dependency.module_name
        module_name = "." * (imported.relative_level if imported else 0) + (module_name or "")
        symbol_name = dependency.symbol_name
        alias = dependency.alias
        if symbol_name:
            expression = f"from {module_name} import {symbol_name}"
            return f"{expression} as {alias}" if alias else expression
        expression = f"import {module_name}"
        return f"{expression} as {alias}" if alias else expression

    @staticmethod
    def _uncovered_explanation(classification: str) -> str:
        return {
            "import_only_reference": (
                "The component is imported by a discovered test, but no resolved call from a test was established."
            ),
            "uncertain_candidate": (
                "A test contains an unresolved or ambiguous call candidate for this component; the target is not confirmed."
            ),
            "no_discoverable_test_reference": (
                "No supported static test reference to this component was discovered."
            ),
        }[classification]

    @staticmethod
    def _recommendation_explanation(item: UncoveredComponent) -> str:
        component = item.source_component.qualified_name or item.source_component.file_path
        reason = {
            "import_only_reference": "tests import it but no resolved test call was found",
            "uncertain_candidate": "the possible test call could not be resolved confidently",
            "no_discoverable_test_reference": "no supported static test reference was found",
        }[item.evidence_classification]
        return (
            f"Consider adding or reviewing a Python test for {component}: {reason}. "
            "This is an inferred suggestion, not an existing test or a runtime coverage result."
        )

    @staticmethod
    def _safe_repository_path(file_path: str) -> bool:
        parts = file_path.replace("\\", "/").split("/")
        return bool(
            file_path
            and not file_path.startswith(("/", "\\"))
            and not (len(file_path) >= 2 and file_path[1] == ":")
            and all(part not in {"", ".", ".."} for part in parts)
        )

    @classmethod
    def _valid_graph_node(cls, node: GraphNode) -> bool:
        if not node.file_path or not cls._safe_repository_path(node.file_path):
            return False
        expected_id = (
            f"file:{node.file_path}"
            if node.node_type == "File"
            else f"symbol:{node.file_path}:{node.qualified_name}"
        )
        return node.node_id == expected_id

    @staticmethod
    def _safe_parse_errors(
        errors: list[ParseError], repository_root: Path
    ) -> list[ParseError]:
        root_forms = {
            str(repository_root).casefold(),
            repository_root.as_posix().casefold(),
        }
        safe_errors = []
        for error in errors:
            message = error.message
            lowered = message.casefold()
            if any(root and root in lowered for root in root_forms):
                message = "Python test source could not be parsed or read."
            safe_errors.append(error.model_copy(update={"message": message}))
        return safe_errors
