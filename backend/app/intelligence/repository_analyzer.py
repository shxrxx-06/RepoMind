"""Orchestrate parser, dependency, call, and graph analysis for a repository."""

from pathlib import Path

from backend.app.graph.models import RepositoryAnalysisResult
from backend.app.graph.repository_graph import RepositoryGraphBuilder
from backend.app.ingestion.scanner import scan_repository
from backend.app.intelligence.call_analyzer import CallAnalyzer
from backend.app.intelligence.dependency_analyzer import (
    DependencyAnalyzer,
    normalize_repo_path,
)
from backend.app.parsing.models import ParsedFile, ParseError
from backend.app.parsing.python_parser import PythonParser


class RepositoryAnalyzer:
    """Analyze Python files without changing repository ingestion behavior."""

    def __init__(self) -> None:
        self.parser = PythonParser()
        self.dependency_analyzer = DependencyAnalyzer()
        self.call_analyzer = CallAnalyzer()
        self.graph_builder = RepositoryGraphBuilder()

    def analyze(
        self,
        repository_root: str | Path,
        parsed_files: list[ParsedFile] | None = None,
    ) -> RepositoryAnalysisResult:
        """Return repository analysis; parse files with the existing scanner by default."""
        root = Path(repository_root).resolve()
        path_errors: list[ParseError] = []
        if parsed_files is None:
            parsed_files = self._parse_repository(root)
        else:
            parsed_files, path_errors = self._normalize_file_paths(root, parsed_files)
        parsed_files = sorted(parsed_files, key=lambda item: normalize_repo_path(item.file_path))
        dependencies = self.dependency_analyzer.analyze(parsed_files)
        function_calls, call_errors = self.call_analyzer.analyze(
            parsed_files,
            root,
            dependencies,
        )
        errors = self._unique_errors(
            path_errors
            + [error for parsed in parsed_files for error in parsed.errors]
            + call_errors
        )
        graph = self.graph_builder.build(parsed_files, dependencies, function_calls)
        return self.graph_builder.to_result(graph, dependencies, function_calls, errors)

    def _parse_repository(self, root: Path) -> list[ParsedFile]:
        if not root.is_dir():
            return []
        scan = scan_repository(root)
        python_paths = [
            root.joinpath(*Path(item["path"]).parts)
            for item in scan["files"]
            if item["language"] == "Python"
        ]
        parsed_files = []
        for path in sorted(python_paths):
            parsed = self.parser.parse_file(path)
            parsed.file_path = path.relative_to(root).as_posix()
            parsed_files.append(parsed)
        return parsed_files

    @staticmethod
    def _normalize_file_paths(
        root: Path, parsed_files: list[ParsedFile]
    ) -> tuple[list[ParsedFile], list[ParseError]]:
        normalized: list[ParsedFile] = []
        errors: list[ParseError] = []
        for parsed in parsed_files:
            file_path = Path(parsed.file_path)
            resolved_path = (file_path if file_path.is_absolute() else root / file_path).resolve()
            try:
                relative_path = resolved_path.relative_to(root).as_posix()
            except ValueError:
                errors.append(
                    ParseError(
                        message=f"Parsed file is outside repository root: {file_path}",
                        error_type="PathOutsideRepositoryError",
                    )
                )
                continue
            normalized.append(parsed.model_copy(update={"file_path": relative_path}))
        return normalized, errors

    @staticmethod
    def _unique_errors(errors: list[ParseError]) -> list[ParseError]:
        unique: dict[tuple[str, str, int, int], ParseError] = {}
        for error in errors:
            unique.setdefault(
                (error.error_type, error.message, error.line, error.column), error
            )
        return list(unique.values())
