"""Prepare concise, symbol-level embedding documents from parser output."""

from __future__ import annotations

import hashlib
import tokenize
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from backend.app.embeddings.models import CodeDocument
from backend.app.intelligence.dependency_analyzer import normalize_repo_path
from backend.app.parsing.models import CodeSymbol, ParsedFile


class CodeDocumentPreparer:
    """Turn each parsed class/function/method into its own semantic document."""

    def __init__(self, max_source_chars: int = 8_000) -> None:
        if max_source_chars < 0:
            raise ValueError("max_source_chars must be non-negative")
        self.max_source_chars = max_source_chars

    def prepare(
        self,
        parsed_files: list[ParsedFile],
        source_texts: Mapping[str, str] | None = None,
        repository_root: str | Path | None = None,
    ) -> list[CodeDocument]:
        """Build deterministic documents while keeping source metadata out of text."""
        text_by_path = {
            normalize_repo_path(path): text for path, text in (source_texts or {}).items()
        }
        root = Path(repository_root) if repository_root is not None else None
        documents: list[CodeDocument] = []
        for parsed in sorted(parsed_files, key=lambda item: normalize_repo_path(item.file_path)):
            file_path = normalize_repo_path(parsed.file_path)
            source = text_by_path.get(file_path)
            if source is None and root is not None:
                source = self._read_source(root, file_path)
            lines = source.splitlines() if source is not None else []
            for symbol in sorted(
                parsed.symbols,
                key=lambda item: (item.start_line, item.end_line, item.qualified_name),
            ):
                snippet = self._snippet(symbol, lines)
                text = self._document_text(parsed.language, file_path, symbol, snippet)
                identity = (
                    f"{file_path}\0{symbol.qualified_name}\0"
                    f"{symbol.start_line}\0{symbol.end_line}"
                )
                document_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
                documents.append(
                    CodeDocument(
                        document_id=document_id,
                        text=text,
                        source_file_path=file_path,
                        qualified_symbol_name=symbol.qualified_name,
                        symbol_type=str(symbol.symbol_type),
                        start_line=symbol.start_line,
                        end_line=symbol.end_line,
                        source_snippet=snippet,
                        docstring=symbol.docstring,
                        decorators=list(symbol.decorators),
                        bases=list(symbol.bases),
                        parameters=[parameter.model_copy(deep=True) for parameter in symbol.parameters],
                        return_annotation=symbol.return_annotation,
                        method_type=(
                            str(symbol.method_type) if symbol.method_type is not None else None
                        ),
                        is_async=symbol.is_async,
                    )
                )
        return documents

    def _snippet(self, symbol: CodeSymbol, lines: list[str]) -> str | None:
        if not lines or self.max_source_chars == 0:
            return None
        start = max(symbol.start_line - 1, 0)
        if symbol.symbol_type == "class":
            code = lines[start] if start < len(lines) else ""
        else:
            end = min(symbol.end_line, len(lines))
            code = "\n".join(lines[start:end])
        code = code.strip()
        if not code:
            return None
        if len(code) > self.max_source_chars:
            code = code[: self.max_source_chars].rstrip() + "\n# [source truncated]"
        return code

    @staticmethod
    def _document_text(
        language: str,
        file_path: str,
        symbol: CodeSymbol,
        source_snippet: str | None,
    ) -> str:
        lines = [
            (
                f"{language} {'async ' if symbol.is_async else ''}{symbol.symbol_type}: "
                f"{symbol.qualified_name}"
            ),
            f"File: {file_path}",
        ]
        if symbol.bases:
            lines.append(f"Bases: {', '.join(symbol.bases)}")
        if symbol.decorators:
            lines.append(f"Decorators: {', '.join(symbol.decorators)}")
        if symbol.parameters:
            rendered_parameters = []
            for parameter in symbol.parameters:
                prefix = {
                    "var_positional": "*",
                    "var_keyword": "**",
                }.get(parameter.kind, "")
                value = f"{prefix}{parameter.name}"
                if parameter.annotation:
                    value += f": {parameter.annotation}"
                if parameter.default is not None:
                    value += f" = {parameter.default}"
                rendered_parameters.append(value)
            lines.append(f"Parameters: {', '.join(rendered_parameters)}")
        if symbol.method_type:
            lines.append(f"Method type: {symbol.method_type}")
        if symbol.return_annotation:
            lines.append(f"Returns: {symbol.return_annotation}")
        if symbol.docstring:
            lines.append(f"Description: {symbol.docstring.strip()}")
        if source_snippet:
            lines.extend(("Source:", source_snippet))
        return "\n".join(lines)

    @staticmethod
    def _read_source(root: Path, file_path: str) -> str | None:
        path = (root / Path(*PurePosixPath(file_path).parts)).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            return None
        try:
            with tokenize.open(path) as source_file:
                return source_file.read()
        except (OSError, UnicodeError, SyntaxError):
            return None
