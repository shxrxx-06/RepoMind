"""Conservative, AST-only function and method call extraction."""

import ast
import tokenize
from collections import defaultdict
from pathlib import Path, PurePosixPath

from backend.app.graph.models import (
    CallResolution,
    DependencyInfo,
    DependencyResolution,
    FunctionCallInfo,
)
from backend.app.intelligence.dependency_analyzer import normalize_repo_path
from backend.app.parsing.models import CodeSymbol, ParsedFile, ParseError

MAX_CALL_SCOPE_DEPTH = 100


class _Scope:
    def __init__(self, name: str, kind: str) -> None:
        self.name = name
        self.kind = kind


class _CallVisitor(ast.NodeVisitor):
    """Track lexical callers while collecting call expressions iteratively via AST visitor."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.scopes: list[_Scope] = []
        self.calls: list[tuple[ast.Call, str, tuple[_Scope, ...]]] = []
        self.errors: list[ParseError] = []

    @property
    def caller(self) -> str:
        return self.scopes[-1].name if self.scopes else "<module>"

    def _visit_definition_header(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            if argument.annotation:
                self.visit(argument.annotation)
        if node.args.vararg and node.args.vararg.annotation:
            self.visit(node.args.vararg.annotation)
        if node.args.kwarg and node.args.kwarg.annotation:
            self.visit(node.args.kwarg.annotation)
        if node.returns:
            self.visit(node.returns)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._visit_definition_header(node)
        qualified = f"{self.scopes[-1].name}.{node.name}" if self.scopes else node.name
        if len(self.scopes) >= MAX_CALL_SCOPE_DEPTH:
            self.errors.append(
                ParseError(
                    message=f"Call analysis exceeded scope nesting limit ({MAX_CALL_SCOPE_DEPTH})",
                    error_type="NestingLimitError",
                    line=node.lineno,
                    column=node.col_offset,
                )
            )
            return
        self.scopes.append(_Scope(qualified, "function"))
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in [*node.decorator_list, *node.bases]:
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)
        qualified = f"{self.scopes[-1].name}.{node.name}" if self.scopes else node.name
        if len(self.scopes) >= MAX_CALL_SCOPE_DEPTH:
            self.errors.append(
                ParseError(
                    message=f"Call analysis exceeded scope nesting limit ({MAX_CALL_SCOPE_DEPTH})",
                    error_type="NestingLimitError",
                    line=node.lineno,
                    column=node.col_offset,
                )
            )
            return
        self.scopes.append(_Scope(qualified, "class"))
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append((node, self.caller, tuple(self.scopes)))
        self.generic_visit(node)


class CallAnalyzer:
    """Extract calls and resolve only targets supported by static syntax."""

    def analyze(
        self,
        parsed_files: list[ParsedFile],
        repository_root: str | Path,
        dependencies: list[DependencyInfo],
    ) -> tuple[list[FunctionCallInfo], list[ParseError]]:
        root = Path(repository_root).resolve()
        parsed_by_path = {normalize_repo_path(item.file_path): item for item in parsed_files}
        symbols_by_path: dict[str, list[CodeSymbol]] = {
            path: parsed.symbols for path, parsed in parsed_by_path.items()
        }
        dependency_map: dict[str, list[DependencyInfo]] = defaultdict(list)
        for dependency in dependencies:
            dependency_map[dependency.source_file].append(dependency)

        calls: list[FunctionCallInfo] = []
        errors: list[ParseError] = []
        for source_file in sorted(parsed_by_path):
            parsed = parsed_by_path[source_file]
            source_path = root.joinpath(*PurePosixPath(source_file).parts)
            try:
                with tokenize.open(source_path) as source_stream:
                    source_code = source_stream.read()
                tree = ast.parse(source_code, filename=source_file)
            except (OSError, UnicodeError, SyntaxError, RecursionError) as exc:
                errors.append(
                    ParseError(
                        message=getattr(exc, "msg", str(exc)),
                        error_type=type(exc).__name__,
                        line=getattr(exc, "lineno", 0) or 0,
                        column=getattr(exc, "offset", 0) or 0,
                    )
                )
                continue

            visitor = _CallVisitor(source_file)
            try:
                visitor.visit(tree)
            except RecursionError:
                errors.append(
                    ParseError(
                        message="Call analysis exceeded Python AST traversal depth",
                        error_type="RecursionError",
                    )
                )
            errors.extend(visitor.errors)
            for call, caller, scopes in visitor.calls:
                function_info = self._resolve_call(
                    call,
                    caller,
                    scopes,
                    source_file,
                    parsed.symbols,
                    symbols_by_path,
                    dependency_map[source_file],
                )
                calls.append(function_info)

        return (
            sorted(
                calls,
                key=lambda item: (
                    item.source_file,
                    item.line_number,
                    item.caller_qualified_name,
                    item.call_expression,
                ),
            ),
            errors,
        )

    @classmethod
    def _resolve_call(
        cls,
        call: ast.Call,
        caller: str,
        scopes: tuple[_Scope, ...],
        source_file: str,
        source_symbols: list[CodeSymbol],
        symbols_by_path: dict[str, list[CodeSymbol]],
        dependencies: list[DependencyInfo],
    ) -> FunctionCallInfo:
        called_name = ast.unparse(call.func)
        targets: list[tuple[str, CodeSymbol]] = []

        if isinstance(call.func, ast.Name):
            name = call.func.id
            targets = cls._resolve_lexical(name, scopes, source_file, source_symbols)
            if not targets:
                targets = cls._resolve_from_import(name, dependencies, symbols_by_path)
        elif isinstance(call.func, ast.Attribute):
            receiver = call.func.value
            method_name = call.func.attr
            if isinstance(receiver, ast.Name) and receiver.id in {"self", "cls"}:
                class_scopes = [scope for scope in reversed(scopes) if scope.kind == "class"]
                if class_scopes:
                    targets = cls._symbols_named(
                        source_file,
                        f"{class_scopes[0].name}.{method_name}",
                        symbols_by_path,
                    )
            if not targets:
                targets = cls._resolve_class_or_module_attribute(
                    receiver,
                    method_name,
                    source_file,
                    source_symbols,
                    dependencies,
                    symbols_by_path,
                )

        unique_targets = {
            (path, symbol.qualified_name, symbol.start_line): (path, symbol)
            for path, symbol in targets
        }
        if len(unique_targets) == 1:
            target_file, target_symbol = next(iter(unique_targets.values()))
            target_name = target_symbol.qualified_name
            resolution = CallResolution.RESOLVED
        elif len(unique_targets) > 1:
            target_file = target_name = None
            resolution = CallResolution.AMBIGUOUS
        else:
            target_file = target_name = None
            resolution = CallResolution.UNRESOLVED

        return FunctionCallInfo(
            caller_qualified_name=caller,
            called_name=called_name,
            source_file=source_file,
            line_number=call.lineno,
            call_expression=ast.unparse(call),
            resolution_status=resolution,
            target_qualified_name=target_name,
            target_file=target_file,
        )

    @staticmethod
    def _symbols_named(
        path: str,
        qualified_name: str,
        symbols_by_path: dict[str, list[CodeSymbol]],
        *,
        symbol_name: str | None = None,
    ) -> list[tuple[str, CodeSymbol]]:
        return [
            (path, symbol)
            for symbol in symbols_by_path.get(path, [])
            if symbol.qualified_name == qualified_name
            and (symbol_name is None or symbol.name == symbol_name)
        ]

    @classmethod
    def _resolve_lexical(
        cls,
        name: str,
        scopes: tuple[_Scope, ...],
        source_file: str,
        source_symbols: list[CodeSymbol],
    ) -> list[tuple[str, CodeSymbol]]:
        function_scopes = [scope.name for scope in reversed(scopes) if scope.kind == "function"]
        for scope_name in function_scopes:
            candidate = f"{scope_name}.{name}"
            found = [
                (source_file, symbol)
                for symbol in source_symbols
                if symbol.qualified_name == candidate
                and symbol.symbol_type in {"function", "method"}
            ]
            if found:
                return found
        return [
            (source_file, symbol)
            for symbol in source_symbols
            if symbol.qualified_name == name and symbol.symbol_type == "function"
        ]

    @classmethod
    def _resolve_from_import(
        cls,
        name: str,
        dependencies: list[DependencyInfo],
        symbols_by_path: dict[str, list[CodeSymbol]],
    ) -> list[tuple[str, CodeSymbol]]:
        targets: list[tuple[str, CodeSymbol]] = []
        for dependency in dependencies:
            if (
                dependency.resolution != DependencyResolution.INTERNAL
                or not dependency.imported_file
                or not dependency.symbol_name
                or (dependency.alias or dependency.symbol_name) != name
            ):
                continue
            targets.extend(
                (dependency.imported_file, symbol)
                for symbol in symbols_by_path.get(dependency.imported_file, [])
                if symbol.name == dependency.symbol_name
                and symbol.symbol_type in {"function", "method", "class"}
            )
        return targets

    @classmethod
    def _resolve_class_or_module_attribute(
        cls,
        receiver: ast.expr,
        attribute: str,
        source_file: str,
        source_symbols: list[CodeSymbol],
        dependencies: list[DependencyInfo],
        symbols_by_path: dict[str, list[CodeSymbol]],
    ) -> list[tuple[str, CodeSymbol]]:
        if isinstance(receiver, ast.Name):
            receiver_name = receiver.id
            class_targets: list[tuple[str, CodeSymbol]] = [
                (source_file, symbol)
                for symbol in source_symbols
                if symbol.name == receiver_name and symbol.symbol_type == "class"
            ]
            for dependency in dependencies:
                if (
                    dependency.resolution == DependencyResolution.INTERNAL
                    and dependency.imported_file
                    and dependency.symbol_name
                    and (dependency.alias or dependency.symbol_name) == receiver_name
                ):
                    class_targets.extend(
                        (dependency.imported_file, symbol)
                        for symbol in symbols_by_path.get(dependency.imported_file, [])
                        if symbol.name == dependency.symbol_name and symbol.symbol_type == "class"
                    )
            methods: list[tuple[str, CodeSymbol]] = []
            for path, target_class in class_targets:
                methods.extend(
                    cls._symbols_named(
                        path,
                        f"{target_class.qualified_name}.{attribute}",
                        symbols_by_path,
                        symbol_name=attribute,
                    )
                )
            if methods:
                return methods

        receiver_text = ast.unparse(receiver)
        for dependency in dependencies:
            if dependency.resolution != DependencyResolution.INTERNAL or not dependency.imported_file:
                continue
            binding = dependency.alias
            if dependency.symbol_name:
                binding = binding or dependency.symbol_name
            else:
                binding = binding or (dependency.module_name or "").split(".", 1)[0]
            expected = binding
            if not dependency.alias and not dependency.symbol_name:
                expected = dependency.module_name or binding
            if receiver_text != expected:
                continue
            # `from package import Thing` may refer to a class; `import package.module`
            # and `from package import submodule` refer to a module file.
            if dependency.symbol_name:
                imported_class = any(
                    symbol.name == dependency.symbol_name and symbol.symbol_type == "class"
                    for symbol in symbols_by_path.get(dependency.imported_file, [])
                )
                if imported_class:
                    class_symbols = [
                        symbol
                        for symbol in symbols_by_path[dependency.imported_file]
                        if symbol.name == dependency.symbol_name and symbol.symbol_type == "class"
                    ]
                    return [
                        (dependency.imported_file, candidate)
                        for class_symbol in class_symbols
                        for candidate in symbols_by_path[dependency.imported_file]
                        if candidate.qualified_name == f"{class_symbol.qualified_name}.{attribute}"
                    ]
            return [
                (dependency.imported_file, symbol)
                for symbol in symbols_by_path.get(dependency.imported_file, [])
                if symbol.qualified_name == attribute
                and symbol.symbol_type in {"function", "class"}
            ]
        return []
