"""Python source parser built on the standard library AST module."""

import ast
import tokenize
from pathlib import Path

from backend.app.parsing.base import BaseParser
from backend.app.parsing.models import (
    CodeSymbol,
    ImportInfo,
    MethodType,
    ParameterInfo,
    ParsedFile,
    ParseError,
    SymbolType,
)

MAX_SCOPE_NESTING = 100


class PythonParser(BaseParser):
    """Extract imports and nested class/function definitions without execution."""

    language = "Python"

    def parse(self, source_code: str, file_path: str | Path) -> ParsedFile:
        """Parse Python text and return structured errors for invalid/deep input."""
        result = ParsedFile(file_path=str(file_path), language=self.language)
        try:
            tree = ast.parse(source_code, filename=str(file_path))
            result.imports = self._extract_imports(tree)
            result.symbols, nesting_errors = self._extract_symbols(tree)
            result.errors.extend(nesting_errors)
        except SyntaxError as exc:
            result.errors.append(
                ParseError(
                    message=exc.msg,
                    error_type="SyntaxError",
                    line=exc.lineno or 0,
                    column=exc.offset or 0,
                )
            )
        except RecursionError:
            result.errors.append(
                ParseError(
                    message="Python source exceeds the parser's nesting limit",
                    error_type="RecursionError",
                )
            )
        except (ValueError, TypeError) as exc:
            result.errors.append(
                ParseError(message=str(exc), error_type=type(exc).__name__)
            )
        return result

    def parse_file(self, file_path: str | Path) -> ParsedFile:
        """Read UTF-8 or PEP 263 encoded source; report unavailable files."""
        path = Path(file_path)
        try:
            with tokenize.open(path) as source_file:
                source_code = source_file.read()
        except (OSError, UnicodeError, SyntaxError) as exc:
            return ParsedFile(
                file_path=str(file_path),
                language=self.language,
                errors=[
                    ParseError(
                        message=str(exc),
                        error_type=type(exc).__name__,
                        line=getattr(exc, "lineno", 0) or 0,
                        column=getattr(exc, "offset", 0) or 0,
                    )
                ],
            )
        return self.parse(source_code, file_path)

    @staticmethod
    def _text(node: ast.AST | None) -> str | None:
        return ast.unparse(node) if node is not None else None

    @classmethod
    def _parameters(cls, arguments: ast.arguments) -> list[ParameterInfo]:
        positional = [*arguments.posonlyargs, *arguments.args]
        defaults: dict[str, ast.expr] = {}
        if arguments.defaults:
            defaults = {
                arg.arg: default
                for arg, default in zip(
                    positional[-len(arguments.defaults):], arguments.defaults
                )
            }

        parameters: list[ParameterInfo] = []
        for arg in arguments.posonlyargs:
            parameters.append(cls._parameter(arg, "positional_only", defaults))
        for arg in arguments.args:
            parameters.append(cls._parameter(arg, "positional_or_keyword", defaults))
        if arguments.vararg:
            parameters.append(cls._parameter(arguments.vararg, "var_positional", {}))
        for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
            parameters.append(
                ParameterInfo(
                    name=arg.arg,
                    kind="keyword_only",
                    annotation=cls._text(arg.annotation),
                    default=cls._text(default),
                )
            )
        if arguments.kwarg:
            parameters.append(cls._parameter(arguments.kwarg, "var_keyword", {}))
        return parameters

    @classmethod
    def _parameter(
        cls,
        argument: ast.arg,
        kind: str,
        defaults: dict[str, ast.expr],
    ) -> ParameterInfo:
        return ParameterInfo(
            name=argument.arg,
            kind=kind,
            annotation=cls._text(argument.annotation),
            default=cls._text(defaults[argument.arg]) if argument.arg in defaults else None,
        )

    @staticmethod
    def _method_type(node: ast.FunctionDef | ast.AsyncFunctionDef) -> MethodType:
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            name = target.id if isinstance(target, ast.Name) else (
                target.attr if isinstance(target, ast.Attribute) else None
            )
            if name == "classmethod":
                return MethodType.CLASS
            if name == "staticmethod":
                return MethodType.STATIC
        return MethodType.INSTANCE

    @classmethod
    def _extract_symbols(
        cls, tree: ast.Module
    ) -> tuple[list[CodeSymbol], list[ParseError]]:
        symbols: list[CodeSymbol] = []
        errors: list[ParseError] = []
        # Each stack item is a statement to inspect, its scope path, and scope kind.
        stack: list[tuple[ast.AST, tuple[str, ...], str | None, int]] = [
            (statement, (), None, 0) for statement in reversed(tree.body)
        ]
        while stack:
            node, parent_path, parent_kind, depth = stack.pop()
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if depth >= MAX_SCOPE_NESTING:
                    errors.append(
                        ParseError(
                            message=f"Definitions exceed maximum nesting depth ({MAX_SCOPE_NESTING})",
                            error_type="NestingLimitError",
                            line=node.lineno,
                            column=node.col_offset,
                        )
                    )
                    continue

                qualified_name = ".".join((*parent_path, node.name))
                decorators = [ast.unparse(item) for item in node.decorator_list]
                common = {
                    "name": node.name,
                    "qualified_name": qualified_name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno or node.lineno,
                    "decorators": decorators,
                    "docstring": ast.get_docstring(node, clean=True),
                }
                if isinstance(node, ast.ClassDef):
                    symbol = CodeSymbol(
                        **common,
                        symbol_type=SymbolType.CLASS,
                        bases=[ast.unparse(base) for base in node.bases],
                    )
                    next_kind = "class"
                else:
                    is_method = parent_kind == "class"
                    symbol = CodeSymbol(
                        **common,
                        symbol_type=(SymbolType.METHOD if is_method else SymbolType.FUNCTION),
                        parameters=cls._parameters(node.args),
                        return_annotation=cls._text(node.returns),
                        is_async=isinstance(node, ast.AsyncFunctionDef),
                        method_type=cls._method_type(node) if is_method else None,
                    )
                    next_kind = "function"
                symbols.append(symbol)
                stack.extend(
                    (statement, (*parent_path, node.name), next_kind, depth + 1)
                    for statement in reversed(node.body)
                )
                continue

            # Follow statement/control-flow children, never expressions or type syntax.
            children = [
                child
                for child in ast.iter_child_nodes(node)
                if isinstance(
                    child,
                    (
                        ast.stmt,
                        ast.ExceptHandler,
                        ast.match_case,
                    ),
                )
            ]
            stack.extend((child, parent_path, parent_kind, depth) for child in reversed(children))
        return symbols, errors

    @staticmethod
    def _extract_imports(tree: ast.Module) -> list[ImportInfo]:
        imports: list[ImportInfo] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(
                    ImportInfo(
                        module_name=alias.name,
                        alias=alias.asname,
                        line_number=node.lineno,
                    )
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                imports.extend(
                    ImportInfo(
                        module_name=node.module,
                        symbol_name=alias.name,
                        alias=alias.asname,
                        line_number=node.lineno,
                        relative_level=node.level,
                    )
                    for alias in node.names
                )
        return sorted(
            imports,
            key=lambda item: (
                item.line_number,
                item.module_name or "",
                item.symbol_name or "",
                item.alias or "",
            ),
        )
