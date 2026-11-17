"""Serializable Pydantic models for language parser results."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class SymbolType(StrEnum):
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class MethodType(StrEnum):
    INSTANCE = "instance"
    CLASS = "class"
    STATIC = "static"


class ParameterInfo(BaseModel):
    """One parameter with source-independent annotation and default text."""

    name: str
    kind: str
    annotation: str | None = None
    default: str | None = None


class CodeSymbol(BaseModel):
    """A class or callable definition found in a source file."""

    name: str
    qualified_name: str
    symbol_type: SymbolType
    start_line: int
    end_line: int
    bases: list[str] = []
    decorators: list[str] = []
    docstring: str | None = None
    parameters: list[ParameterInfo] = []
    return_annotation: str | None = None
    is_async: bool = False
    method_type: MethodType | None = None

    model_config = ConfigDict(use_enum_values=True)


class ImportInfo(BaseModel):
    """One imported module or symbol, preserving alias and relative level."""

    module_name: str | None
    symbol_name: str | None = None
    alias: str | None = None
    line_number: int
    relative_level: int = 0


class ParseError(BaseModel):
    """A non-fatal parser or source loading error."""

    message: str
    error_type: str
    line: int = 0
    column: int = 0


class ParsedFile(BaseModel):
    """Structured parse result for one source file."""

    file_path: str
    language: str = "Python"
    symbols: list[CodeSymbol] = []
    imports: list[ImportInfo] = []
    errors: list[ParseError] = []

    model_config = ConfigDict(use_enum_values=True)
