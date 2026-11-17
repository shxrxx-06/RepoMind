import json

from backend.app.parsing import MethodType, PythonParser, SymbolType


def _by_name(parsed, qualified_name):
    return next(symbol for symbol in parsed.symbols if symbol.qualified_name == qualified_name)


def test_extracts_basic_function_metadata_and_line_numbers():
    source = '"""File docs."""\n\ndef add(a: int, b: int = 2) -> int:\n    """Add values."""\n    return a + b\n'
    parsed = PythonParser().parse(source, "src/math_utils.py")

    function = _by_name(parsed, "add")
    assert function.symbol_type == SymbolType.FUNCTION
    assert (function.start_line, function.end_line) == (3, 5)
    assert function.docstring == "Add values."
    assert function.return_annotation == "int"
    assert [(p.name, p.annotation, p.default) for p in function.parameters] == [
        ("a", "int", None), ("b", "int", "2")
    ]
    assert parsed.file_path == "src/math_utils.py"


def test_extracts_class_methods_and_method_kinds():
    parsed = PythonParser().parse(
        "class UserService(BaseService):\n"
        "    @classmethod\n"
        "    def load(cls, name): pass\n"
        "    @staticmethod\n"
        "    def normalize(value): pass\n"
        "    def logout(self): pass\n",
        "service.py",
    )

    cls = _by_name(parsed, "UserService")
    assert cls.symbol_type == SymbolType.CLASS
    assert cls.bases == ["BaseService"]
    assert cls.start_line == 1
    assert cls.end_line == 6
    assert _by_name(parsed, "UserService.load").method_type == MethodType.CLASS
    assert _by_name(parsed, "UserService.normalize").method_type == MethodType.STATIC
    assert _by_name(parsed, "UserService.logout").method_type == MethodType.INSTANCE


def test_extracts_nested_classes_and_functions():
    parsed = PythonParser().parse(
        "class Outer:\n"
        "    class Inner:\n"
        "        def method(self):\n"
        "            def helper(value): return value\n"
        "            return helper(1)\n"
        "def wrapper():\n"
        "    def nested(): pass\n",
        "nested.py",
    )
    assert _by_name(parsed, "Outer.Inner").symbol_type == SymbolType.CLASS
    assert _by_name(parsed, "Outer.Inner.method").symbol_type == SymbolType.METHOD
    assert _by_name(parsed, "Outer.Inner.method.helper").symbol_type == SymbolType.FUNCTION
    assert _by_name(parsed, "wrapper.nested").symbol_type == SymbolType.FUNCTION


def test_extracts_async_and_decorated_functions():
    parsed = PythonParser().parse(
        "@cache(ttl=10)\nasync def fetch(url: str):\n    return url\n", "api.py"
    )
    function = _by_name(parsed, "fetch")
    assert function.is_async is True
    assert function.decorators == ["cache(ttl=10)"]


def test_extracts_parameter_kinds_annotations_and_defaults():
    parsed = PythonParser().parse(
        "def f(a, /, b: int = 3, *items: str, flag: bool = True, **options: object) -> None:\n pass\n",
        "params.py",
    )
    parameters = _by_name(parsed, "f").parameters
    assert [(p.name, p.kind, p.annotation, p.default) for p in parameters] == [
        ("a", "positional_only", None, None),
        ("b", "positional_or_keyword", "int", "3"),
        ("items", "var_positional", "str", None),
        ("flag", "keyword_only", "bool", "True"),
        ("options", "var_keyword", "object", None),
    ]
    assert _by_name(parsed, "f").return_annotation == "None"


def test_extracts_import_forms_aliases_and_relative_levels():
    parsed = PythonParser().parse(
        "import os\nimport numpy as np\nfrom collections import Counter as C\nfrom .helpers import run\n",
        "pkg/module.py",
    )
    assert [
        (item.module_name, item.symbol_name, item.alias, item.line_number, item.relative_level)
        for item in parsed.imports
    ] == [
        ("os", None, None, 1, 0),
        ("numpy", None, "np", 2, 0),
        ("collections", "Counter", "C", 3, 0),
        ("helpers", "run", None, 4, 1),
    ]


def test_empty_and_unicode_source_parse_without_errors():
    parser = PythonParser()
    assert parser.parse("", "empty.py").errors == []
    parsed = parser.parse('def привет():\n    """München 🌍"""\n', "unicode.py")
    assert _by_name(parsed, "привет").docstring == "München 🌍"


def test_syntax_error_has_line_and_column():
    parsed = PythonParser().parse("def broken(:\n    pass\n", "broken.py")
    error = parsed.errors[0]
    assert error.error_type == "SyntaxError"
    assert error.line == 1
    assert error.column > 0
    assert parsed.symbols == []


def test_parse_file_reports_missing_source(tmp_path):
    result = PythonParser().parse_file(tmp_path / "missing.py")
    assert result.file_path.endswith("missing.py")
    assert result.errors[0].error_type == "FileNotFoundError"


def test_result_serializes_to_json():
    parsed = PythonParser().parse(
        "from typing import Any\nclass Store:\n def get(self, key: str) -> Any: pass\n",
        "store.py",
    )
    payload = json.loads(parsed.model_dump_json())
    assert payload["file_path"] == "store.py"
    assert payload["symbols"][0]["symbol_type"] == "class"
    assert payload["symbols"][1]["qualified_name"] == "Store.get"
    assert payload["imports"][0]["module_name"] == "typing"


def test_nested_definitions_are_reported_in_source_order():
    parsed = PythonParser().parse(
        "def first():\n def inner(): pass\ndef second(): pass\n", "order.py"
    )
    assert [symbol.qualified_name for symbol in parsed.symbols] == [
        "first", "first.inner", "second"
    ]
