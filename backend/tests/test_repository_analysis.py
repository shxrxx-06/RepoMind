import json

from backend.app.graph.models import (
    CallResolution,
    DependencyResolution,
    GraphNodeType,
    RelationshipType,
)
from backend.app.intelligence.dependency_analyzer import DependencyAnalyzer
from backend.app.intelligence.repository_analyzer import RepositoryAnalyzer
from backend.app.parsing.python_parser import PythonParser


def _parse_sources(root, sources):
    parser = PythonParser()
    parsed = []
    for relative_path, source in sources.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        parsed.append(parser.parse(source, relative_path))
    return parsed


def test_resolves_absolute_and_package_imports(tmp_path):
    parsed = _parse_sources(
        tmp_path,
        {
            "app.py": "from pkg.models import User\nimport pkg\n",
            "pkg/__init__.py": "from .models import User\n",
            "pkg/models.py": "class User: pass\n",
        },
    )
    dependencies = DependencyAnalyzer().analyze(parsed)
    assert [(item.source_file, item.imported_file, item.resolution) for item in dependencies] == [
        ("app.py", "pkg/models.py", DependencyResolution.INTERNAL),
        ("app.py", "pkg/__init__.py", DependencyResolution.INTERNAL),
        ("pkg/__init__.py", "pkg/models.py", DependencyResolution.INTERNAL),
    ]


def test_resolves_relative_imports_and_init_packages(tmp_path):
    parsed = _parse_sources(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/sub/__init__.py": "",
            "pkg/sub/worker.py": "from .. import shared\nfrom . import helpers\n",
            "pkg/shared.py": "",
            "pkg/sub/helpers.py": "",
        },
    )
    dependencies = DependencyAnalyzer().analyze(parsed)
    assert [(item.imported_file, item.resolution) for item in dependencies
            if item.source_file.endswith("worker.py")] == [
        ("pkg/shared.py", DependencyResolution.INTERNAL),
        ("pkg/sub/helpers.py", DependencyResolution.INTERNAL),
    ]


def test_distinguishes_external_and_unresolved_imports(tmp_path):
    parsed = _parse_sources(
        tmp_path,
        {"app.py": "import requests\nimport local\nfrom local.missing import item\n",
         "local/module.py": ""},
    )
    dependencies = DependencyAnalyzer().analyze(parsed)
    assert [item.resolution for item in dependencies] == [
        DependencyResolution.EXTERNAL,
        DependencyResolution.UNRESOLVED,
        DependencyResolution.UNRESOLVED,
    ]
    assert dependencies[0].imported_file is None


def test_circular_dependencies_are_resolved_without_recursion(tmp_path):
    parsed = _parse_sources(tmp_path, {"a.py": "import b\n", "b.py": "import a\n"})
    dependencies = DependencyAnalyzer().analyze(parsed)
    assert {(item.source_file, item.imported_file) for item in dependencies} == {
        ("a.py", "b.py"),
        ("b.py", "a.py"),
    }


def test_extracts_direct_method_and_nested_calls(tmp_path):
    parsed = _parse_sources(
        tmp_path,
        {
            "main.py": (
                "def load_config(): return {}\n"
                "class Service:\n"
                "    def run(self):\n"
                "        def helper(): return load_config()\n"
                "        self.save()\n"
                "        helper()\n"
                "    def save(self): pass\n"
            )
        },
    )
    result = RepositoryAnalyzer().analyze(tmp_path, parsed)
    calls = result.function_calls
    assert [(item.caller_qualified_name, item.called_name, item.resolution_status)
            for item in calls] == [
        ("Service.run.helper", "load_config", CallResolution.RESOLVED),
        ("Service.run", "self.save", CallResolution.RESOLVED),
        ("Service.run", "helper", CallResolution.RESOLVED),
    ]
    self_call = next(item for item in calls if item.called_name == "self.save")
    config_call = next(item for item in calls if item.called_name == "load_config")
    assert self_call.target_qualified_name == "Service.save"
    assert config_call.target_qualified_name == "load_config"
    assert self_call.line_number == 5


def test_resolves_imported_calls_and_preserves_unresolved_methods(tmp_path):
    parsed = _parse_sources(
        tmp_path,
        {
            "main.py": "from helpers import save\ndef run(db):\n    save()\n    db.connect()\n",
            "helpers.py": "def save(): pass\n",
        },
    )
    result = RepositoryAnalyzer().analyze(tmp_path, parsed)
    calls = {item.called_name: item for item in result.function_calls}
    assert calls["save"].resolution_status == CallResolution.RESOLVED
    assert calls["save"].target_file == "helpers.py"
    assert calls["db.connect"].resolution_status == CallResolution.UNRESOLVED
    assert calls["db.connect"].target_qualified_name is None


def test_ambiguous_call_is_not_claimed_as_resolved(tmp_path):
    parsed = _parse_sources(
        tmp_path,
        {"ambiguous.py": "def target(): pass\ndef caller(): target()\ndef target(): pass\n"},
    )
    result = RepositoryAnalyzer().analyze(tmp_path, parsed)
    call = next(item for item in result.function_calls if item.called_name == "target")
    assert call.resolution_status == CallResolution.AMBIGUOUS
    assert call.target_file is None


def test_builds_typed_directed_graph_and_serializes(tmp_path):
    parsed = _parse_sources(
        tmp_path,
        {
            "main.py": "from helper import work\nclass Runner:\n def run(self): work()\n",
            "helper.py": "def work(): pass\n",
        },
    )
    result = RepositoryAnalyzer().analyze(tmp_path, parsed)
    graph = result.to_networkx()
    file_id = "file:main.py"
    class_id = "symbol:main.py:Runner"
    method_id = "symbol:main.py:Runner.run"
    function_id = "symbol:helper.py:work"
    assert graph.is_directed()
    assert graph.nodes[class_id]["node_type"] == GraphNodeType.CLASS
    assert graph.nodes[method_id]["node_type"] == GraphNodeType.METHOD
    assert graph.nodes[function_id]["node_type"] == GraphNodeType.FUNCTION
    assert graph.has_edge(file_id, class_id)
    assert graph.has_edge(class_id, method_id)
    assert graph[file_id][class_id]["relationships"][0]["relationship_type"] == (
        RelationshipType.DEFINES
    )
    assert any(edge.relationship_type == RelationshipType.IMPORTS for edge in result.relationships)
    assert any(edge.relationship_type == RelationshipType.CALLS for edge in result.relationships)
    encoded = json.loads(result.model_dump_json())
    assert encoded["nodes"]
    assert encoded["relationships"]


def test_empty_repository_and_empty_python_file(tmp_path):
    empty_result = RepositoryAnalyzer().analyze(tmp_path)
    assert empty_result.nodes == []
    assert empty_result.dependencies == []
    assert empty_result.function_calls == []
    parsed = _parse_sources(tmp_path, {"empty.py": ""})
    file_result = RepositoryAnalyzer().analyze(tmp_path, parsed)
    assert [node.file_path for node in file_result.nodes] == ["empty.py"]
    assert file_result.errors == []


def test_default_repository_scan_uses_relative_paths(tmp_path):
    _parse_sources(tmp_path, {"pkg/module.py": "def run(): pass\n"})
    result = RepositoryAnalyzer().analyze(tmp_path)
    assert [node.file_path for node in result.nodes if node.node_type == GraphNodeType.FILE] == [
        "pkg/module.py"
    ]


def test_malformed_python_file_is_reported_without_crashing(tmp_path):
    parsed = _parse_sources(tmp_path, {"broken.py": "def broken(:\n"})
    result = RepositoryAnalyzer().analyze(tmp_path, parsed)
    assert result.errors
    assert result.errors[0].error_type == "SyntaxError"
    assert result.errors[0].line == 1
