"""Resolve Python import statements against an ingested repository."""

from pathlib import PurePosixPath

from backend.app.graph.models import DependencyInfo, DependencyResolution
from backend.app.parsing.models import ImportInfo, ParsedFile


def normalize_repo_path(path: str) -> str:
    """Normalize repository paths to portable POSIX relative paths."""
    return PurePosixPath(path.replace("\\", "/")).as_posix()


class DependencyAnalyzer:
    """Resolve parsed imports to source files without assuming all imports are local."""

    def analyze(self, parsed_files: list[ParsedFile]) -> list[DependencyInfo]:
        """Return one resolution record per import binding in deterministic order."""
        paths = sorted({normalize_repo_path(item.file_path) for item in parsed_files})
        module_index: dict[str, list[str]] = {}
        for path in paths:
            module = self._module_for_path(path)
            if module:
                module_index.setdefault(module, []).append(path)

        top_level_names = {module.split(".", 1)[0] for module in module_index if module}
        results: list[DependencyInfo] = []
        for parsed in sorted(parsed_files, key=lambda item: normalize_repo_path(item.file_path)):
            source = normalize_repo_path(parsed.file_path)
            for imported in parsed.imports:
                module_name = self._absolute_module(source, imported)
                resolved_path = self._resolve_import(module_name, imported, module_index)
                resolution = self._classify(
                    module_name,
                    imported,
                    resolved_path,
                    module_index,
                    top_level_names,
                )
                results.append(
                    DependencyInfo(
                        source_file=source,
                        imported_file=resolved_path if resolution == DependencyResolution.INTERNAL else None,
                        module_name=module_name,
                        symbol_name=imported.symbol_name,
                        alias=imported.alias,
                        line_number=imported.line_number,
                        resolution=resolution,
                    )
                )
        return sorted(
            results,
            key=lambda item: (
                item.source_file,
                item.line_number,
                item.module_name or "",
                item.symbol_name or "",
                item.alias or "",
            ),
        )

    @staticmethod
    def _module_for_path(path: str) -> str:
        pure_path = PurePosixPath(path)
        if pure_path.suffix != ".py":
            return ""
        parts = list(pure_path.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    @classmethod
    def _absolute_module(cls, source: str, imported: ImportInfo) -> str:
        if not imported.relative_level:
            return imported.module_name or ""
        source_module = cls._module_for_path(source)
        package_parts = source_module.split(".") if source_module else []
        if not source.endswith("/__init__.py") and source != "__init__.py" and package_parts:
            package_parts.pop()
        levels_up = max(imported.relative_level - 1, 0)
        if levels_up:
            package_parts = package_parts[: max(0, len(package_parts) - levels_up)]
        if imported.module_name:
            package_parts.extend(imported.module_name.split("."))
        return ".".join(package_parts)

    @classmethod
    def _resolve_import(
        cls,
        module_name: str,
        imported: ImportInfo,
        module_index: dict[str, list[str]],
    ) -> str | None:
        module_candidates: list[str] = []
        if imported.symbol_name and imported.symbol_name != "*":
            module_candidates.append(".".join(filter(None, (module_name, imported.symbol_name))))
        module_candidates.append(module_name)
        for candidate in module_candidates:
            paths = module_index.get(candidate, [])
            if len(paths) == 1:
                return paths[0]
            if len(paths) > 1:
                return None
        return None

    @classmethod
    def _classify(
        cls,
        module_name: str,
        imported: ImportInfo,
        resolved_path: str | None,
        module_index: dict[str, list[str]],
        top_level_names: set[str],
    ) -> DependencyResolution:
        if resolved_path:
            return DependencyResolution.INTERNAL
        candidates = [module_name]
        if imported.symbol_name and imported.symbol_name != "*":
            candidates.insert(0, ".".join(filter(None, (module_name, imported.symbol_name))))
        if any(len(module_index.get(candidate, [])) > 1 for candidate in candidates):
            return DependencyResolution.UNRESOLVED
        if imported.relative_level or not module_name:
            return DependencyResolution.UNRESOLVED
        if module_name.split(".", 1)[0] in top_level_names:
            return DependencyResolution.UNRESOLVED
        return DependencyResolution.EXTERNAL
