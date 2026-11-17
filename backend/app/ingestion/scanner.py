import os
from collections import Counter
from pathlib import Path
from typing import Any

from backend.app.core.config import settings

LANGUAGE_MAP = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".java": "Java",
    ".cpp": "C++", ".c": "C", ".h": "C/C++ Header", ".cs": "C#",
    ".go": "Go", ".rs": "Rust", ".php": "PHP", ".rb": "Ruby",
    ".swift": "Swift", ".kt": "Kotlin", ".sql": "SQL",
    ".html": "HTML", ".css": "CSS", ".scss": "SCSS",
    ".json": "JSON", ".yaml": "YAML", ".yml": "YAML",
    ".xml": "XML", ".md": "Markdown", ".sh": "Shell",
    ".ipynb": "Jupyter Notebook",
}

IGNORED_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".pytest_cache", ".ruff_cache", "dist", "build",
    "target", "vendor", "coverage", ".next", ".nuxt",
}
IGNORED_FILES = {".env", ".ds_store", "thumbs.db"}
BINARY_EXTENSIONS = {
    ".7z", ".avif", ".bmp", ".class", ".dll", ".dylib", ".exe", ".gif",
    ".gz", ".ico", ".jpeg", ".jpg", ".pdf", ".png", ".pyc", ".so",
    ".tar", ".tif", ".tiff", ".war", ".webp", ".woff", ".woff2", ".zip",
}


def scan_repository(repository_path: Path) -> dict[str, Any]:
    """Collect portable file metadata, language counts, and directory names."""
    repository_path = repository_path.resolve(strict=True)
    files: list[dict[str, Any]] = []
    directories: set[str] = set()
    language_counts: Counter[str] = Counter()
    total_size_bytes = 0
    skipped_files = 0
    max_file_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024

    for current, dirnames, filenames in os.walk(repository_path, followlinks=False):
        current_path = Path(current)
        dirnames[:] = sorted(
            name for name in dirnames
            if name.lower() not in IGNORED_DIRS
            and not (current_path / name).is_symlink()
        )
        for directory in dirnames:
            directories.add((current_path / directory).relative_to(repository_path).as_posix())

        for filename in sorted(filenames):
            path = current_path / filename
            if (filename.lower() in IGNORED_FILES or filename.lower().startswith(".env.")
                    or path.is_symlink()):
                skipped_files += 1
                continue
            try:
                size = path.stat().st_size
                with path.open("rb") as source:
                    binary_sample = source.read(8192)
            except OSError:
                skipped_files += 1
                continue
            if path.suffix.lower() in BINARY_EXTENSIONS or b"\x00" in binary_sample:
                skipped_files += 1
                continue
            if size > max_file_bytes or len(files) >= settings.MAX_REPOSITORY_FILES:
                skipped_files += 1
                continue
            relative_path = path.relative_to(repository_path).as_posix()
            language = LANGUAGE_MAP.get(path.suffix.lower())
            files.append({"path": relative_path, "size_bytes": size, "language": language})
            total_size_bytes += size
            if language:
                language_counts[language] += 1

    return {
        "total_files": len(files),
        "total_size_bytes": total_size_bytes,
        "skipped_files": skipped_files,
        "language_distribution": dict(sorted(language_counts.items())),
        "directories": sorted(directories),
        "files": files,
    }
