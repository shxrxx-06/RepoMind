"""Common interface for language-specific source parsers."""

from abc import ABC, abstractmethod
from pathlib import Path

from backend.app.parsing.models import ParsedFile


class BaseParser(ABC):
    """Interface that lets consumers use different language parsers uniformly."""

    language: str

    @abstractmethod
    def parse(self, source_code: str, file_path: str | Path) -> ParsedFile:
        """Parse source text while preserving its supplied repository path."""

    @abstractmethod
    def parse_file(self, file_path: str | Path) -> ParsedFile:
        """Read and parse a source file, returning errors instead of raising."""
