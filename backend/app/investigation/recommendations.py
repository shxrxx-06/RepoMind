"""Recommendations grounded only in test files that exist in the repository."""

from pathlib import Path

from backend.app.investigation.models import TestRecommendation


class TestRecommender:
    """Find existing Python test files that mention investigated symbols."""

    def recommend(
        self, repository_root: str | Path | None, symbol_names: set[str]
    ) -> list[TestRecommendation]:
        if repository_root is None:
            return [TestRecommendation(reason="No matching tests were found; repository root was not provided.")]
        root = Path(repository_root)
        if not root.is_dir() or not symbol_names:
            return [TestRecommendation(reason="No matching tests were found in the repository.")]
        recommendations: list[TestRecommendation] = []
        for path in sorted(root.rglob("test*.py")):
            if any(
                part in {".git", ".venv", "venv", "__pycache__", "site-packages"}
                for part in path.parts
            ):
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            matches = sorted(name for name in symbol_names if name and name in content)
            if matches:
                recommendations.append(
                    TestRecommendation(
                        test_file_path=path.relative_to(root).as_posix(),
                        matched_symbols=matches,
                        reason="Existing test source references the investigated symbol.",
                    )
                )
        if not recommendations:
            return [TestRecommendation(reason="No matching tests were found in the repository.")]
        return recommendations
