import logging
import uuid
from pathlib import Path
from urllib.parse import urlparse

from git import GitCommandError, Repo

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


def validate_github_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme != "https":
        raise ValueError("Only HTTPS GitHub URLs are supported")
    if parsed.hostname != "github.com":
        raise ValueError("Only github.com repositories are supported")
    if parsed.username or parsed.password or parsed.port:
        raise ValueError("Credentials and custom ports are not allowed")
    if parsed.query or parsed.fragment:
        raise ValueError("Query strings and fragments are not allowed")
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) != 2:
        raise ValueError("Expected URL format: https://github.com/owner/repo")
    owner, repository = parts
    repository = repository.removesuffix(".git")
    def valid_part(part: str) -> bool:
        return bool(part) and part.replace("-", "").replace("_", "").isalnum()

    if not valid_part(owner) or not valid_part(repository):
        raise ValueError("Invalid repository URL")
    return f"https://github.com/{owner}/{repository}.git"


def clone_github_repository(url: str) -> Path:
    logger.info("GitHub ingestion: validating URL")
    clone_url = validate_github_url(url)
    logger.info("GitHub ingestion: URL validation succeeded for %s", clone_url)
    logger.info("GitHub ingestion: generating repository ID")
    repository_id = str(uuid.uuid4())
    logger.info("GitHub ingestion: generated repository ID %s", repository_id)
    destination = settings.REPOSITORIES_DIR / repository_id
    logger.info("GitHub ingestion: creating clone destination %s", destination)
    destination.mkdir(parents=True, exist_ok=False)
    logger.info("GitHub ingestion: clone destination created %s", destination)
    try:
        logger.info("GitHub ingestion: starting Git clone from %s", clone_url)
        Repo.clone_from(
            clone_url,
            destination,
            depth=1,
            multi_options=["--filter=blob:none"],
        )
        logger.info("GitHub ingestion: Git clone completed at %s", destination)
    except (GitCommandError, OSError, ValueError):
        import shutil

        shutil.rmtree(destination, ignore_errors=True)
        raise
    logger.info("GitHub ingestion: returning clone destination %s", destination)
    return destination
