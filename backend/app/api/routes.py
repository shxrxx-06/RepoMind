import json
import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, HttpUrl

from backend.app.core.config import settings
from backend.app.ingestion.github import clone_github_repository
from backend.app.ingestion.scanner import scan_repository
from backend.app.ingestion.zip_handler import extract_repository_zip

router = APIRouter(prefix="/api/v1", tags=["Repository"])
logger = logging.getLogger(__name__)


class GitHubRequest(BaseModel):
    url: HttpUrl


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "RepoMind API"}


@router.post("/repositories/upload")
async def upload_repository(file: UploadFile = File(...)) -> dict:  # noqa: B008
    if not file.filename or Path(file.filename).suffix.lower() != ".zip":
        raise HTTPException(400, "Only ZIP files are supported")
    limit = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    temp_path: Path | None = None
    repository_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=settings.TEMP_DIR, suffix=".zip", delete=False) as temp:
            temp_path = Path(temp.name)
            size = 0
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, "ZIP exceeds the upload size limit")
                temp.write(chunk)
        repository_path = extract_repository_zip(temp_path)
        metadata = scan_repository(repository_path)
        return {
            "message": "Repository uploaded and ingested successfully",
            "repository_id": repository_path.name,
            "metadata": metadata,
        }
    except HTTPException:
        if repository_path:
            shutil.rmtree(repository_path, ignore_errors=True)
        raise
    except ValueError as exc:
        if repository_path:
            shutil.rmtree(repository_path, ignore_errors=True)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("ZIP repository ingestion failed")
        if repository_path:
            shutil.rmtree(repository_path, ignore_errors=True)
        raise HTTPException(500, "Unable to process repository ZIP") from exc
    finally:
        await file.close()
        if temp_path:
            temp_path.unlink(missing_ok=True)


@router.post("/repositories/github")
def ingest_github_repository(request: GitHubRequest) -> dict:
    repository_path: Path | None = None
    try:
        logger.info("GitHub endpoint: starting ingestion for %s", request.url)
        repository_path = clone_github_repository(str(request.url))
        logger.info("GitHub endpoint: starting scanner for %s", repository_path)
        metadata = scan_repository(repository_path)
        logger.info(
            "GitHub endpoint: scanner completed for %s (%d files)",
            repository_path,
            metadata["total_files"],
        )
        logger.info("GitHub endpoint: validating metadata JSON serialization")
        json.dumps(metadata)
        logger.info("GitHub endpoint: metadata JSON serialization succeeded")
        logger.info("GitHub endpoint: constructing response")
        response = {
            "message": "GitHub repository ingested successfully",
            "repository_id": repository_path.name,
            "metadata": metadata,
        }
        jsonable_encoder(response)
        logger.info(
            "GitHub endpoint: response constructed for repository %s",
            repository_path.name,
        )
        return response
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("GitHub repository ingestion failed")
        if repository_path:
            shutil.rmtree(repository_path, ignore_errors=True)
        raise HTTPException(502, "Unable to clone or scan GitHub repository") from exc
