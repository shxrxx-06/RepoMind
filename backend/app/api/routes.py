import json
import logging
import shutil
import tempfile
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, HttpUrl

from backend.app.core.config import settings
from backend.app.graph.models import RepositoryAnalysisResult
from backend.app.ingestion.github import clone_github_repository
from backend.app.ingestion.scanner import scan_repository
from backend.app.ingestion.zip_handler import extract_repository_zip
from backend.app.investigation.models import InvestigationRequest, InvestigationResponse
from backend.app.services.investigation_runtime import (
    InvestigationRuntime,
    InvestigationRuntimeError,
)
from backend.app.services.repository_catalog import (
    load_repository_summary,
    save_repository_summary,
)

router = APIRouter(prefix="/api/v1", tags=["Repository"])
logger = logging.getLogger(__name__)
investigation_runtime = InvestigationRuntime()


class GitHubRequest(BaseModel):
    url: HttpUrl


class RepositoryListItem(BaseModel):
    repository_id: str
    total_files: int
    language_distribution: dict[str, int]


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "RepoMind API"}


@router.get("/repositories", response_model=list[RepositoryListItem])
def list_repositories() -> list[RepositoryListItem]:
    """List repository directories that were successfully ingested."""
    repositories = []
    for path in sorted(settings.REPOSITORIES_DIR.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        try:
            UUID(path.name)
            resolved_path = path.resolve()
            resolved_path.relative_to(settings.REPOSITORIES_DIR.resolve())
            metadata = load_repository_summary(path.name)
            if metadata is None:
                metadata = scan_repository(resolved_path)
                _persist_repository_summary(path.name, metadata)
        except (ValueError, OSError):
            continue
        repositories.append(
            RepositoryListItem(
                repository_id=path.name,
                total_files=metadata["total_files"],
                language_distribution=metadata["language_distribution"],
            )
        )
    return repositories


def _persist_repository_summary(repository_id: str, metadata: dict) -> None:
    try:
        save_repository_summary(repository_id, metadata)
    except Exception:
        logger.exception("Unable to persist repository listing metadata")


@router.post(
    "/repositories/{repository_id}/investigate",
    response_model=InvestigationResponse,
)
def investigate_repository(
    repository_id: str,
    request: InvestigationRequest,
) -> InvestigationResponse:
    """Run the existing investigation service against an ingested repository."""
    try:
        UUID(repository_id)
    except ValueError as exc:
        raise HTTPException(404, "Repository was not found") from exc
    repository_path = (settings.REPOSITORIES_DIR / repository_id).resolve()
    try:
        repository_path.relative_to(settings.REPOSITORIES_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(404, "Repository was not found") from exc
    if not repository_path.is_dir():
        raise HTTPException(404, "Repository was not found")
    try:
        service = investigation_runtime.for_repository(repository_id, repository_path)
        response = service.investigate(request)
        return response
    except InvestigationRuntimeError as exc:
        logger.exception("Unable to prepare repository investigation")
        raise HTTPException(503, "Repository intelligence is unavailable for investigation") from exc
    except Exception as exc:
        logger.exception("Repository investigation failed")
        raise HTTPException(500, "Repository investigation failed") from exc


@router.get(
    "/repositories/{repository_id}/graph",
    response_model=RepositoryAnalysisResult,
)
def get_repository_graph(repository_id: str) -> RepositoryAnalysisResult:
    """Return the existing serialized analysis graph for an ingested repository."""
    try:
        UUID(repository_id)
    except ValueError as exc:
        raise HTTPException(404, "Repository was not found") from exc
    repository_path = (settings.REPOSITORIES_DIR / repository_id).resolve()
    try:
        repository_path.relative_to(settings.REPOSITORIES_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(404, "Repository was not found") from exc
    if not repository_path.is_dir():
        raise HTTPException(404, "Repository was not found")
    try:
        analysis = investigation_runtime.analysis_for_repository(
            repository_id,
            repository_path,
        )
        return _repository_graph_response(analysis, repository_path)
    except InvestigationRuntimeError as exc:
        logger.exception("Unable to prepare repository graph")
        raise HTTPException(503, "Repository graph is unavailable") from exc
    except Exception as exc:
        logger.exception("Repository graph generation failed")
        raise HTTPException(500, "Repository graph generation failed") from exc


def _repository_graph_response(
    analysis: RepositoryAnalysisResult,
    repository_path: Path,
) -> RepositoryAnalysisResult:
    """Keep graph errors useful without returning local absolute filesystem paths."""
    root_forms = {str(repository_path), repository_path.as_posix()}
    errors = []
    for error in analysis.errors:
        message = error.message
        if any(root and root.casefold() in message.casefold() for root in root_forms):
            message = "Python source could not be parsed or read."
        errors.append(error.model_copy(update={"message": message}))
    return analysis.model_copy(update={"errors": errors})


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
        response = {
            "message": "Repository uploaded and ingested successfully",
            "repository_id": repository_path.name,
            "metadata": metadata,
        }
        _persist_repository_summary(repository_path.name, metadata)
        return response
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
        _persist_repository_summary(repository_path.name, metadata)
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
