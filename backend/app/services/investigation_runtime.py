"""Build and cache the existing investigation services for ingested repositories."""

from hashlib import sha256
from pathlib import Path
from threading import Lock

from backend.app.core.config import settings
from backend.app.embeddings.models import EmbeddingBatch
from backend.app.embeddings.service import EmbeddingService
from backend.app.graph.models import RepositoryAnalysisResult
from backend.app.ingestion.scanner import scan_repository
from backend.app.intelligence.document_preparer import CodeDocumentPreparer
from backend.app.intelligence.repository_analyzer import RepositoryAnalyzer
from backend.app.investigation.service import InvestigationService
from backend.app.parsing.models import ParsedFile
from backend.app.parsing.python_parser import PythonParser
from backend.app.rag.provider import configured_provider
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.storage.local_embedding_store import LocalEmbeddingStore


class InvestigationRuntimeError(RuntimeError):
    """Raised when repository intelligence cannot be prepared."""


class RepositorySnapshotChanged(InvestigationRuntimeError):
    """Raised when repository contents change while an analysis is running."""


class InvestigationRuntime:
    """Lazily prepare local repository intelligence and reuse it across requests."""

    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        self.store = LocalEmbeddingStore()
        self._services: dict[str, InvestigationService] = {}
        self._analyses: dict[str, RepositoryAnalysisResult] = {}
        self._parsed_files: dict[str, list[ParsedFile]] = {}
        self._fingerprints: dict[str, str] = {}
        self._repository_locks: dict[str, Lock] = {}
        self._repository_locks_guard = Lock()

    def for_repository(self, repository_id: str, repository_root: Path) -> InvestigationService:
        repository_root = repository_root.resolve(strict=True)
        if not repository_root.is_dir():
            raise InvestigationRuntimeError("Repository directory is unavailable")

        with self._repository_locks_guard:
            lock = self._repository_locks.setdefault(repository_id, Lock())
        with lock:
            try:
                fingerprint = self._repository_fingerprint(repository_root)
            except Exception as exc:
                raise InvestigationRuntimeError("Repository snapshot could not be checked") from exc
            if self._fingerprints.get(repository_id) != fingerprint:
                self._clear_repository_cache(repository_id)
            service = self._services.get(repository_id)
            if service is not None:
                return service

            try:
                analysis, store_path = self._prepare_repository(
                    repository_id,
                    repository_root,
                    fingerprint,
                )
                retriever = HybridRetriever(store_path, self.embedding_service, store=self.store)
                service = InvestigationService(
                    retriever,
                    analysis,
                    repository_root=str(repository_root),
                    provider=configured_provider(settings),
                )
            except InvestigationRuntimeError:
                raise
            except Exception as exc:
                raise InvestigationRuntimeError(
                    "Repository intelligence could not be prepared"
                ) from exc

            self._services[repository_id] = service
            self._analyses[repository_id] = analysis
            self._fingerprints[repository_id] = fingerprint
            return service

    def analysis_for_repository(
        self,
        repository_id: str,
        repository_root: Path,
    ) -> RepositoryAnalysisResult:
        """Return the graph analysis cached with the repository investigation service."""
        repository_root = repository_root.resolve(strict=True)
        if not repository_root.is_dir():
            raise InvestigationRuntimeError("Repository directory is unavailable")
        with self._repository_locks_guard:
            lock = self._repository_locks.setdefault(repository_id, Lock())
        with lock:
            try:
                fingerprint = self._repository_fingerprint(repository_root)
            except Exception as exc:
                raise InvestigationRuntimeError("Repository snapshot could not be checked") from exc
            if self._fingerprints.get(repository_id) != fingerprint:
                self._clear_repository_cache(repository_id)
            analysis = self._analyses.get(repository_id)
            if analysis is None:
                try:
                    parsed_files, analysis = self._analyze_repository(repository_root)
                except Exception as exc:
                    raise InvestigationRuntimeError(
                        "Repository graph could not be generated"
                    ) from exc
                self._parsed_files[repository_id] = parsed_files
                self._analyses[repository_id] = analysis
            self._fingerprints[repository_id] = fingerprint
            return analysis

    def invalidate_repository(self, repository_id: str) -> None:
        """Explicitly discard this process's parsed, graph, and service caches."""
        with self._repository_locks_guard:
            lock = self._repository_locks.setdefault(repository_id, Lock())
        with lock:
            self._clear_repository_cache(repository_id)

    @staticmethod
    def repository_fingerprint(repository_root: Path) -> str:
        """Return the current source fingerprint or a safe runtime error."""
        try:
            resolved_root = repository_root.resolve(strict=True)
            if not resolved_root.is_dir():
                raise OSError("Repository directory is unavailable")
            return InvestigationRuntime._repository_fingerprint(resolved_root)
        except Exception as exc:
            raise InvestigationRuntimeError("Repository snapshot could not be checked") from exc

    def _clear_repository_cache(self, repository_id: str) -> None:
        self._services.pop(repository_id, None)
        self._analyses.pop(repository_id, None)
        self._parsed_files.pop(repository_id, None)
        self._fingerprints.pop(repository_id, None)

    def _prepare_repository(
        self,
        repository_id: str,
        repository_root: Path,
        fingerprint: str,
    ) -> tuple[RepositoryAnalysisResult, Path]:
        intelligence_dir = settings.DATA_DIR / "intelligence"
        store_path = intelligence_dir / f"{repository_id}-{fingerprint[:16]}.npz"
        intelligence_dir.mkdir(parents=True, exist_ok=True)

        parsed_files = self._parsed_files.get(repository_id)
        analysis = self._analyses.get(repository_id)
        if parsed_files is None or analysis is None:
            parsed_files, analysis = self._analyze_repository(repository_root)
            self._parsed_files[repository_id] = parsed_files
            self._analyses[repository_id] = analysis
        self._ensure_embedding_store(parsed_files, repository_root, store_path)

        return analysis, store_path

    @staticmethod
    def _repository_fingerprint(repository_root: Path) -> str:
        """Hash scanned Python paths and source bytes for snapshot-aware cache reuse."""
        digest = sha256()
        scan = scan_repository(repository_root)
        python_files = sorted(
            item["path"]
            for item in scan["files"]
            if item["language"] == "Python"
        )
        for relative_path in python_files:
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            source_path = repository_root.joinpath(*Path(relative_path).parts)
            with source_path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _analyze_repository(
        repository_root: Path,
    ) -> tuple[list[ParsedFile], RepositoryAnalysisResult]:
        parsed_files = []
        parser = PythonParser()
        scan = scan_repository(repository_root)
        for item in scan["files"]:
            if item["language"] != "Python":
                continue
            source_path = repository_root.joinpath(*Path(item["path"]).parts)
            parsed = parser.parse_file(source_path)
            parsed.file_path = source_path.relative_to(repository_root).as_posix()
            parsed_files.append(parsed)
        analysis = RepositoryAnalyzer().analyze(repository_root, parsed_files)
        return parsed_files, analysis

    def _ensure_embedding_store(
        self,
        parsed_files: list[ParsedFile],
        repository_root: Path,
        store_path: Path,
    ) -> None:
        try:
            existing = self.store.load(store_path)
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            existing = EmbeddingBatch(
                embeddings=[],
                model_name="",
                storage_error=f"Unable to read stored embedding collection: {type(exc).__name__}",
            )
        if self._is_compatible(existing):
            return
        documents = CodeDocumentPreparer().prepare(
            parsed_files,
            repository_root=repository_root,
        )
        if documents:
            batch = self.embedding_service.embed_documents(documents)
        else:
            batch = EmbeddingBatch(embeddings=[], model_name=self.embedding_service.model_name)
        self.store.save(batch, store_path)

    def _is_compatible(self, batch: EmbeddingBatch) -> bool:
        """Check model and vector dimensions; regenerate unreadable or stale stores."""
        if batch.storage_error or batch.model_name != self.embedding_service.model_name:
            return False
        if not batch.embeddings:
            return True

        dimensions = {item.embedding_dimension for item in batch.embeddings}
        if len(dimensions) != 1 or batch.embedding_dimension not in dimensions:
            return False

        expected_dimension = self.embedding_service.known_embedding_dimension
        if expected_dimension is None:
            expected_dimension = self.embedding_service.embedding_dimension
        return batch.embedding_dimension == expected_dimension
