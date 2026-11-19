from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from time import sleep

import pytest

from backend.app.embeddings.models import CodeDocument, CodeEmbedding, EmbeddingBatch
from backend.app.graph.models import RepositoryAnalysisResult
from backend.app.parsing.models import ParsedFile
from backend.app.services.investigation_runtime import (
    InvestigationRuntime,
    InvestigationRuntimeError,
)


def _embedding(model="model-a", dimension=3):
    return CodeEmbedding(
        document_id="doc-1",
        text="function f",
        source_file_path="module.py",
        qualified_symbol_name="f",
        symbol_type="function",
        start_line=1,
        end_line=2,
        vector=[1.0] + [0.0] * (dimension - 1),
        model_name=model,
        embedding_dimension=dimension,
    )


class FakeEmbeddings:
    model_name = "model-a"
    known_embedding_dimension = 3

    @property
    def embedding_dimension(self):
        raise AssertionError("Known compatible stores should not force model loading")

    def embed_documents(self, documents):
        return EmbeddingBatch(
            embeddings=[_embedding() for _ in documents],
            model_name=self.model_name,
            embedding_dimension=3,
        )


class FakeStore:
    def __init__(self, loaded):
        self.loaded = loaded
        self.saved = []

    def load(self, path):
        return self.loaded

    def save(self, batch, path):
        self.saved.append((batch, path))


def test_embedding_store_requires_matching_model_and_dimension():
    runtime = InvestigationRuntime()
    runtime.embedding_service = FakeEmbeddings()

    matching = EmbeddingBatch(
        embeddings=[_embedding()], model_name="model-a", embedding_dimension=3
    )
    same_dimension_different_model = EmbeddingBatch(
        embeddings=[_embedding(model="model-b")], model_name="model-b", embedding_dimension=3
    )
    different_dimension = EmbeddingBatch(
        embeddings=[_embedding(dimension=4)], model_name="model-a", embedding_dimension=4
    )

    assert runtime._is_compatible(matching)
    assert not runtime._is_compatible(same_dimension_different_model)
    assert not runtime._is_compatible(different_dimension)


def test_matching_embedding_store_is_reused_without_embedding_or_model_load(tmp_path):
    runtime = InvestigationRuntime()
    runtime.embedding_service = FakeEmbeddings()
    runtime.store = FakeStore(
        EmbeddingBatch(
            embeddings=[_embedding()], model_name="model-a", embedding_dimension=3
        )
    )

    runtime._ensure_embedding_store([], tmp_path, tmp_path / "existing.npz")

    assert runtime.store.saved == []


def test_compatible_store_with_unknown_dimension_loads_model_only_to_validate():
    class UnknownDimensionEmbeddings:
        model_name = "model-a"
        known_embedding_dimension = None
        dimension_loads = 0

        @property
        def embedding_dimension(self):
            self.dimension_loads += 1
            return 3

    runtime = InvestigationRuntime()
    model = UnknownDimensionEmbeddings()
    runtime.embedding_service = model
    batch = EmbeddingBatch(
        embeddings=[_embedding()], model_name="model-a", embedding_dimension=3
    )

    assert runtime._is_compatible(batch)
    assert model.dimension_loads == 1


def test_corrupted_store_is_regenerated_and_saved_atomically_through_store(tmp_path, monkeypatch):
    runtime = InvestigationRuntime()
    runtime.embedding_service = FakeEmbeddings()
    runtime.store = FakeStore(
        EmbeddingBatch(embeddings=[], model_name="", storage_error="corrupted")
    )
    document = CodeDocument(
        document_id="doc-1",
        text="function f",
        source_file_path="module.py",
        qualified_symbol_name="f",
        symbol_type="function",
        start_line=1,
        end_line=2,
    )

    class Preparer:
        def prepare(self, parsed_files, repository_root):
            return [document]

    monkeypatch.setattr(
        "backend.app.services.investigation_runtime.CodeDocumentPreparer", Preparer
    )
    destination = tmp_path / "embeddings.npz"

    runtime._ensure_embedding_store([ParsedFile(file_path="module.py")], tmp_path, destination)

    assert len(runtime.store.saved) == 1
    saved_batch, saved_path = runtime.store.saved[0]
    assert saved_path == destination
    assert saved_batch.model_name == "model-a"
    assert saved_batch.embedding_dimension == 3


@pytest.mark.parametrize(
    "stored_batch",
    [
        EmbeddingBatch(
            embeddings=[_embedding(model="old-model")],
            model_name="old-model",
            embedding_dimension=3,
        ),
        EmbeddingBatch(
            embeddings=[_embedding(dimension=4)],
            model_name="model-a",
            embedding_dimension=4,
        ),
    ],
)
def test_incompatible_model_or_dimension_is_regenerated(tmp_path, monkeypatch, stored_batch):
    runtime = InvestigationRuntime()
    runtime.embedding_service = FakeEmbeddings()
    runtime.store = FakeStore(stored_batch)
    document = CodeDocument(
        document_id="doc-1",
        text="function f",
        source_file_path="module.py",
        qualified_symbol_name="f",
        symbol_type="function",
        start_line=1,
        end_line=2,
    )

    class Preparer:
        def prepare(self, parsed_files, repository_root):
            return [document]

    monkeypatch.setattr(
        "backend.app.services.investigation_runtime.CodeDocumentPreparer", Preparer
    )
    runtime._ensure_embedding_store([], tmp_path, tmp_path / "stale.npz")

    assert len(runtime.store.saved) == 1
    assert runtime.store.saved[0][0].model_name == "model-a"
    assert runtime.store.saved[0][0].embedding_dimension == 3


def test_empty_repository_saves_empty_store_without_loading_model(tmp_path):
    runtime = InvestigationRuntime()
    runtime.embedding_service = FakeEmbeddings()
    runtime.store = FakeStore(
        EmbeddingBatch(embeddings=[], model_name="", storage_error="missing")
    )

    runtime._ensure_embedding_store([], tmp_path, tmp_path / "empty.npz")

    batch, _ = runtime.store.saved[0]
    assert batch.embeddings == []
    assert batch.model_name == "model-a"
    assert batch.embedding_dimension is None


def test_same_repository_initializes_once_and_shares_cached_service(tmp_path, monkeypatch):
    runtime = InvestigationRuntime()
    repository = tmp_path / "repo"
    repository.mkdir()
    calls = 0
    calls_lock = Lock()

    def prepare(repository_id, repository_root):
        nonlocal calls
        with calls_lock:
            calls += 1
        sleep(0.04)
        return RepositoryAnalysisResult(), tmp_path / "store.npz"

    monkeypatch.setattr(runtime, "_prepare_repository", prepare)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: runtime.for_repository("repo", repository), range(6)))

    assert calls == 1
    assert all(result is results[0] for result in results)


def test_different_repositories_initialize_concurrently(tmp_path, monkeypatch):
    runtime = InvestigationRuntime()
    repositories = [tmp_path / "first", tmp_path / "second"]
    for repository in repositories:
        repository.mkdir()
    barrier = Barrier(2)

    def prepare(repository_id, repository_root):
        barrier.wait(timeout=2)
        return RepositoryAnalysisResult(), tmp_path / f"{repository_id}.npz"

    monkeypatch.setattr(runtime, "_prepare_repository", prepare)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda args: runtime.for_repository(*args),
                [("first", repositories[0]), ("second", repositories[1])],
            )
        )

    assert len(results) == 2
    assert results[0] is not results[1]


def test_failed_initialization_releases_repository_lock(tmp_path, monkeypatch):
    runtime = InvestigationRuntime()
    repository = tmp_path / "repo"
    repository.mkdir()
    attempts = 0

    def prepare(repository_id, repository_root):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("failed once")
        return RepositoryAnalysisResult(), tmp_path / "store.npz"

    monkeypatch.setattr(runtime, "_prepare_repository", prepare)
    with pytest.raises(InvestigationRuntimeError):
        runtime.for_repository("repo", repository)
    result = runtime.for_repository("repo", repository)

    assert result is not None
    assert attempts == 2


def test_graph_analysis_is_cached_without_initializing_embedding_model(tmp_path, monkeypatch):
    runtime = InvestigationRuntime()
    repository = tmp_path / "repo"
    repository.mkdir()
    analysis = RepositoryAnalysisResult()
    calls = 0

    def analyze(repository_root):
        nonlocal calls
        calls += 1
        return [], analysis

    monkeypatch.setattr(runtime, "_analyze_repository", analyze)
    first = runtime.analysis_for_repository("repo", repository)
    second = runtime.analysis_for_repository("repo", repository)

    assert first is analysis
    assert second is analysis
    assert calls == 1
    assert runtime._services == {}
    assert runtime.embedding_service.known_embedding_dimension is None
