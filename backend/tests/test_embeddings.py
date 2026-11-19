import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.embeddings import (
    DEFAULT_MODEL_NAME,
    CodeDocument,
    EmbeddingGenerationError,
    EmbeddingModelLoadError,
    EmbeddingService,
)
from backend.app.embeddings.models import EmbeddingBatch
from backend.app.intelligence.document_preparer import CodeDocumentPreparer
from backend.app.parsing.python_parser import PythonParser
from backend.app.storage.local_embedding_store import LocalEmbeddingStore


def _document(document_id="doc-1", text="UserService.authenticate"):
    return CodeDocument(
        document_id=document_id,
        text=text,
        source_file_path="app/services/user.py",
        qualified_symbol_name="UserService.authenticate",
        symbol_type="method",
        start_line=12,
        end_line=19,
        source_snippet="def authenticate(self, token: str) -> bool: ...",
        docstring="Validate a token.",
    )


class FakeModel:
    def __init__(self, dimension=3):
        self.dimension = dimension
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        base_vectors = [[3.0, 4.0, 0.0], [0.0, 5.0, 12.0], [8.0, 0.0, 6.0]]
        return np.asarray([base_vectors[index % len(base_vectors)] for index, _ in enumerate(texts)])


def test_prepares_independent_class_and_function_documents_with_metadata(tmp_path):
    source = (
        'class UserService(BaseService):\n'
        '    """Account operations."""\n'
        '    @cache\n'
        '    def authenticate(self, token: str) -> bool:\n'
        '        """Validate a token."""\n'
        '        return bool(token)\n'
    )
    path = tmp_path / "app/services/user.py"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    parsed = PythonParser().parse(source, "app/services/user.py")

    documents = CodeDocumentPreparer().prepare(
        [parsed], repository_root=tmp_path
    )

    assert len(documents) == 2
    class_doc, method_doc = documents
    assert class_doc.qualified_symbol_name == "UserService"
    assert class_doc.bases == ["BaseService"]
    assert class_doc.docstring == "Account operations."
    assert method_doc.qualified_symbol_name == "UserService.authenticate"
    assert method_doc.source_file_path == "app/services/user.py"
    assert method_doc.parameters[1].annotation == "str"
    assert method_doc.return_annotation == "bool"
    assert method_doc.decorators == ["cache"]
    assert "Validate a token" in method_doc.text
    assert "return bool(token)" in method_doc.source_snippet
    assert class_doc.document_id != method_doc.document_id


def test_preparer_uses_supplied_source_and_is_deterministic():
    parsed = PythonParser().parse("def run():\n    return 1\n", "main.py")
    preparer = CodeDocumentPreparer()
    first = preparer.prepare([parsed], {"main.py": "def run():\n    return 1\n"})
    second = preparer.prepare([parsed], {"main.py": "def run():\n    return 1\n"})
    assert first[0].document_id == second[0].document_id
    assert first[0].source_snippet == "def run():\n    return 1"


def test_empty_input_does_not_load_the_model():
    loader_calls = []

    def loader(name, device):
        loader_calls.append((name, device))
        return FakeModel()

    service = EmbeddingService(model_loader=loader)
    result = service.embed_documents([])
    assert result.embeddings == []
    assert result.model_name == DEFAULT_MODEL_NAME
    assert result.embedding_dimension is None
    assert result.processing_time_seconds == 0
    assert loader_calls == []


def test_single_and_batch_embeddings_are_normalized_and_reuse_model():
    model = FakeModel()
    loader_calls = []

    def loader(name, device):
        loader_calls.append((name, device))
        return model

    service = EmbeddingService(model_name="test/model", device="cpu", model_loader=loader)
    single = service.embed_document(_document())
    batch = service.embed_documents([_document("doc-2"), _document("doc-3")], batch_size=2)

    assert single.vector == pytest.approx([0.6, 0.8, 0.0])
    assert [np.linalg.norm(item.vector) for item in batch.embeddings] == pytest.approx([1.0, 1.0])
    assert batch.embedding_dimension == service.embedding_dimension == 3
    assert batch.model_name == "test/model"
    assert model.calls[1][1]["batch_size"] == 2
    assert model.calls[1][1]["normalize_embeddings"] is True
    assert len(loader_calls) == 1


def test_embedding_dimension_mismatch_is_reported():
    service = EmbeddingService(model_loader=lambda name, device: FakeModel(dimension=4))
    with pytest.raises(EmbeddingGenerationError, match="expected 4"):
        service.embed_document(_document())


def test_model_loading_failure_has_actionable_error():
    def failing_loader(name, device):
        raise OSError("model cache unavailable")

    service = EmbeddingService(model_loader=failing_loader)
    with pytest.raises(EmbeddingModelLoadError, match="model cache unavailable"):
        service.embed_document(_document())


def test_default_loader_is_lazy_and_shared_across_service_instances(monkeypatch):
    loads = []
    model = FakeModel()

    def sentence_transformer(model_name, device):
        loads.append((model_name, device))
        return model

    EmbeddingService._model_cache.clear()
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=sentence_transformer),
    )
    first = EmbeddingService(model_name="lazy/test", device="cpu")
    second = EmbeddingService(model_name="lazy/test", device="cpu")
    assert loads == []
    assert first.embedding_dimension == 3
    assert second.embedding_dimension == 3
    assert loads == [("lazy/test", "cpu")]


def test_embedding_batch_and_document_are_json_serializable():
    service = EmbeddingService(model_loader=lambda name, device: FakeModel())
    result = service.embed_documents([_document()])
    payload = json.loads(result.model_dump_json())
    assert payload["embeddings"][0]["document_id"] == "doc-1"
    assert len(payload["embeddings"][0]["vector"]) == 3


def test_local_store_round_trips_vectors_and_metadata(tmp_path):
    service = EmbeddingService(model_loader=lambda name, device: FakeModel())
    batch = service.embed_documents([_document("one"), _document("two", "UserService.logout")])
    path = tmp_path / "embeddings.npz"
    store = LocalEmbeddingStore()

    store.save(batch, path)
    loaded = store.load(path)

    assert loaded.storage_error is None
    assert loaded.model_name == batch.model_name
    assert loaded.embedding_dimension == 3
    assert [item.document_id for item in loaded.embeddings] == ["one", "two"]
    assert loaded.embeddings[1].qualified_symbol_name == "UserService.authenticate"
    assert loaded.embeddings[0].vector == pytest.approx(batch.embeddings[0].vector)
    with np.load(path, allow_pickle=False) as data:
        assert data["vectors"].shape == (2, 3)


def test_local_store_handles_missing_and_corrupted_files(tmp_path):
    store = LocalEmbeddingStore()
    missing = store.load(tmp_path / "missing.npz")
    assert missing.embeddings == []
    assert "not found" in missing.storage_error

    corrupt_path = tmp_path / "corrupt.npz"
    corrupt_path.write_bytes(b"not a numpy archive")
    corrupted = store.load(corrupt_path)
    assert corrupted.embeddings == []
    assert corrupted.storage_error is not None


def test_local_store_can_save_and_load_an_empty_collection(tmp_path):
    batch = EmbeddingBatch(embeddings=[], model_name="test/model", embedding_dimension=3)
    store = LocalEmbeddingStore()
    path = tmp_path / "empty.npz"

    store.save(batch, path)
    loaded = store.load(path)

    assert loaded.embeddings == []
    assert loaded.embedding_dimension == 3
    assert loaded.storage_error is None
