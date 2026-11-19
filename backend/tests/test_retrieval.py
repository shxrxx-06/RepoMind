import json

import pytest

from backend.app.embeddings.models import CodeEmbedding, EmbeddingBatch
from backend.app.retrieval import HybridRetriever, RetrievalFilters, RetrievalQuery
from backend.app.storage.local_embedding_store import LocalEmbeddingStore

MODEL_NAME = "test/retrieval-model"


class FakeQueryEmbeddingService:
    model_name = MODEL_NAME

    def __init__(self, vector=None):
        self.vector = vector or [1.0, 0.0]
        self.calls = []

    def embed_text(self, text):
        self.calls.append(text)
        return self.vector


def _embedding(
    document_id,
    qualified_name,
    vector,
    *,
    path="src/services/user.py",
    symbol_type="function",
    docstring=None,
    snippet=None,
    start_line=3,
    end_line=8,
):
    return CodeEmbedding(
        document_id=document_id,
        text=f"Python {symbol_type}: {qualified_name}\n{docstring or ''}\n{snippet or ''}",
        source_file_path=path,
        qualified_symbol_name=qualified_name,
        symbol_type=symbol_type,
        start_line=start_line,
        end_line=end_line,
        source_snippet=snippet,
        docstring=docstring,
        vector=vector,
        model_name=MODEL_NAME,
        embedding_dimension=2,
    )


def _engine(tmp_path, documents, query_vector=None):
    store_path = tmp_path / "embeddings.npz"
    LocalEmbeddingStore().save(
        EmbeddingBatch(
            embeddings=documents,
            model_name=MODEL_NAME,
            embedding_dimension=2,
        ),
        store_path,
    )
    service = FakeQueryEmbeddingService(query_vector)
    return HybridRetriever(store_path, service), service


def test_semantic_retrieval_returns_cosine_top_k(tmp_path):
    docs = [
        _embedding("a", "Service.alpha", [1.0, 0.0]),
        _embedding("b", "Service.beta", [0.8, 0.6]),
        _embedding("c", "Service.gamma", [0.0, 1.0]),
    ]
    engine, query_service = _engine(tmp_path, docs)

    result = engine.search(
        RetrievalQuery(query="account service", top_k=2, keyword_weight=0)
    )

    assert [item.document_id for item in result.results] == ["a", "b"]
    assert result.results[0].semantic_score == pytest.approx(1.0)
    assert result.results[0].semantic_rank == 1
    assert result.results[0].source_file_path == "src/services/user.py"
    assert result.results[0].start_line == 3
    assert query_service.calls == ["account service"]


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_semantic_search_skips_non_finite_stored_vectors(tmp_path, invalid_value):
    engine = HybridRetriever(tmp_path / "unused.npz", FakeQueryEmbeddingService())
    invalid = _embedding("invalid", "Service.invalid", [1.0, 0.0]).model_copy(
        update={"vector": [invalid_value, 0.0]}
    )
    valid = _embedding("valid", "Service.valid", [0.8, 0.6])

    ranked, warning = engine._semantic_search("find valid", [invalid, valid], limit=5)

    assert warning is None
    assert [document.document_id for document, _score in ranked] == ["valid"]


def test_valid_vectors_keep_correct_ranking_when_invalid_vectors_are_present(tmp_path):
    engine = HybridRetriever(tmp_path / "unused.npz", FakeQueryEmbeddingService())
    documents = [
        _embedding("nan", "Service.nan", [1.0, 0.0]).model_copy(
            update={"vector": [float("nan"), 0.0]}
        ),
        _embedding("best", "Service.best", [1.0, 0.0]),
        _embedding("infinity", "Service.infinity", [1.0, 0.0]).model_copy(
            update={"vector": [float("-inf"), 0.0]}
        ),
        _embedding("next", "Service.next", [0.8, 0.6]),
    ]

    ranked, warning = engine._semantic_search("find best", documents, limit=5)

    assert warning is None
    assert [document.document_id for document, _score in ranked] == ["best", "next"]
    assert [score for _document, score in ranked] == pytest.approx([1.0, 0.8])


def test_keyword_exact_identifier_and_case_insensitive_match(tmp_path):
    docs = [
        _embedding("exact", "UserService.authenticate", [1.0, 0.0]),
        _embedding(
            "description",
            "UserService.validate",
            [0.0, 1.0],
            docstring="Authenticate an incoming user token.",
        ),
    ]
    engine, query_service = _engine(tmp_path, docs)

    result = engine.search(
        RetrievalQuery(query="AUTHENTICATE", semantic_weight=0, top_k=2)
    )

    assert [item.document_id for item in result.results] == ["exact", "description"]
    assert result.results[0].keyword_rank == 1
    assert "exact symbol-name match" in result.results[0].ranking_explanation
    assert query_service.calls == []


def test_hybrid_rrf_merges_channels_and_rewards_intersection(tmp_path):
    docs = [
        _embedding("semantic-first", "Other.alpha", [1.0, 0.0]),
        _embedding("both-second", "UserService.authenticate", [0.9, 0.1]),
        _embedding(
            "keyword-only",
            "UserService.other",
            [0.0, 1.0],
            docstring="Authenticate a user session.",
        ),
    ]
    engine, _ = _engine(tmp_path, docs)

    result = engine.search(RetrievalQuery(query="authenticate", top_k=3))

    assert [item.document_id for item in result.results] == [
        "both-second",
        "keyword-only",
        "semantic-first",
    ]
    intersection = result.results[0]
    assert intersection.semantic_rank == 2
    assert intersection.keyword_rank == 1
    assert intersection.hybrid_score > result.results[1].hybrid_score
    assert "reciprocal rank fusion" in intersection.ranking_explanation.lower()
    assert len({item.document_id for item in result.results}) == len(result.results)


def test_duplicate_document_ids_are_removed(tmp_path):
    docs = [
        _embedding("duplicate", "Service.first", [1.0, 0.0]),
        _embedding("duplicate", "Service.second", [0.0, 1.0]),
    ]
    engine, _ = _engine(tmp_path, docs)
    result = engine.search(RetrievalQuery(query="service", top_k=10, keyword_weight=0))
    assert len(result.results) == 1
    assert result.results[0].document_id == "duplicate"


def test_metadata_filters_apply_before_ranking(tmp_path):
    docs = [
        _embedding("one", "UserService.authenticate", [1.0, 0.0]),
        _embedding(
            "two",
            "AdminService.authenticate",
            [0.9, 0.1],
            path="src/admin/service.py",
            symbol_type="method",
        ),
    ]
    engine, _ = _engine(tmp_path, docs)

    result = engine.search(
        RetrievalQuery(
            query="authenticate",
            filters=RetrievalFilters(
                path_prefix="SRC/services",
                symbol_type="FUNCTION",
                qualified_name="UserService.authenticate",
            ),
        )
    )

    assert result.total_documents == 2
    assert result.filtered_documents == 1
    assert [item.document_id for item in result.results] == ["one"]


def test_empty_query_and_empty_store_are_graceful(tmp_path):
    empty_path = tmp_path / "empty.npz"
    LocalEmbeddingStore().save(
        EmbeddingBatch(embeddings=[], model_name=MODEL_NAME, embedding_dimension=2),
        empty_path,
    )
    service = FakeQueryEmbeddingService()
    engine = HybridRetriever(empty_path, service)

    blank_query = engine.search(RetrievalQuery(query="   "))
    empty_store = engine.search(RetrievalQuery(query="find handler"))

    assert blank_query.results == []
    assert "empty" in blank_query.warnings[0].lower()
    assert empty_store.results == []
    assert "no code documents" in empty_store.warnings[0]
    assert service.calls == []


def test_missing_and_corrupted_stores_return_warnings(tmp_path):
    service = FakeQueryEmbeddingService()
    missing = HybridRetriever(tmp_path / "missing.npz", service).search(
        RetrievalQuery(query="user")
    )
    corrupt_path = tmp_path / "corrupt.npz"
    corrupt_path.write_bytes(b"not an archive")
    corrupted = HybridRetriever(corrupt_path, service).search(
        RetrievalQuery(query="user")
    )

    assert missing.results == []
    assert "not found" in missing.warnings[0]
    assert corrupted.results == []
    assert "Unable to load embedding store" in corrupted.warnings[0]
    assert service.calls == []


def test_tied_scores_have_stable_document_id_order(tmp_path):
    docs = [
        _embedding("z-doc", "WidgetService.run", [1.0, 0.0]),
        _embedding("a-doc", "WidgetService.run", [1.0, 0.0]),
    ]
    engine, _ = _engine(tmp_path, docs)
    query = RetrievalQuery(query="widget", top_k=2)

    first = engine.search(query)
    second = engine.search(query)

    assert [item.document_id for item in first.results] == ["a-doc", "z-doc"]
    assert [item.document_id for item in second.results] == ["a-doc", "z-doc"]


def test_retrieval_response_preserves_source_and_serializes(tmp_path):
    doc = _embedding(
        "source-doc",
        "UserService.authenticate",
        [1.0, 0.0],
        path="backend/app/user.py",
        start_line=21,
        end_line=29,
        snippet="return verify(token)",
    )
    engine, _ = _engine(tmp_path, [doc])

    response = engine.search(RetrievalQuery(query="authenticate"))

    item = response.results[0]
    assert item.source_file_path == "backend/app/user.py"
    assert (item.start_line, item.end_line) == (21, 29)
    assert item.source_content == "return verify(token)"
    assert json.loads(response.model_dump_json())["results"][0]["document_id"] == "source-doc"

