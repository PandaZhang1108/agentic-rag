"""确认三种 Milvus 检索的请求参数和聊天检索入口。"""

import asyncio
import sys
from types import SimpleNamespace

from langchain_core.documents import Document

import milvus_store
import retriever


def _settings(mode):
    return SimpleNamespace(
        milvus_search_mode=mode, milvus_collection="docs",
        milvus_candidate_k=10, milvus_rrf_k=60,
    )


class FakeClient:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        return self.hits

    def hybrid_search(self, **kwargs):
        self.calls.append(("hybrid", kwargs))
        return self.hits


def test_dense_and_bm25_use_different_query_inputs():
    client = FakeClient([[{"id": "a", "entity": {"doc_id": "a", "text": "one"}}]])
    embedded = []

    def embed(text):
        embedded.append(text)
        return [0.1, 0.2]

    dense_docs = milvus_store.search_milvus(client, "问题", 5, _settings("dense"), embed)
    bm25_docs = milvus_store.search_milvus(client, "FastAPI int", 5, _settings("bm25"), embed)

    assert client.calls[0][1]["data"] == [[0.1, 0.2]]
    assert client.calls[0][1]["anns_field"] == "dense"
    assert client.calls[1][1]["data"] == ["FastAPI int"]
    assert client.calls[1][1]["anns_field"] == "sparse"
    assert embedded == ["问题"]  # BM25 不应调用 Embedding 模型
    assert [doc.metadata["doc_id"] for doc in dense_docs] == ["a"]
    assert [doc.metadata["doc_id"] for doc in bm25_docs] == ["a"]


def test_hybrid_uses_two_candidate_lists_and_rrf(monkeypatch):
    class Request:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Ranker:
        def __init__(self, k):
            self.k = k

    monkeypatch.setitem(sys.modules, "pymilvus", SimpleNamespace(
        AnnSearchRequest=Request, RRFRanker=Ranker,
    ))
    hits = [[
        {"id": "b", "entity": {"doc_id": "b", "text": "two", "source_id": "s2"}},
        {"id": "b", "entity": {"doc_id": "b", "text": "two"}},
        {"id": "a", "entity": {"doc_id": "a", "text": "one", "source_id": "s1"}},
    ]]
    client = FakeClient(hits)
    docs = milvus_store.search_milvus(client, "int path", 2, _settings("hybrid"),
                                      lambda _: [0.1, 0.2])
    method, params = client.calls[0]

    assert method == "hybrid"
    assert [request.kwargs["anns_field"] for request in params["reqs"]] == ["dense", "sparse"]
    assert [request.kwargs["limit"] for request in params["reqs"]] == [10, 10]
    assert params["ranker"].k == 60
    assert params["limit"] == 2
    assert [doc.metadata["doc_id"] for doc in docs] == ["b", "a"]


def test_empty_results_and_empty_index():
    assert milvus_store.hits_to_documents([], 5) == []
    assert milvus_store.hits_to_documents([[]], 5) == []

    class EmptyIndex:
        def has_collection(self, **kwargs):
            return True

        def query(self, **kwargs):
            return []

    try:
        milvus_store.require_index(EmptyIndex(), "docs")
    except RuntimeError as exc:
        assert "为空" in str(exc)
    else:
        raise AssertionError("空集合不应被当成已建好的索引")


def test_chat_retrieval_uses_milvus(monkeypatch):
    client = object()
    calls = []
    monkeypatch.setattr(retriever, "settings", SimpleNamespace(
        retrieve_k=3,
        retrieval_timeout_seconds=5,
        reranker_enabled=False,
    ))
    monkeypatch.setattr(retriever, "open_milvus", lambda: client)
    monkeypatch.setattr(
        retriever,
        "search_milvus",
        lambda actual_client, query, k, settings, embed: (
            calls.append((actual_client, query, k)),
            [Document(page_content="milvus path", metadata={"doc_id": "x"})],
        )[1],
    )
    result = asyncio.run(retriever.asearch("hello"))
    assert calls == [(client, "hello", 3)]
    assert result[0].page_content == "milvus path"


def test_chat_retrieval_reranks_more_candidates_then_returns_final_k(monkeypatch):
    """Agent 要先向 Milvus 要 8 块，再由 Reranker 只留下分数最高的 5 块。"""
    client = object()
    calls = []
    docs = [
        Document(page_content=f"doc-{index}", metadata={"doc_id": str(index)})
        for index in range(8)
    ]

    class FakeReranker:
        def predict(self, pairs, show_progress_bar=False):
            assert len(pairs) == 8
            assert show_progress_bar is False
            return [0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

    monkeypatch.setattr(retriever, "settings", SimpleNamespace(
        retrieve_k=5,
        retrieval_timeout_seconds=5,
        reranker_enabled=True,
        reranker_candidate_k=8,
        reranker_timeout_seconds=5,
    ))
    monkeypatch.setattr(retriever, "open_milvus", lambda: client)
    monkeypatch.setattr(retriever, "get_reranker_model", lambda: FakeReranker())
    monkeypatch.setattr(
        retriever,
        "search_milvus",
        lambda actual_client, query, k, settings, embed: (
            calls.append((actual_client, query, k)),
            docs,
        )[1],
    )

    result = asyncio.run(retriever.asearch("hello"))

    assert calls == [(client, "hello", 8)]
    assert [doc.metadata["doc_id"] for doc in result] == ["1", "2", "3", "4", "5"]
    assert result[0].metadata["reranker_score"] == 0.9
