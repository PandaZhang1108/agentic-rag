"""Reranker A/B 实验的纯逻辑测试，不连接模型或 Milvus。"""

from langchain_core.documents import Document

from scripts.reranker_ab import rerank_documents, summarize


class FakeCrossEncoder:
    def predict(self, pairs, **_kwargs):
        return [0.1 if "不相关" in document else 0.9 for _, document in pairs]


def test_reranker_moves_relevant_document_to_front():
    docs = [
        Document(page_content="不相关资料", metadata={"doc_id": "noise"}),
        Document(page_content="正确资料", metadata={"doc_id": "answer"}),
    ]

    ranked, scores = rerank_documents("问题", docs, FakeCrossEncoder(), final_k=1)

    assert ranked[0].metadata["doc_id"] == "answer"
    assert scores == [0.9]


def test_summary_integrates_only_when_quality_and_latency_pass():
    record = {
        "dense_top5": {"recall": 0.5, "mrr": 0.5},
        "dense_top10_candidates": {"recall": 1.0},
        "cross_encoder_top5": {"recall": 1.0, "mrr": 1.0},
        "latency_ms": {
            "dense_top5": 10.0,
            "reranker": 100.0,
            "dense_top10_plus_reranker": 110.0,
        },
    }

    result = summarize([record], final_k=5, candidate_k=10)

    assert result["recall_gain"] == 0.5
    assert result["mrr_gain"] == 0.5
    assert result["decision"] == "integrate"


def test_summary_rejects_reranker_that_loses_recall():
    record = {
        "dense_top5": {"recall": 1.0, "mrr": 0.5},
        "dense_top10_candidates": {"recall": 1.0},
        "cross_encoder_top5": {"recall": 0.5, "mrr": 1.0},
        "latency_ms": {
            "dense_top5": 10.0,
            "reranker": 20.0,
            "dense_top10_plus_reranker": 30.0,
        },
    }

    result = summarize([record], final_k=5, candidate_k=10)

    assert result["decision"] == "do_not_integrate"
