"""Dense Top5 与 Dense Top10 + Cross-Encoder Top5 的离线 A/B 实验。"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import get_settings
from core.evals import EvalCase, mrr_at_k, recall_at_k
from milvus_store import search_milvus
from retriever import get_embedding_model, open_milvus


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return round(ordered[index], 1)


def _source_ids(docs: list[Document]) -> list[str]:
    return [str(doc.metadata.get("source_id", "unknown")) for doc in docs]


def _ranked_rows(docs: list[Document], scores: list[float] | None = None) -> list[dict]:
    rows = []
    for index, doc in enumerate(docs):
        row = {
            "rank": index + 1,
            "doc_id": str(doc.metadata.get("doc_id", "unknown")),
            "source_id": str(doc.metadata.get("source_id", "unknown")),
            "text_preview": doc.page_content[:180].replace("\n", " "),
        }
        if scores is not None:
            row["reranker_score"] = round(float(scores[index]), 6)
        rows.append(row)
    return rows


def rerank_documents(
    question: str,
    candidates: list[Document],
    model: Any,
    final_k: int,
) -> tuple[list[Document], list[float]]:
    """让模型同时阅读“问题 + 文档”，按相关性分数重新排序。"""
    if not candidates:
        return [], []
    pairs = [[question, doc.page_content] for doc in candidates]
    raw_scores = model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)
    scores = [float(score) for score in raw_scores]
    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    selected = ranked[:final_k]
    return [doc for doc, _ in selected], [score for _, score in selected]


def summarize(records: list[dict], final_k: int, candidate_k: int) -> dict:
    dense_latency = [record["latency_ms"]["dense_top5"] for record in records]
    rerank_latency = [record["latency_ms"]["reranker"] for record in records]
    total_latency = [record["latency_ms"]["dense_top10_plus_reranker"] for record in records]
    dense_recall = statistics.mean(record["dense_top5"]["recall"] for record in records)
    dense_mrr = statistics.mean(record["dense_top5"]["mrr"] for record in records)
    candidate_recall = statistics.mean(
        record["dense_top10_candidates"]["recall"] for record in records
    )
    reranked_recall = statistics.mean(
        record["cross_encoder_top5"]["recall"] for record in records
    )
    reranked_mrr = statistics.mean(
        record["cross_encoder_top5"]["mrr"] for record in records
    )
    recall_gain = reranked_recall - dense_recall
    mrr_gain = reranked_mrr - dense_mrr
    added_p95 = _percentile(total_latency, 0.95) - _percentile(dense_latency, 0.95)
    headroom = candidate_recall - dense_recall
    worth_integrating = (
        headroom >= 0.05
        and reranked_recall >= dense_recall
        and (recall_gain >= 0.05 or mrr_gain >= 0.05)
        and added_p95 <= 500
    )
    return {
        "cases": len(records),
        f"dense_recall@{final_k}": round(dense_recall, 4),
        f"dense_mrr@{final_k}": round(dense_mrr, 4),
        f"dense_candidate_recall@{candidate_k}": round(candidate_recall, 4),
        f"reranked_recall@{final_k}": round(reranked_recall, 4),
        f"reranked_mrr@{final_k}": round(reranked_mrr, 4),
        "recall_gain": round(recall_gain, 4),
        "mrr_gain": round(mrr_gain, 4),
        "candidate_headroom": round(headroom, 4),
        "dense_latency_p50_ms": _percentile(dense_latency, 0.50),
        "dense_latency_p95_ms": _percentile(dense_latency, 0.95),
        "reranker_only_latency_p50_ms": _percentile(rerank_latency, 0.50),
        "reranker_only_latency_p95_ms": _percentile(rerank_latency, 0.95),
        "reranked_total_latency_p50_ms": _percentile(total_latency, 0.50),
        "reranked_total_latency_p95_ms": _percentile(total_latency, 0.95),
        "added_p95_ms": round(added_p95, 1),
        "decision": "integrate" if worth_integrating else "do_not_integrate",
        "decision_rule": {
            "candidate_headroom_at_least": 0.05,
            "reranked_recall_not_lower": True,
            "recall_or_mrr_gain_at_least": 0.05,
            "added_p95_ms_at_most": 500,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="domain/evalset.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--model", default="BAAI/bge-reranker-base")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.final_k < 1 or args.candidate_k < args.final_k:
        parser.error("candidate-k 必须大于等于 final-k，且两者都要大于 0")

    cases = [case for case in EvalCase.load(args.dataset) if case.relevant_ids]
    if not cases:
        parser.error("数据集中没有带 relevant_ids 的检索评测题")

    settings = get_settings().model_copy(update={"milvus_search_mode": "dense"})
    client = open_milvus()
    embedding_model = get_embedding_model()
    reranker = CrossEncoder(
        args.model,
        device=args.device,
        max_length=512,
    )

    # 先热身，避免把首次加载模型的几秒钟算进第一道题。
    warmup_question = cases[0].question
    warmup_vector = embedding_model.embed_query(warmup_question)
    warmup_docs = search_milvus(
        client,
        warmup_question,
        args.candidate_k,
        settings,
        lambda _text: warmup_vector,
    )
    rerank_documents(warmup_question, warmup_docs, reranker, args.final_k)

    records = []
    for case in cases:
        baseline_start = time.perf_counter()
        baseline_docs = search_milvus(
            client,
            case.question,
            args.final_k,
            settings,
            embedding_model.embed_query,
        )
        baseline_ms = (time.perf_counter() - baseline_start) * 1000

        candidate_start = time.perf_counter()
        candidate_docs = search_milvus(
            client,
            case.question,
            args.candidate_k,
            settings,
            embedding_model.embed_query,
        )
        candidate_ms = (time.perf_counter() - candidate_start) * 1000

        reranker_start = time.perf_counter()
        reranked_docs, reranker_scores = rerank_documents(
            case.question, candidate_docs, reranker, args.final_k
        )
        reranker_ms = (time.perf_counter() - reranker_start) * 1000

        dense_ids = _source_ids(baseline_docs)
        candidate_ids = _source_ids(candidate_docs)
        reranked_ids = _source_ids(reranked_docs)
        records.append({
            "id": case.id,
            "question": case.question,
            "relevant_ids": case.relevant_ids,
            "dense_top5": {
                "recall": recall_at_k(dense_ids, case.relevant_ids, args.final_k),
                "mrr": mrr_at_k(dense_ids, case.relevant_ids, args.final_k),
                "results": _ranked_rows(baseline_docs),
            },
            "dense_top10_candidates": {
                "recall": recall_at_k(candidate_ids, case.relevant_ids, args.candidate_k),
                "results": _ranked_rows(candidate_docs),
            },
            "cross_encoder_top5": {
                "recall": recall_at_k(reranked_ids, case.relevant_ids, args.final_k),
                "mrr": mrr_at_k(reranked_ids, case.relevant_ids, args.final_k),
                "results": _ranked_rows(reranked_docs, reranker_scores),
            },
            "latency_ms": {
                "dense_top5": round(baseline_ms, 1),
                "dense_top10": round(candidate_ms, 1),
                "reranker": round(reranker_ms, 1),
                "dense_top10_plus_reranker": round(candidate_ms + reranker_ms, 1),
            },
        })
        print(f"{case.id}: dense={records[-1]['dense_top5']['recall']:.2f} "
              f"reranked={records[-1]['cross_encoder_top5']['recall']:.2f}")

    payload = {
        "experiment": "dense_top5_vs_dense_top10_cross_encoder_top5",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "dataset": args.dataset,
            "collection": settings.milvus_collection,
            "embedding_model": settings.embedding_model_path,
            "reranker_model": args.model,
            "device": args.device,
            "candidate_k": args.candidate_k,
            "final_k": args.final_k,
            "metric_scope": "source_id/page-level; not chunk-level answer sufficiency",
        },
        "summary": summarize(records, args.final_k, args.candidate_k),
        "cases": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"已保存：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
