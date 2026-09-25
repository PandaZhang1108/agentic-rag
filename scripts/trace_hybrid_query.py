# ruff: noqa: E402
"""记录一次 Milvus Hybrid 检索的两路候选与 RRF 最终结果。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pymilvus import AnnSearchRequest, RRFRanker

from config import get_settings
from core.evals import EvalCase
from milvus_store import OUTPUT_FIELDS, new_client, require_index
from retriever import get_embedding_model


def _ranked_rows(result) -> list[dict]:
    rows = []
    for rank, hit in enumerate(result[0] if result else [], start=1):
        entity = hit.get("entity", {})
        rows.append(
            {
                "rank": rank,
                "doc_id": str(entity.get("doc_id", hit.get("id", ""))),
                "source_id": str(entity.get("source_id", "")),
                "score": hit.get("distance", hit.get("score")),
                "text_preview": str(entity.get("text", ""))[:160].replace("\n", " "),
            }
        )
    return rows


def _target_hits(rows: list[dict], relevant_ids: list[str]) -> list[dict]:
    expected = set(relevant_ids)
    return [
        {"rank": row["rank"], "doc_id": row["doc_id"]}
        for row in rows
        if row["source_id"] in expected
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace 一次 Hybrid 检索")
    parser.add_argument("case_id", help="评测题号，例如 q002")
    parser.add_argument("--dataset", default="domain/evalset.jsonl")
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--final-k", type=int, default=None)
    parser.add_argument("--rrf-k", type=float, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    case = next(
        (item for item in EvalCase.load(args.dataset) if item.id == args.case_id),
        None,
    )
    if case is None:
        raise SystemExit(f"数据集中找不到 {args.case_id}")

    settings = get_settings()
    candidate_k = args.candidate_k or settings.milvus_candidate_k
    final_k = args.final_k or settings.retrieve_k
    rrf_k = args.rrf_k or settings.milvus_rrf_k

    client = new_client(settings.milvus_uri, settings.milvus_token)
    require_index(client, settings.milvus_collection)
    vector = get_embedding_model().embed_query(case.question)
    common = {
        "collection_name": settings.milvus_collection,
        "output_fields": OUTPUT_FIELDS,
    }

    dense_request = AnnSearchRequest(
        data=[vector],
        anns_field="dense",
        param={},
        limit=candidate_k,
    )
    bm25_request = AnnSearchRequest(
        data=[case.question],
        anns_field="sparse",
        param={},
        limit=candidate_k,
    )
    dense_rows = _ranked_rows(
        client.search(
            **common,
            data=[vector],
            anns_field="dense",
            limit=candidate_k,
        )
    )
    bm25_rows = _ranked_rows(
        client.search(
            **common,
            data=[case.question],
            anns_field="sparse",
            limit=candidate_k,
        )
    )
    final_rows = _ranked_rows(
        client.hybrid_search(
            **common,
            reqs=[dense_request, bm25_request],
            ranker=RRFRanker(k=rrf_k),
            limit=final_k,
        )
    )

    stages = {
        "dense_candidates": dense_rows,
        "bm25_candidates": bm25_rows,
        "rrf_final": final_rows,
    }
    trace = {
        "trace_type": "milvus_hybrid_retrieval",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_id": case.id,
        "question": case.question,
        "relevant_ids": case.relevant_ids,
        "reference": case.reference,
        "config": {
            "collection": settings.milvus_collection,
            "candidate_k": candidate_k,
            "final_k": final_k,
            "rrf_k": rrf_k,
            "embedding_model": settings.embedding_model_path,
        },
        "stages": stages,
        "target_presence": {
            name: _target_hits(rows, case.relevant_ids) for name, rows in stages.items()
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(trace, file, ensure_ascii=False, indent=2)

    print(f"题目：{case.question}")
    print(f"人工标注的正确页面：{', '.join(case.relevant_ids)}")
    for stage, hits in trace["target_presence"].items():
        print(f"{stage}: {hits or '未命中'}")
    print(f"Trace已保存：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
