"""并排查看同一道题在三种 Milvus 检索模式下的最终结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_RUNS = (
    Path("eval_results/milvus-dense-ranked-20260915.json"),
    Path("eval_results/milvus-bm25-ranked-20260915.json"),
    Path("eval_results/milvus-hybrid-ranked-20260915.json"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="查看指定用例的三种 Milvus 检索结果")
    parser.add_argument("case_id", help="例如 q002")
    parser.add_argument(
        "runs",
        nargs="*",
        type=Path,
        help="要比较的结果 JSON；不传时使用已有的 dense、BM25、hybrid 三组结果",
    )
    args = parser.parse_args()

    paths = tuple(args.runs) or DEFAULT_RUNS
    for index, path in enumerate(paths):
        data = json.loads(path.read_text(encoding="utf-8"))
        if index == 0:
            sample = next(
                (item for item in data["dataset"]["samples"] if item["id"] == args.case_id),
                None,
            )
            if sample is None:
                raise SystemExit(f"{path} 的数据集快照中找不到 {args.case_id}")
            print(f"题目：{sample['question']}")
            print(f"人工标注的正确页面：{', '.join(sample['relevant_ids']) or '无'}")
            print(f"参考答案：{sample.get('reference') or '未提供'}")

        result = next(
            (item for item in data["cases"] if item["id"] == args.case_id),
            None,
        )
        if result is None:
            raise SystemExit(f"{path} 中找不到 {args.case_id}")

        config = data["config"]
        recall_key = next(key for key in data["summary"] if key.startswith("recall@"))
        cutoff = recall_key.split("@", 1)[1]
        print(
            f"\n[{config['search_mode']}] "
            f"final_k={config['retrieve_k']} "
            f"candidate_k={config.get('candidate_k')} "
            f"rrf_k={config.get('rrf_k')}"
        )
        for rank, doc_id in enumerate(result["retrieved_doc_ids"], start=1):
            print(f"  {rank}. {doc_id}")
        print(f"  Recall@{cutoff}={result['recall']}  MRR@{cutoff}={result['mrr']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
