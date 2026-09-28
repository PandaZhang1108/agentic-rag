"""答案人工评分闭环的离线测试：不调用模型、向量库或网络。"""

import asyncio
import json
from dataclasses import asdict

import pytest

from core import evals
from core.evals import CaseResult, EvalCase


def _raw_run(tmp_path, monkeypatch):
    monkeypatch.setattr(evals, "RESULTS_DIR", tmp_path)
    case = EvalCase(
        id="q001",
        question="FastAPI 的两个文档地址是什么？",
        relevant_ids=["first_steps"],
        reference="Swagger UI 在 /docs，ReDoc 在 /redoc。",
        rubric="必须同时回答 /docs 和 /redoc。",
    )
    result = CaseResult(
        id=case.id,
        difficulty=case.difficulty,
        retrieved_ids=["first_steps"],
        retrieved_contexts=["Swagger UI: /docs; ReDoc: /redoc"],
        answer="两个地址分别是 /docs 和 /redoc。",
    )
    path = evals.save_run(
        "raw-answer", evals.summarize([result]), [result], {}, cases=[case]
    )
    return path, case, result


def test_review_template_contains_frozen_answer_and_context(tmp_path, monkeypatch):
    run_path, _, result = _raw_run(tmp_path, monkeypatch)
    review_path = tmp_path / "review.json"

    evals.export_review_template(run_path, review_path)

    record = json.loads(review_path.read_text(encoding="utf-8"))[0]
    assert record["answer"] == result.answer
    assert record["retrieved_contexts"] == result.retrieved_contexts
    assert set(record["dimensions"]) == set(evals.ANSWER_DIMENSIONS)
    assert all(value is None for value in record["dimensions"].values())


def test_apply_scores_uses_dimensions_without_rerunning_agent(tmp_path, monkeypatch):
    run_path, _, _ = _raw_run(tmp_path, monkeypatch)
    review_path = tmp_path / "review.json"
    evals.export_review_template(run_path, review_path)
    records = json.loads(review_path.read_text(encoding="utf-8"))
    records[0].update({
        "dimensions": {
            "correctness": 1.0,
            "completeness": 1.0,
            "faithfulness": 1.0,
            "relevance": 0.5,
        },
        "reason": "事实完整且有资料支持，但表达略显冗余。",
        "evaluator": "human:test",
    })
    review_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    scored_path = evals.apply_scores_to_run(run_path, review_path, "scored-answer")

    payload = json.loads(scored_path.read_text(encoding="utf-8"))
    assert payload["summary"]["answer_score_mean"] == 0.875
    assert payload["summary"]["answer_score_coverage"] == 1.0
    assert payload["summary"]["answer_score_by_dimension"]["relevance"] == 0.5
    assert payload["cases"][0]["answer_evaluator"] == "human:test"


def test_apply_scores_rejects_grade_for_changed_answer(tmp_path, monkeypatch):
    run_path, case, result = _raw_run(tmp_path, monkeypatch)
    review_path = tmp_path / "review.json"
    evals.export_review_template(run_path, review_path)
    records = json.loads(review_path.read_text(encoding="utf-8"))
    records[0].update({
        "dimensions": {dimension: 1.0 for dimension in evals.ANSWER_DIMENSIONS},
        "reason": "正确。",
        "evaluator": "human:test",
    })
    review_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    payload = json.loads(run_path.read_text(encoding="utf-8"))
    payload["cases"][0] = asdict(result)
    payload["cases"][0]["answer"] = "已经变化的新答案"
    run_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="不能沿用"):
        evals.apply_scores_to_run(run_path, review_path, "should-not-exist")


def test_retrieval_metrics_exclude_cases_without_local_ground_truth():
    answerable = CaseResult(
        id="q001",
        difficulty="easy",
        retrieval_evaluable=True,
        recall=1.0,
        mrr=1.0,
    )
    outside_local_corpus = CaseResult(
        id="q002",
        difficulty="out_of_scope",
        retrieval_evaluable=False,
        recall=0.0,
        mrr=0.0,
    )

    summary = evals.summarize([answerable, outside_local_corpus])

    assert summary["recall@5"] == 1.0
    assert summary["mrr@5"] == 1.0
    assert summary["retrieval_metric_coverage"] == 0.5
    assert "out_of_scope" not in summary["recall_by_difficulty"]


def test_retrieval_metrics_apply_k_to_each_parallel_tool_call():
    batches = [
        ["security", "noise-a", "noise-b"],
        ["sse", "noise-c", "noise-d"],
    ]

    assert evals.recall_at_k_per_call(batches, ["security", "sse"], k=2) == 1.0
    assert evals.mrr_at_k_per_call(batches, ["security", "sse"], k=2) == 1.0


def test_runner_keeps_parallel_retrieval_batches_separate():
    async def agent(_question):
        return {
            "retrieved_ids": ["security", "noise", "sse"],
            "retrieval_batches": [["security", "noise"], ["sse"]],
            "answer": "combined",
        }

    case = EvalCase(
        id="multi-hop",
        question="combine",
        relevant_ids=["security", "sse"],
    )
    result = asyncio.run(evals.run_eval(agent, [case], k=1))[0]

    assert result.retrieval_batches == [["security", "noise"], ["sse"]]
    assert result.recall == 1.0
    assert result.mrr == 1.0


def test_compare_marks_higher_latency_as_regression(tmp_path, monkeypatch):
    monkeypatch.setattr(evals, "RESULTS_DIR", tmp_path)
    common = {
        "dataset": {"selected_sha256": "same-dataset"},
        "cases": [],
    }
    (tmp_path / "before.json").write_text(json.dumps({
        **common,
        "summary": {"recall@5": 0.8, "latency_p50_ms": 100.0},
    }), encoding="utf-8")
    (tmp_path / "after.json").write_text(json.dumps({
        **common,
        "summary": {"recall@5": 0.9, "latency_p50_ms": 150.0},
    }), encoding="utf-8")

    report = evals.compare("before", "after")

    assert "| recall@5 | 0.8 | 0.9 | 🟢 +0.1000 |" in report
    assert "| latency_p50_ms | 100.0 | 150.0 | 🔴 +50.0000 |" in report
