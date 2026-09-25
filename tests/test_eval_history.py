"""多轮评测入口的离线回归：不调用模型、数据库或外部工具。"""

import asyncio
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.evals import EvalCase, _main, answer_fingerprint, run_eval


def followup():
    return EvalCase(
        id="followup",
        question="那另一个页面呢？",
        history=[
            {"role": "user", "content": "FastAPI 的 Swagger UI 页面在哪？"},
            {"role": "assistant", "content": "默认在 /docs。"},
        ],
    )


@pytest.mark.parametrize(
    "history",
    [
        None,
        {},
        [{"role": "system", "content": "x"}],
        [{"role": "user", "content": ""}],
        [{"role": "user"}],
    ],
)
def test_invalid_history(history):
    with pytest.raises(ValueError):
        EvalCase(id="bad", question="x", history=history)


def test_old_dataset_and_fingerprint_remain_compatible():
    cases = EvalCase.load(Path(__file__).resolve().parents[1] / "domain/evalset.jsonl")
    assert cases and all(case.history == [] for case in cases)
    case = cases[0]
    old_data = asdict(case)
    old_data.pop("history")
    old = hashlib.sha256(
        json.dumps(
            {"case": old_data, "answer": "answer"}, ensure_ascii=False, sort_keys=True
        ).encode()
    ).hexdigest()
    assert answer_fingerprint(case, "answer") == old


def test_changed_history_invalidates_manual_grade():
    case = followup()
    original = answer_fingerprint(case, "ReDoc /redoc")
    case.history[0]["content"] = "另一个框架的文档"
    assert answer_fingerprint(case, "ReDoc /redoc") != original


def test_runner_never_silently_discards_history():
    calls = []

    async def plain(question):
        calls.append(question)
        return {"answer": "ok"}

    results = asyncio.run(run_eval(plain, [followup(), EvalCase(id="single", question="hello")]))
    assert "case_agent" in results[0].error
    assert results[1].error is None
    assert calls == ["hello"]


def test_runner_passes_complete_case():
    case = followup()

    async def plain(question):
        raise AssertionError("不应走单轮入口")

    async def whole(received):
        assert received is case
        assert "FastAPI" in received.history[0]["content"]
        return {"answer": "ReDoc /redoc"}

    result = asyncio.run(run_eval(plain, [case], case_agent=whole))[0]
    assert result.error is None and result.answer == "ReDoc /redoc"
    assert result.answer_status == "not_evaluated"


def test_retrieval_cli_rejects_multiturn_before_loading_agent(monkeypatch):
    dataset = Path(__file__).resolve().parents[1] / "domain/evalset_multiturn.jsonl"
    monkeypatch.setattr("sys.argv", ["evals", "run", "--dataset", str(dataset), "--tag", "x"])
    with pytest.raises(SystemExit) as exc:
        _main()
    assert exc.value.code == 2


def test_adapter_sends_history_and_only_counts_current_output(monkeypatch):
    from domain import eval_adapter

    seen = []

    class FakeGraph:
        async def ainvoke(self, state, config):
            seen.append(state["messages"])
            return {
                "messages": state["messages"]
                + [
                    AIMessage(
                        content="",
                        tool_calls=[{"id": "t", "name": "retrieve_fastapi_docs", "args": {}}],
                    ),
                    ToolMessage(content="[source_id:current]", tool_call_id="t"),
                    AIMessage(content="另一个是 ReDoc，默认在 /redoc。"),
                ]
            }

    monkeypatch.setattr(eval_adapter, "_graph", FakeGraph())
    case = followup()
    case.history[1]["content"] += " [source_id:old]"
    result = asyncio.run(eval_adapter.agent_case_fn(case))
    assert [type(m) for m in seen[0]] == [HumanMessage, AIMessage, HumanMessage]
    assert [m.content for m in seen[0]] == [x["content"] for x in case.history] + [case.question]
    assert result["retrieved_ids"] == ["current"]
    assert result["tools_called"] == ["retrieve_fastapi_docs"]
    assert "/redoc" in result["answer"]

    asyncio.run(eval_adapter.agent_fn("新问题"))
    assert len(seen[1]) == 1 and seen[1][0].content == "新问题"


def test_adapter_does_not_report_old_answer_when_current_turn_has_none(monkeypatch):
    from domain import eval_adapter

    class FakeGraph:
        async def ainvoke(self, state, config):
            return state

    monkeypatch.setattr(eval_adapter, "_graph", FakeGraph())
    result = asyncio.run(eval_adapter.agent_case_fn(followup()))
    assert result["answer"] == ""
    assert result["retrieved_ids"] == []
