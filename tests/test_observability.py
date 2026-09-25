"""Langfuse config and local Token ledger must remain deterministic and fail-open."""

import sys
from types import SimpleNamespace

import observability
from core.llm import LLMCall, UsageLedger, clear_ledger, current_ledger, start_ledger


def test_usage_ledger_distinguishes_unknown_usage_from_zero():
    ledger = UsageLedger(calls=[LLMCall(label="answer", model="m")])
    assert ledger.total_input_tokens is None
    assert ledger.total_output_tokens is None
    assert ledger.cost(1.0, 2.0) is None

    measured = UsageLedger(
        calls=[LLMCall(label="answer", model="m", input_tokens=100, output_tokens=20)]
    )
    assert measured.total_input_tokens == 100
    assert measured.total_output_tokens == 20
    assert measured.cost(1.0, 2.0) == 0.00014


def test_request_ledgers_do_not_leak_between_runs():
    first = start_ledger()
    first.calls.append(LLMCall(label="one", model="m", input_tokens=1, output_tokens=2))
    clear_ledger()
    assert current_ledger() is None

    second = start_ledger()
    assert second.calls == []
    clear_ledger()


def test_traced_config_adds_session_request_and_callback(monkeypatch):
    class FakeHandler:
        pass

    settings = SimpleNamespace(
        langfuse_enabled=True,
        milvus_search_mode="hybrid",
        milvus_collection="docs",
    )
    monkeypatch.setattr(observability, "get_settings", lambda: settings)
    monkeypatch.setitem(
        sys.modules,
        "langfuse.langchain",
        SimpleNamespace(CallbackHandler=FakeHandler),
    )

    result = observability.traced_config(
        {"recursion_limit": 10},
        session_id="thread-1",
        request_id="request-1",
        run_name="chat",
    )

    assert isinstance(result["callbacks"][0], FakeHandler)
    assert result["run_name"] == "chat"
    assert result["metadata"]["langfuse_session_id"] == "thread-1"
    assert result["metadata"]["request_id"] == "request-1"
    assert result["metadata"]["retrieval_mode"] == "hybrid"
