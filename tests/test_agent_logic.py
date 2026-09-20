import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent_graph
from agent_graph import (
    _collect_tool_context,
    _get_latest_question,
    route_after_grade,
    route_on_tool_calls,
)


def test_latest_question_in_multi_turn():
    messages = [
        HumanMessage(content="第一轮问题"),
        AIMessage(content="第一轮回答"),
        HumanMessage(content="第二轮问题"),
    ]

    assert _get_latest_question(messages) == "第二轮问题"


def test_latest_question_after_rewrite():

    messages = [
        HumanMessage(content="原始问题"),
        AIMessage(content="", additional_kwargs={}),
        HumanMessage(content="重写后的问题"),
    ]
    assert _get_latest_question(messages) == "重写后的问题"


def test_collect_context_gathers_parallel_tool_calls():

    messages = [
        HumanMessage(content="问题"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "retrieve_fastapi_docs", "args": {"query": "x"}, "id": "1"},
                {"name": "web_search", "args": {"query": "x"}, "id": "2"},
            ],
        ),
        ToolMessage(content="资料A", tool_call_id="1", name="retrieve_fastapi_docs"),
        ToolMessage(content="资料B", tool_call_id="2", name="web_search"),
    ]
    context = _collect_tool_context(messages)
    assert "资料A" in context
    assert "资料B" in context


def test_collect_context_stops_at_previous_round():

    messages = [
        HumanMessage(content="第一轮"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
        ToolMessage(content="上一轮的旧资料", tool_call_id="1", name="t"),
        AIMessage(content="第一轮回答"),
        HumanMessage(content="第二轮"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "2"}]),
        ToolMessage(content="这一轮的新资料", tool_call_id="2", name="t"),
    ]
    context = _collect_tool_context(messages)
    assert "这一轮的新资料" in context
    assert "上一轮的旧资料" not in context


def test_collect_context_empty_when_no_tools():
    assert _collect_tool_context([HumanMessage(content="hi")]) == ""


def test_route_relevant_goes_to_answer():
    assert route_after_grade({"grade": "yes", "rewrite_count": 0}) == "generate_answer"


def test_route_irrelevant_first_time_rewrites():
    assert route_after_grade({"grade": "no", "rewrite_count": 0}) == "rewrite_question"


def test_route_gives_up_at_limit():

    from config import get_settings

    limit = get_settings().max_rewrites
    assert route_after_grade({"grade": "no", "rewrite_count": limit}) == "give_up"
    assert route_after_grade({"grade": "no", "rewrite_count": limit + 5}) == "give_up"


def test_route_handles_missing_keys():

    assert route_after_grade({}) == "generate_answer"


@pytest.mark.parametrize(
    "last_message,expected",
    [
        (AIMessage(content="直接回答"), "__end__"),
        (
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
            "tools",
        ),
    ],
)
def test_route_on_tool_calls(last_message, expected):

    assert route_on_tool_calls({"messages": [last_message]}) == expected


def test_route_rewrites_before_limit():
    from config import get_settings

    limit = get_settings().max_rewrites

    actual = route_after_grade(
        {
            "grade": "no",
            "rewrite_count": limit - 1,
        }
    )

    assert actual == "rewrite_question"


async def test_simple_greeting_skips_model(monkeypatch):
    async def fail_if_model_is_called(*args, **kwargs):
        raise AssertionError("问候语不应该调用模型")

    monkeypatch.setattr(
        agent_graph,
        "_ainvoke_with_timeout",
        fail_if_model_is_called,
    )

    node = agent_graph.make_generate_query_or_respond([])

    result = await node({"messages": [HumanMessage(content="你好")]})

    assert result["messages"][0].content == "你好！有什么可以帮你？"
    assert result["rewrite_count"] == 0
