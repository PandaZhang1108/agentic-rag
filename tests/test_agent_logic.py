"""Pure routing and message-processing tests without external services."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent_graph
from agent_graph import (
    _collect_tool_context,
    _get_latest_question,
    route_after_grade,
    route_on_tool_calls,
)


# 多轮对话下应使用最新的用户问题。
def test_latest_question_in_multi_turn():
    messages = [
        HumanMessage(content="第一轮问题"),
        AIMessage(content="第一轮回答"),
        HumanMessage(content="第二轮问题"),
    ]
    assert _get_latest_question(messages) == "第二轮问题"


def test_latest_question_after_rewrite():
    """重写循环里追加的新 HumanMessage，应该成为"最新问题"。"""
    messages = [
        HumanMessage(content="原始问题"),
        AIMessage(content="", additional_kwargs={}),
        HumanMessage(content="重写后的问题"),
    ]
    assert _get_latest_question(messages) == "重写后的问题"


# 收集本轮全部工具结果，而不是只取最后一条。
def test_collect_context_gathers_parallel_tool_calls():
    """
    模型一次并行调两个工具时，消息序列长这样：
        AIMessage(tool_calls=[A, B])
        ToolMessage(A 的结果)
        ToolMessage(B 的结果)
    原来的 messages[-1].content 只能拿到 B，A 的结果直接丢了。
    """
    messages = [
        HumanMessage(content="问题"),
        AIMessage(content="", tool_calls=[
            {"name": "retrieve_fastapi_docs", "args": {"query": "x"}, "id": "1"},
            {"name": "web_search", "args": {"query": "x"}, "id": "2"},
        ]),
        ToolMessage(content="资料A", tool_call_id="1", name="retrieve_fastapi_docs"),
        ToolMessage(content="资料B", tool_call_id="2", name="web_search"),
    ]
    context = _collect_tool_context(messages)
    assert "资料A" in context
    assert "资料B" in context


def test_collect_context_stops_at_previous_round():
    """上一轮的工具结果不该混进这一轮的 context。"""
    messages = [
        HumanMessage(content="第一轮"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
        ToolMessage(content="上一轮的旧资料", tool_call_id="1", name="t"),
        AIMessage(content="第一轮回答"),           # ← 这条是分界线
        HumanMessage(content="第二轮"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "2"}]),
        ToolMessage(content="这一轮的新资料", tool_call_id="2", name="t"),
    ]
    context = _collect_tool_context(messages)
    assert "这一轮的新资料" in context
    assert "上一轮的旧资料" not in context


def test_collect_context_empty_when_no_tools():
    assert _collect_tool_context([HumanMessage(content="hi")]) == ""


def test_generate_prompt_requires_cross_passage_synthesis():
    prompt = agent_graph.GENERATE_SYSTEM_PROMPT
    assert "combining facts from multiple retrieved passages" in prompt
    assert "MUST combine them into a concrete answer" in prompt
    assert "ready-made end-to-end example is not missing information" in prompt
    assert "do not describe an application-wide solution" in prompt
    assert "provide one integrated implementation" in prompt
    assert "described but not wired in" in prompt
    assert "after considering all passages together" in prompt
    assert "publication or observation date" in prompt
    assert "do not present that claim as the current status" in prompt
    assert "Never write `value: T = None`" in prompt
    assert "`value: T | None = None`" in prompt
    assert "Required parameters must appear before parameters with defaults" in prompt
    assert "`item_id: int, item: Item, q: str | None = None`" in prompt


def test_router_prompt_separates_tool_calls_from_visible_answer():
    prompt = agent_graph.ROUTER_SYSTEM_PROMPT
    assert "Today is {current_date}" in prompt
    assert "MUST call exactly one grounding tool" in prompt
    assert "call retrieve_fastapi_docs" in prompt
    assert "call web_search" in prompt
    assert "including CSS" in prompt
    assert "Leave assistant content empty" in prompt


# 重写次数必须有上限。
def test_route_relevant_goes_to_answer():
    assert route_after_grade({"grade": "yes", "rewrite_count": 0}) == "generate_answer"


def test_route_irrelevant_first_time_rewrites():
    assert route_after_grade({"grade": "no", "rewrite_count": 0}) == "rewrite_question"


def test_route_gives_up_at_limit():
    """
    到达上限后必须走兜底节点，而不是继续循环。
    原来没有这个判断，会一路循环到 recursion_limit 抛异常，
    用户等 40 秒最后看到"处理出错了"。
    """
    from config import get_settings

    limit = get_settings().max_rewrites
    assert route_after_grade({"grade": "no", "rewrite_count": limit}) == "give_up"
    assert route_after_grade({"grade": "no", "rewrite_count": limit + 5}) == "give_up"


def test_route_handles_missing_keys():
    """
    TypedDict 没有默认值，state 里可能压根没有这两个键。
    代码里必须用 .get(key, 默认值)，直接 state["rewrite_count"] 会 KeyError。
    这个测试就是钉住这一点。
    """
    assert route_after_grade({}) == "generate_answer"


# ==============================================================================
# 工具调用路由
# ==============================================================================
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

    actual = route_after_grade({
        "grade": "no",
        "rewrite_count": limit - 1,
    })

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

    result = await node({
        "messages": [
            HumanMessage(content="你好")
        ]
    })

    assert result["messages"][0].content == "你好！有什么可以帮你？"
    assert result["rewrite_count"] == 0
