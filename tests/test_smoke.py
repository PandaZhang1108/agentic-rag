"""Dependency API smoke tests that require no database, network, or API keys."""

import inspect


def test_langgraph_core_api_exists():
    """StateGraph / START / END / MessagesState —— 这几个是图的骨架，最不该变的。"""
    from langgraph.graph import END, START, MessagesState, StateGraph

    assert callable(StateGraph)
    assert START is not None and END is not None
    # MessagesState 必须是 TypedDict（有 __annotations__ 且含 messages 键）
    assert "messages" in MessagesState.__annotations__


def test_toolnode_import_path():
    """
    ToolNode 的位置。

    背景：LangGraph 1.0 把 langgraph.prebuilt 标记为废弃，
    功能迁到 langchain.agents，但 1.x 保持向后兼容。
    这个测试会在"兼容层被真正移除"的那天失败 —— 那正是你需要知道的时刻。
    """
    from langgraph.prebuilt import ToolNode

    assert callable(ToolNode)


def test_checkpointer_accepts_pool():
    """AsyncPostgresSaver must continue to accept a connection pool."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    sig = inspect.signature(AsyncPostgresSaver.__init__)
    assert "conn" in sig.parameters, "AsyncPostgresSaver 的构造签名变了，检查 main.py"
    assert hasattr(AsyncPostgresSaver, "setup")


def test_trim_messages_signature():
    """Guard the trim_messages parameters used by agent_graph._trim."""
    from langchain_core.messages import trim_messages

    sig = inspect.signature(trim_messages)
    for param in ("strategy", "token_counter", "max_tokens", "start_on", "include_system"):
        assert param in sig.parameters, f"trim_messages 少了参数 {param}"


def test_trim_messages_actually_trims():
    """光检查签名不够，实际跑一次 —— 参数名没变但行为变了的情况也发生过。"""
    from langchain_core.messages import AIMessage, HumanMessage, trim_messages

    msgs = []
    for i in range(10):
        msgs.append(HumanMessage(content=f"q{i}"))
        msgs.append(AIMessage(content=f"a{i}"))

    trimmed = trim_messages(
        msgs,
        strategy="last",
        token_counter=len,
        max_tokens=4,
        start_on="human",
        include_system=True,
        allow_partial=False,
    )
    assert len(trimmed) <= 4
    assert isinstance(trimmed[0], HumanMessage), "start_on='human' 应保证首条是人类消息"


def test_tool_decorator_supports_async():
    """agent_graph 里 @tool 挂在 async 函数上，这个能力不能丢。"""
    import asyncio

    from langchain_core.tools import tool

    @tool
    async def dummy(query: str) -> str:
        """Dummy tool for smoke testing."""
        return f"ok:{query}"

    assert dummy.name == "dummy"
    assert asyncio.run(dummy.ainvoke({"query": "x"})) == "ok:x"


def test_structured_output_available():
    """grade_documents 依赖 with_structured_output。"""
    from langchain_core.language_models.chat_models import BaseChatModel

    assert hasattr(BaseChatModel, "with_structured_output")


def test_our_graph_compiles_without_db():
    """
    最有价值的一个：真的把图搭起来（不接数据库、不调模型）。
    节点名字写错、边连错、State 定义不对，全都会在这里暴露。

    注意 compile() 不传 checkpointer 也能编译 —— 这正好让我们能在
    没有 Postgres 的 CI 环境里验证图结构。
    """
    from agent_graph import build_workflow

    graph = build_workflow(mcp_tools=[]).compile()
    node_names = set(graph.get_graph().nodes)

    for expected in (
        "generate_query_or_respond",
        "retrieve",
        "grade_documents",
        "rewrite_question",
        "generate_answer",
        "give_up",
    ):
        assert expected in node_names, f"节点 {expected} 不在图里"
