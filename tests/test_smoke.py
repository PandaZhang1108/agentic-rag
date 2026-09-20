import inspect


def test_langgraph_core_api_exists():

    from langgraph.graph import END, START, MessagesState, StateGraph

    assert callable(StateGraph)
    assert START is not None and END is not None

    assert "messages" in MessagesState.__annotations__


def test_toolnode_import_path():

    from langgraph.prebuilt import ToolNode

    assert callable(ToolNode)


def test_checkpointer_accepts_pool():

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    sig = inspect.signature(AsyncPostgresSaver.__init__)
    assert "conn" in sig.parameters, "AsyncPostgresSaver 的构造签名变了，检查 main.py"
    assert hasattr(AsyncPostgresSaver, "setup")


def test_trim_messages_signature():

    from langchain_core.messages import trim_messages

    sig = inspect.signature(trim_messages)
    for param in ("strategy", "token_counter", "max_tokens", "start_on", "include_system"):
        assert param in sig.parameters, f"trim_messages 少了参数 {param}"


def test_trim_messages_actually_trims():

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

    import asyncio

    from langchain_core.tools import tool

    @tool
    async def dummy(query: str) -> str:
        """Dummy tool for smoke testing."""
        return f"ok:{query}"

    assert dummy.name == "dummy"
    assert asyncio.run(dummy.ainvoke({"query": "x"})) == "ok:x"


def test_structured_output_available():

    from langchain_core.language_models.chat_models import BaseChatModel

    assert hasattr(BaseChatModel, "with_structured_output")


def test_our_graph_compiles_without_db():

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
