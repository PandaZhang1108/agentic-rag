"""
================================================================================
tests/test_api.py —— 接口层测试
================================================================================
你原来只测了两个纯函数（recall_at_k / mrr_at_k）。那是好的开始，
但接口层完全没测 —— 而 P0-1 那个"前端没发 X-API-Key，所有请求 401"的 bug，
只要有下面第一个测试就绝不可能上线。

这就是接口测试的价值：它守住的是【模块之间的约定】，
而单元测试只守住模块内部。

【怎么在没有数据库的情况下测】
lifespan 里要连 Postgres，CI 里没有。解法是把 graph 换成一个假的：
下面用 dependency-free 的方式直接替换 app.state.graph，
绕过 lifespan。这叫 test double（测试替身）。

跑法：
    pip install -r requirements-dev.txt
    pytest tests/test_api.py -v
================================================================================
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock

import pytest

# 在 import main 之前放入测试专用配置。
# 这里必须明确覆盖，不能用 setdefault：整套测试一起运行时，别的测试可能已经
# 把 .env 里的值放进环境变量。setdefault 看到“盒子里已有东西”就不会再写，
# 最后接口会拿真实开发 key 和 test-api-key 比较，正确请求也会得到 401。
os.environ["POSTGRES_URL"] = "postgresql://x@localhost/x"
os.environ["DEEPSEEK_API_KEY"] = "test-key"
os.environ["TAVILY_API_KEY"] = "test-key"
os.environ["API_KEY"] = "test-api-key"
os.environ["MCP_ENABLED"] = "false"

# get_settings 会记住第一次构造出来的配置。环境变量改完后要清掉旧结果，
# 保证 main 下一次取到的是上面这套测试配置，而不是之前缓存的 .env 配置。
from config import get_settings

get_settings.cache_clear()

import httpx  # noqa: E402


class FakeGraph:
    """
    假的图：不调模型，直接吐两个 token。

    语法讲解 —— 异步生成器：
        函数体里同时有 `async def` 和 `yield`，它就是异步生成器，
        调用方用 `async for x in gen()` 消费。

        这正是你之前困惑的第三种 yield：
          - LangGraph 节点          → return
          - lifespan 上下文管理器    → yield（一次，分隔启动/关闭）
          - 流式产出（这里 / SSE）   → yield（多次，边算边吐）
        三个 yield 语义完全不同，别混成一件事。
    """

    async def astream(self, inputs, config=None, stream_mode=None):
        class Chunk:
            def __init__(self, content):
                self.content = content

        for piece in ["你好", "，世界"]:
            yield Chunk(piece), {"langgraph_node": "generate_answer"}


class ToolDecisionThenAnswerGraph:
    """入口节点带解释文字并调用工具时，只应把最终答案发给用户。"""

    async def astream(self, inputs, config=None, stream_mode=None):
        class Chunk:
            def __init__(self, content):
                self.content = content

        yield Chunk("I'll look that up."), {
            "langgraph_node": "generate_query_or_respond"
        }
        yield Chunk("最终答案"), {"langgraph_node": "generate_answer"}


class DirectEntryAnswerGraph:
    """入口节点没有进入工具链时，直答内容仍必须返回。"""

    async def astream(self, inputs, config=None, stream_mode=None):
        class Chunk:
            def __init__(self, content):
                self.content = content

        for piece in ["直接", "回答"]:
            yield Chunk(piece), {"langgraph_node": "generate_query_or_respond"}


class ErrorAfterTokenGraph:
    """先产出一个正常片段，再模拟图在流式处理中故障。"""

    async def astream(self, inputs, config=None, stream_mode=None):
        class Chunk:
            content = "正在查询"

        yield Chunk(), {"langgraph_node": "generate_answer"}
        raise RuntimeError("模拟图运行失败")


class ModelTimeoutGraph:
    """用真正的 wait_for 制造超时，但不调用真实大模型。"""

    async def astream(self, inputs, config=None, stream_mode=None):
        # 假装模型需要 1 秒；计时器只肯等 0.01 秒，所以一定会超时。
        await asyncio.wait_for(asyncio.sleep(1), timeout=0.01)

        # 只为了让这个函数保持“异步生成器”类型；超时后不会执行到这里。
        if False:
            yield None


class CountingGraph:
    """记录实际产出了几个片段，用来证明断连后没有继续计算。"""

    def __init__(self):
        self.produced = 0

    async def astream(self, inputs, config=None, stream_mode=None):
        class Chunk:
            def __init__(self, content):
                self.content = content

        for number in range(10):
            self.produced += 1
            yield Chunk(str(number)), {"langgraph_node": "generate_answer"}


@pytest.fixture
async def client():
    """
    语法讲解 —— pytest fixture：
        被 @pytest.fixture 装饰的函数，可以作为参数名注入到测试函数里。
        测试函数写 `async def test_x(client):`，pytest 就会自动调用这个
        fixture 并把结果传进去。yield 之后的代码是清理逻辑（又一种 yield 用法）。

    这里用 ASGITransport 直接把请求打进 app 对象，不经过真实网络端口 ——
    快、且不占端口，CI 里可以并行跑。
    """
    from main import app

    app.state.graph = FakeGraph()      # 替换掉真图，绕过数据库依赖

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ==============================================================================
# 鉴权（这组测试能挡住 P0-1）
# ==============================================================================
async def test_chat_without_api_key_returns_401(client):
    resp = await client.post("/chat", json={"message": "你好"})
    assert resp.status_code == 401


async def test_chat_with_wrong_api_key_returns_401(client):
    resp = await client.post(
        "/chat", json={"message": "你好"}, headers={"X-API-Key": "wrong"}
    )
    assert resp.status_code == 401


async def test_chat_with_correct_api_key_succeeds(client):
    resp = await client.post(
        "/chat", json={"message": "你好"}, headers={"X-API-Key": "test-api-key"}
    )
    assert resp.status_code == 200


# ==============================================================================
# 输入校验
# ==============================================================================
@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},                        # 空字符串
        {"message": "   "},                     # 全是空白
        {"message": "x" * 5000},                # 超长
        {"message": "hi", "thread_id": "not-a-uuid"},
        {},                                     # 缺字段
    ],
)
async def test_invalid_payload_returns_422(client, payload):
    resp = await client.post(
        "/chat", json=payload, headers={"X-API-Key": "test-api-key"}
    )
    assert resp.status_code == 422


# ==============================================================================
# SSE 格式（这组测试能挡住"后端加了 error 事件但前端不认"这类问题）
# ==============================================================================
async def test_sse_stream_shape(client):
    resp = await client.post(
        "/chat", json={"message": "你好"}, headers={"X-API-Key": "test-api-key"}
    )
    body = resp.text

    lines = [ln for ln in body.split("\n\n") if ln.strip()]
    assert lines[0].startswith("data: ")
    assert lines[-1] == "data: [DONE]"

    first = json.loads(lines[0][len("data: "):])
    assert first["type"] == "thread_id"

    tokens = [
        json.loads(ln[len("data: "):])
        for ln in lines[1:-1]
        if ln.startswith("data: ")
    ]
    assert "".join(t["value"] for t in tokens if t["type"] == "token") == "你好，世界"


async def test_sse_hides_entry_tool_narration(client):
    from main import app

    app.state.graph = ToolDecisionThenAnswerGraph()
    resp = await client.post(
        "/chat", json={"message": "查资料"}, headers={"X-API-Key": "test-api-key"}
    )

    assert "最终答案" in resp.text
    assert "I'll look that up." not in resp.text


async def test_sse_keeps_direct_entry_answer(client):
    from main import app

    app.state.graph = DirectEntryAnswerGraph()
    resp = await client.post(
        "/chat", json={"message": "闲聊"}, headers={"X-API-Key": "test-api-key"}
    )

    assert "直接" in resp.text
    assert "回答" in resp.text


async def test_sse_sends_error_event_when_graph_fails_mid_stream(client):
    """HTTP 流已经开始后，只能在流内发送 error，再用 DONE 正常收尾。"""
    from main import app

    app.state.graph = ErrorAfterTokenGraph()

    resp = await client.post(
        "/chat", json={"message": "你好"}, headers={"X-API-Key": "test-api-key"}
    )
    lines = [line for line in resp.text.split("\n\n") if line.strip()]
    events = [
        json.loads(line[len("data: "):])
        for line in lines[:-1]
        if line.startswith("data: ")
    ]

    assert [event["type"] for event in events] == ["thread_id", "token", "error"]
    assert events[1]["value"] == "正在查询"
    assert lines[-1] == "data: [DONE]"


async def test_model_timeout_becomes_controlled_sse_error(client):
    """模型等待超时后，浏览器应收到可理解的错误和明确的结束标记。"""
    from main import app

    app.state.graph = ModelTimeoutGraph()

    resp = await client.post(
        "/chat", json={"message": "你好"}, headers={"X-API-Key": "test-api-key"}
    )
    lines = [line for line in resp.text.split("\n\n") if line.strip()]
    events = [
        json.loads(line[len("data: "):])
        for line in lines[:-1]
        if line.startswith("data: ")
    ]

    assert [event["type"] for event in events] == ["thread_id", "error"]
    assert events[1]["value"] == "处理这次请求时出错了，请稍后重试。"
    assert lines[-1] == "data: [DONE]"


async def test_disconnected_client_stops_reading_graph_stream(client, monkeypatch):
    """浏览器断开后，不再读取LangGraph后续片段，也不发送当前token。"""
    import main

    graph = CountingGraph()
    main.app.state.graph = graph

    async def report_disconnected(_request):
        return True

    monkeypatch.setattr(main.Request, "is_disconnected", report_disconnected)

    resp = await client.post(
        "/chat", json={"message": "你好"}, headers={"X-API-Key": "test-api-key"}
    )
    lines = [line for line in resp.text.split("\n\n") if line.strip()]

    assert graph.produced == 1
    assert all('"type": "token"' not in line for line in lines)
    assert lines[-1] == "data: [DONE]"


async def test_thread_id_is_echoed_back(client):
    """前端靠这个 thread_id 实现多轮对话，传什么必须回什么。"""
    tid = "12345678-1234-1234-1234-123456789abc"
    resp = await client.post(
        "/chat",
        json={"message": "你好", "thread_id": tid},
        headers={"X-API-Key": "test-api-key"},
    )
    assert tid in resp.text


# ==============================================================================
# 健康检查
# ==============================================================================
async def test_healthz_needs_no_auth(client):
    """liveness 探针必须免鉴权，否则负载均衡器会以为服务挂了。"""
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_readyz_returns_200_when_database_is_available(client, monkeypatch):
    """数据库检查成功时，服务才对外声明已经准备好。"""
    import main

    main.app.state.pool = object()
    check_database = AsyncMock(return_value=None)
    monkeypatch.setattr(main, "_check_database", check_database)

    resp = await client.get("/readyz")

    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
    check_database.assert_awaited_once_with(main.app.state.pool)


async def test_readyz_returns_503_when_database_is_unavailable(client, monkeypatch):
    """数据库检查失败时，告诉流量入口暂时不要把请求送进来。"""
    import main

    main.app.state.pool = object()
    check_database = AsyncMock(side_effect=RuntimeError("database down"))
    monkeypatch.setattr(main, "_check_database", check_database)

    resp = await client.get("/readyz")

    assert resp.status_code == 503
    assert resp.json()["detail"] == "database unavailable"
