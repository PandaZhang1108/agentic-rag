import asyncio
import json
import os
from unittest.mock import AsyncMock

import httpx
import pytest

os.environ["POSTGRES_URL"] = "postgresql://x@localhost/x"
os.environ["DEEPSEEK_API_KEY"] = "test-key"
os.environ["TAVILY_API_KEY"] = "test-key"
os.environ["API_KEY"] = "test-api-key"
os.environ["MCP_ENABLED"] = "false"


from config import get_settings

get_settings.cache_clear()


class FakeGraph:
    async def astream(self, inputs, config=None, stream_mode=None):
        class Chunk:
            def __init__(self, content):
                self.content = content

        for piece in ["你好", "，世界"]:
            yield Chunk(piece), {"langgraph_node": "generate_answer"}


class ErrorAfterTokenGraph:
    async def astream(self, inputs, config=None, stream_mode=None):
        class Chunk:
            content = "正在查询"

        yield Chunk(), {"langgraph_node": "generate_answer"}
        raise RuntimeError("模拟图运行失败")


class ModelTimeoutGraph:
    async def astream(self, inputs, config=None, stream_mode=None):

        await asyncio.wait_for(asyncio.sleep(1), timeout=0.01)

        if False:
            yield None


class CountingGraph:
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

    from main import app

    app.state.graph = FakeGraph()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_chat_without_api_key_returns_401(client):
    resp = await client.post("/chat", json={"message": "你好"})
    assert resp.status_code == 401


async def test_chat_with_wrong_api_key_returns_401(client):
    resp = await client.post("/chat", json={"message": "你好"}, headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


async def test_chat_with_correct_api_key_succeeds(client):
    resp = await client.post(
        "/chat", json={"message": "你好"}, headers={"X-API-Key": "test-api-key"}
    )
    assert resp.status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "   "},
        {"message": "x" * 5000},
        {"message": "hi", "thread_id": "not-a-uuid"},
        {},
    ],
)
async def test_invalid_payload_returns_422(client, payload):
    resp = await client.post("/chat", json=payload, headers={"X-API-Key": "test-api-key"})
    assert resp.status_code == 422


async def test_sse_stream_shape(client):
    resp = await client.post(
        "/chat", json={"message": "你好"}, headers={"X-API-Key": "test-api-key"}
    )
    body = resp.text

    lines = [ln for ln in body.split("\n\n") if ln.strip()]
    assert lines[0].startswith("data: ")
    assert lines[-1] == "data: [DONE]"

    first = json.loads(lines[0][len("data: ") :])
    assert first["type"] == "thread_id"

    tokens = [json.loads(ln[len("data: ") :]) for ln in lines[1:-1] if ln.startswith("data: ")]
    assert "".join(t["value"] for t in tokens if t["type"] == "token") == "你好，世界"


async def test_sse_sends_error_event_when_graph_fails_mid_stream(client):

    from main import app

    app.state.graph = ErrorAfterTokenGraph()

    resp = await client.post(
        "/chat", json={"message": "你好"}, headers={"X-API-Key": "test-api-key"}
    )
    lines = [line for line in resp.text.split("\n\n") if line.strip()]
    events = [json.loads(line[len("data: ") :]) for line in lines[:-1] if line.startswith("data: ")]

    assert [event["type"] for event in events] == ["thread_id", "token", "error"]
    assert events[1]["value"] == "正在查询"
    assert lines[-1] == "data: [DONE]"


async def test_model_timeout_becomes_controlled_sse_error(client):

    from main import app

    app.state.graph = ModelTimeoutGraph()

    resp = await client.post(
        "/chat", json={"message": "你好"}, headers={"X-API-Key": "test-api-key"}
    )
    lines = [line for line in resp.text.split("\n\n") if line.strip()]
    events = [json.loads(line[len("data: ") :]) for line in lines[:-1] if line.startswith("data: ")]

    assert [event["type"] for event in events] == ["thread_id", "error"]
    assert events[1]["value"] == "处理这次请求时出错了，请稍后重试。"
    assert lines[-1] == "data: [DONE]"


async def test_disconnected_client_stops_reading_graph_stream(client, monkeypatch):

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

    tid = "12345678-1234-1234-1234-123456789abc"
    resp = await client.post(
        "/chat",
        json={"message": "你好", "thread_id": tid},
        headers={"X-API-Key": "test-api-key"},
    )
    assert tid in resp.text


async def test_healthz_needs_no_auth(client):

    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_readyz_returns_200_when_database_is_available(client, monkeypatch):

    import main

    main.app.state.pool = object()
    check_database = AsyncMock(return_value=None)
    monkeypatch.setattr(main, "_check_database", check_database)

    resp = await client.get("/readyz")

    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
    check_database.assert_awaited_once_with(main.app.state.pool)


async def test_readyz_returns_503_when_database_is_unavailable(client, monkeypatch):

    import main

    main.app.state.pool = object()
    check_database = AsyncMock(side_effect=RuntimeError("database down"))
    monkeypatch.setattr(main, "_check_database", check_database)

    resp = await client.get("/readyz")

    assert resp.status_code == 503
    assert resp.json()["detail"] == "database unavailable"
