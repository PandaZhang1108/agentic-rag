"""
================================================================================
main.py —— 服务入口（重写版）
================================================================================
本文件修复的问题：
    P1-5  Postgres 是单连接         → 换成 AsyncConnectionPool 连接池
    P1-7  多 worker 会出问题        → 启动时检测并明确警告
    P1-8  鉴权失败不走限流          → 把鉴权挪到限流之后
    P1-9  /healthz 是假的           → 拆成 /healthz(存活) + /readyz(就绪，真查数据库)
    P1-11 客户端断开后图还在跑      → SSE 循环里检测断连
================================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agent_graph import build_workflow, get_mcp_tools
from config import get_settings
from core.llm import clear_ledger, start_ledger
from logging_config import new_request_id, request_id_var, setup_logging
from observability import flush_langfuse, traced_config
from retriever import get_embedding_model, get_reranker_model, open_milvus
from schemas import ChatRequest

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("main")


limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):

    settings.require_agent_runtime()

    try:
        open_milvus()
        if settings.milvus_search_mode != "bm25":
            await asyncio.to_thread(get_embedding_model)
            logger.info("embedding_model_ready")
        if settings.reranker_enabled:
            await asyncio.to_thread(get_reranker_model)
            logger.info("reranker_model_ready")
        logger.info("vectorstore_ready")
    except RuntimeError as exc:
        logger.error("vectorstore_missing detail=%s", exc)
        raise

    workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
    if workers > 1:
        logger.warning(
            "multiple_workers_detected workers=%s —— 每个进程仍会单独加载 embedding 模型", workers
        )

    mcp_tools = await get_mcp_tools()

    pool = AsyncConnectionPool(
        conninfo=settings.postgres_url,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        timeout=settings.db_pool_timeout,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        open=False,
    )

    try:
        await pool.open(wait=True, timeout=settings.db_pool_timeout)
        logger.info(
            "db_pool_open min=%s max=%s", settings.db_pool_min_size, settings.db_pool_max_size
        )

        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()

        app.state.graph = build_workflow(mcp_tools).compile(checkpointer=checkpointer)
        app.state.pool = pool

        logger.info("startup_complete")
        yield
    finally:
        await asyncio.to_thread(flush_langfuse)
        logger.info("shutdown_closing_pool")
        await pool.close()


app = FastAPI(title="Agentic RAG", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    rid = new_request_id()
    token = request_id_var.set(rid)
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code

        response.headers["X-Request-ID"] = rid
        return response
    except Exception:
        logger.exception("unhandled_exception")
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s status=%s duration_ms=%.1f",
            request.method,
            request.url.path,
            status_code,
            duration_ms,
        )
        request_id_var.reset(token)


def _check_api_key(x_api_key: str) -> None:
    """
    【原来的问题】
        @app.post("/chat", dependencies=[Depends(verify_api_key)])
        @limiter.limit(...)

        FastAPI 的依赖项在【进入函数体之前】执行，而 slowapi 的限流是
        包在函数体外面的装饰器。所以顺序是：
            依赖项(鉴权) → 抛 401 → 限流器【根本没被执行】
        结果：暴力猜 API key 完全不限速，想试多少次试多少次。

    【现在的做法】
        把鉴权从依赖项挪进函数体，放在限流之后。
        这样每一次失败的尝试都会消耗限流配额。

    secrets.compare_digest 而不是 == ：
        == 比较字符串时会在第一个不同的字符处提前返回，
        攻击者可以通过测量响应时间逐字节猜出密钥（时序攻击）。
        compare_digest 无论如何都比完全部字节，耗时恒定。

        另外它只接受 ASCII 字符串，传入含中文的 header 会抛 TypeError，
        所以下面要包一层 try。
    """
    try:
        ok = secrets.compare_digest(x_api_key, settings.api_key)
    except TypeError:
        ok = False
    if not ok:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


@app.post("/chat")
@limiter.limit(settings.rate_limit)
async def chat(
    request: Request,
    req: ChatRequest,
    x_api_key: str = Header(default="", alias="X-API-Key"),
):
    _check_api_key(x_api_key)

    thread_id = req.thread_id or str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 25,
    }
    config = traced_config(
        config,
        session_id=thread_id,
        request_id=request_id_var.get(),
        run_name="agentic-rag-chat",
    )
    inputs = {
        "messages": [{"role": "user", "content": req.message}],
        "rewrite_count": 0,
        "grade": "yes",
    }

    async def event_generator():

        ledger = start_ledger()
        yield _sse({"type": "thread_id", "value": thread_id})

        pending_entry_tokens: list[str] = []
        entered_downstream_node = False

        try:
            async for chunk, metadata in request.app.state.graph.astream(
                inputs, config=config, stream_mode="messages"
            ):
                if await request.is_disconnected():
                    logger.info("client_disconnected thread_id=%s", thread_id)
                    break

                node = metadata.get("langgraph_node")
                if node == "generate_query_or_respond":
                    if chunk.content:
                        pending_entry_tokens.append(str(chunk.content))
                    continue

                if node:
                    entered_downstream_node = True
                    pending_entry_tokens.clear()

                if chunk.content and node in ("generate_answer", "give_up"):
                    yield _sse({"type": "token", "value": chunk.content})

            if not entered_downstream_node:
                for token in pending_entry_tokens:
                    yield _sse({"type": "token", "value": token})

        except asyncio.CancelledError:
            logger.info("stream_cancelled thread_id=%s", thread_id)
            raise
        except Exception:
            logger.exception("graph_stream_error thread_id=%s", thread_id)
            yield _sse({"type": "error", "value": "处理这次请求时出错了，请稍后重试。"})
        finally:
            logger.info("request_usage thread_id=%s usage=%s", thread_id, ledger.summary())
            clear_ledger()

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(payload: dict) -> str:
    """
    统一构造 SSE 数据行。

    SSE 协议格式：`data: <内容>\\n\\n`，两个换行表示一条消息结束。
    ensure_ascii=False 让中文原样输出而不是 \\uXXXX 转义 ——
    体积更小，调试时肉眼可读。
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/healthz")
async def healthz():
    """
    Liveness（存活探针）：进程还活着吗？

    【关键设计原则】这里【不能】检查数据库。
    因为 liveness 失败的后果是【容器被杀掉重启】。
    数据库临时抖动一下就把所有应用容器杀光重启，
    只会让本来能自愈的故障变成雪崩。

    这就是 liveness 和 readiness 必须分开的原因 ——
    很多人把两者写成同一个接口，这是个经典错误。
    """
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request):
    """
    Readiness（就绪探针）：现在能正常服务吗？

    【原来的问题】只有一个 /healthz，无脑返回 ok。
    数据库挂了它照样绿，负载均衡照样往里打流量。

    readiness 失败的后果是【暂时不给这个实例发流量】，容器不会被杀。
    所以这里应该真的去查依赖。
    """
    pool: AsyncConnectionPool = request.app.state.pool
    try:
        await asyncio.wait_for(_check_database(pool), timeout=3)
    except Exception as exc:
        logger.warning("readiness_check_failed reason=%s", exc)
        raise HTTPException(status_code=503, detail="database unavailable") from exc

    return {"status": "ready"}


async def _check_database(pool: AsyncConnectionPool) -> None:
    """借一条连接执行最小查询；离开代码块时自动归还连接和关闭游标。"""

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")
            await cur.fetchone()
