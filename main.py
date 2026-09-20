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
from logging_config import new_request_id, request_id_var, setup_logging
from retriever import get_embedding_model, open_milvus
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
    inputs = {
        "messages": [{"role": "user", "content": req.message}],
        "rewrite_count": 0,
        "grade": "yes",
    }

    async def event_generator():
        yield _sse({"type": "thread_id", "value": thread_id})

        try:
            async for chunk, metadata in request.app.state.graph.astream(
                inputs, config=config, stream_mode="messages"
            ):
                if await request.is_disconnected():
                    logger.info("client_disconnected thread_id=%s", thread_id)
                    break

                node = metadata.get("langgraph_node")
                if chunk.content and node in (
                    "generate_answer",
                    "give_up",
                    "generate_query_or_respond",
                ):
                    yield _sse({"type": "token", "value": chunk.content})

        except asyncio.CancelledError:
            logger.info("stream_cancelled thread_id=%s", thread_id)
            raise
        except Exception:
            logger.exception("graph_stream_error thread_id=%s", thread_id)
            yield _sse({"type": "error", "value": "处理这次请求时出错了，请稍后重试。"})

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

    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/healthz")
async def healthz():

    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request):

    pool: AsyncConnectionPool = request.app.state.pool
    try:
        await asyncio.wait_for(_check_database(pool), timeout=3)
    except Exception as exc:
        logger.warning("readiness_check_failed reason=%s", exc)
        raise HTTPException(status_code=503, detail="database unavailable") from exc

    return {"status": "ready"}


async def _check_database(pool: AsyncConnectionPool) -> None:

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")
            await cur.fetchone()
