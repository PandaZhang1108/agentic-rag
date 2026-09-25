"""把当前 Agent 包装成 core.evals 能调用的形状。"""

from __future__ import annotations

import asyncio
import re
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from config import get_settings
from core.evals import EvalCase
from core.llm import clear_ledger, start_ledger
from observability import flush_langfuse, traced_config
from retriever import asearch

_SOURCE_ID_PATTERN = re.compile(r"\[source_id:([^\]]+)\]")
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        from agent_graph import build_workflow

        _graph = build_workflow().compile()
    return _graph


async def retrieval_fn(question: str) -> dict:
    """只测当前配置的检索，不调用回答模型或联网工具。"""
    docs = await asearch(question)
    return {
        "retrieved_ids": [str(doc.metadata.get("source_id", "unknown")) for doc in docs],
        "retrieved_doc_ids": [str(doc.metadata.get("doc_id", "unknown")) for doc in docs],
        "tools_called": ["retrieve_fastapi_docs"],
        "answer": "",
    }


async def agent_fn(question: str) -> dict:
    """运行一次完整 Agent，并提取评测器需要的数据。"""
    return await agent_case_fn(EvalCase(id="single-request", question=question))


async def agent_case_fn(case: EvalCase) -> dict:
    """将题目的前文与当前问题一起传入，每条题独立运行。"""
    case.__post_init__()
    inputs = []
    for item in case.history:
        message_type = HumanMessage if item["role"] == "user" else AIMessage
        inputs.append(message_type(content=item["content"], id=str(uuid4())))
    inputs.append(HumanMessage(content=case.question, id=str(uuid4())))
    input_ids = {message.id for message in inputs}
    run_id = str(uuid4())
    ledger = start_ledger()
    try:
        config = traced_config(
            {"recursion_limit": 30},
            session_id=f"eval-{case.id}",
            request_id=run_id,
            run_name="agentic-rag-eval",
            extra_metadata={"eval_case_id": case.id, "run_kind": "evaluation"},
        )
        state = await _get_graph().ainvoke(
            {"messages": inputs, "rewrite_count": 0},
            config=config,
        )
    finally:
        await asyncio.to_thread(flush_langfuse)
        usage = ledger.summary()
        clear_ledger()

    messages = [message for message in state["messages"] if message.id not in input_ids]

    tools_called: list[str] = []
    retrieved_ids: list[str] = []
    answer = ""

    for message in messages:
        if isinstance(message, AIMessage):
            for call in getattr(message, "tool_calls", None) or []:
                name = call.get("name")
                if name and name not in tools_called:
                    tools_called.append(name)

            if not getattr(message, "tool_calls", None) and message.content:
                answer = str(message.content)

        content = getattr(message, "content", "")
        if isinstance(content, str):
            retrieved_ids.extend(_SOURCE_ID_PATTERN.findall(content))

    return {
        "retrieved_ids": retrieved_ids,
        "tools_called": tools_called,
        "answer": answer,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
    }


def current_config() -> dict:
    """保存本次实验的关键参数，保证结果可以追溯。"""
    settings = get_settings()
    return {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "retrieve_k": settings.retrieve_k,
        "embedding_model": settings.embedding_model_path,
        "vector_backend": "milvus",
        "collection": settings.milvus_collection,
        "search_mode": settings.milvus_search_mode,
        "candidate_k": settings.milvus_candidate_k,
        "rrf_k": settings.milvus_rrf_k,
        "analyzer": settings.milvus_analyzer,
    }
