from __future__ import annotations

import re
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from config import get_settings
from core.evals import EvalCase
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

    docs = await asearch(question)
    return {
        "retrieved_ids": [str(doc.metadata.get("source_id", "unknown")) for doc in docs],
        "retrieved_doc_ids": [str(doc.metadata.get("doc_id", "unknown")) for doc in docs],
        "tools_called": ["retrieve_fastapi_docs"],
        "answer": "",
    }


async def agent_fn(question: str) -> dict:

    return await agent_case_fn(EvalCase(id="single-request", question=question))


async def agent_case_fn(case: EvalCase) -> dict:

    case.__post_init__()
    inputs = []
    for item in case.history:
        message_type = HumanMessage if item["role"] == "user" else AIMessage
        inputs.append(message_type(content=item["content"], id=str(uuid4())))
    inputs.append(HumanMessage(content=case.question, id=str(uuid4())))
    input_ids = {message.id for message in inputs}
    state = await _get_graph().ainvoke(
        {"messages": inputs, "rewrite_count": 0},
        config={"recursion_limit": 30},
    )

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
    }


def current_config() -> dict:

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
