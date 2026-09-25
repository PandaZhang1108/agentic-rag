"""Milvus 检索入口。

建索引由 ``ingest.py`` 负责；聊天服务这里只检查索引并执行检索。
``MILVUS_SEARCH_MODE`` 可选 dense、bm25、hybrid，三种策略共用同一份 Milvus 数据。
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import get_settings
from milvus_store import new_client, require_index, search_milvus

logger = logging.getLogger(__name__)
settings = get_settings()

_milvus_client = None
_embedding_model: HuggingFaceEmbeddings | None = None

_embed_sem = asyncio.Semaphore(1)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """首次需要向量时加载模型，之后复用同一个实例。"""
    global _embedding_model
    if _embedding_model is None:
        logger.info("loading_embedding_model", extra={"model": settings.embedding_model_path})
        _embedding_model = HuggingFaceEmbeddings(model_name=settings.embedding_model_path)
    return _embedding_model


def open_milvus():
    """连接 Milvus 并确认集合中已有数据；同一进程只检查一次。"""
    global _milvus_client
    if _milvus_client is None:
        try:
            client = new_client(settings.milvus_uri, settings.milvus_token)
            require_index(client, settings.milvus_collection)
        except Exception as exc:
            raise RuntimeError(f"Milvus 索引未就绪：{exc}") from exc
        _milvus_client = client
    return _milvus_client


async def asearch(query: str, k: int | None = None) -> list[Document]:
    """在线程池中执行 Milvus 检索，避免阻塞 FastAPI 的事件循环。"""
    result_k = k or settings.retrieve_k
    client = open_milvus()

    async with _embed_sem:
        operation = asyncio.to_thread(
            search_milvus,
            client,
            query,
            result_k,
            settings,
            lambda text: get_embedding_model().embed_query(text),
        )
        return await asyncio.wait_for(
            operation,
            timeout=settings.retrieval_timeout_seconds,
        )
