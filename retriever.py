"""Milvus 检索入口。

建索引由 ``ingest.py`` 负责；聊天服务这里只检查索引并执行检索。
``MILVUS_SEARCH_MODE`` 可选 dense、bm25、hybrid，三种策略共用同一份 Milvus 数据。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from config import get_settings
from milvus_store import new_client, require_index, search_milvus

logger = logging.getLogger(__name__)
settings = get_settings()

_milvus_client = None
_embedding_model: HuggingFaceEmbeddings | None = None
_reranker_model: CrossEncoder | None = None
# 本地 embedding 模型没有承诺多线程安全，因此先限制为一次处理一个查询。
_embed_sem = asyncio.Semaphore(1)
# Reranker 同样复用一个模型实例，并一次只处理一个查询，避免 CPU 争抢和内存尖峰。
_rerank_sem = asyncio.Semaphore(1)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """首次需要向量时加载模型，之后复用同一个实例。"""
    global _embedding_model
    if _embedding_model is None:
        logger.info("loading_embedding_model", extra={"model": settings.embedding_model_path})
        _embedding_model = HuggingFaceEmbeddings(model_name=settings.embedding_model_path)
    return _embedding_model


def get_reranker_model() -> CrossEncoder:
    """首次重排时加载模型，之后复用同一个实例。"""
    global _reranker_model
    if _reranker_model is None:
        logger.info(
            "loading_reranker_model",
            extra={
                "model": settings.reranker_model_path,
                "device": settings.reranker_device,
            },
        )
        _reranker_model = CrossEncoder(
            settings.reranker_model_path,
            device=settings.reranker_device,
        )
    return _reranker_model


def rerank_documents(query: str, docs: list[Document], final_k: int) -> list[Document]:
    """让 Cross-Encoder 给问题与每块文档共同打分，再取分数最高的 final_k 块。"""
    if not docs:
        return []
    scores: Any = get_reranker_model().predict(
        [(query, doc.page_content) for doc in docs],
        show_progress_bar=False,
    )
    ranked_indexes = sorted(
        range(len(docs)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )
    ranked_docs = []
    for index in ranked_indexes[:final_k]:
        doc = docs[index]
        ranked_docs.append(Document(
            id=doc.id,
            page_content=doc.page_content,
            metadata={**doc.metadata, "reranker_score": float(scores[index])},
        ))
    return ranked_docs


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
    reranker_enabled = settings.reranker_enabled
    search_k = max(result_k, settings.reranker_candidate_k) if reranker_enabled else result_k
    client = open_milvus()

    async with _embed_sem:
        operation = asyncio.to_thread(
            search_milvus,
            client,
            query,
            search_k,
            settings,
            # BM25 模式不会调用这个函数；dense/hybrid 才会计算查询向量。
            lambda text: get_embedding_model().embed_query(text),
        )
        docs = await asyncio.wait_for(
            operation,
            timeout=settings.retrieval_timeout_seconds,
        )

    if not reranker_enabled:
        return docs[:result_k]

    try:
        async with _rerank_sem:
            operation = asyncio.to_thread(rerank_documents, query, docs, result_k)
            return await asyncio.wait_for(
                operation,
                timeout=settings.reranker_timeout_seconds,
            )
    except Exception:
        # 第二次排序坏了也不能让整个问答瘫痪：记录错误，再退回 Milvus 原始排序。
        logger.exception("reranker_failed_falling_back_to_milvus")
        return docs[:result_k]
