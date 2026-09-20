from __future__ import annotations

import argparse
import logging
import os
import sys

os.environ.setdefault("USER_AGENT", "agentic-rag/1.0")

from bs4 import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings
from domain.fastapi_sources import SOURCE_ID_BY_URL, SOURCE_URLS
from logging_config import setup_logging
from milvus_store import create_index, new_client
from retriever import get_embedding_model

logger = logging.getLogger("ingest")


def build_index(rebuild: bool = False) -> int:
    settings = get_settings()
    client = new_client(settings.milvus_uri, settings.milvus_token)
    exists = client.has_collection(collection_name=settings.milvus_collection)
    if exists and not rebuild:
        logger.info("milvus_index_already_exists skipping (加 --rebuild 可重建)")
        return 0

    logger.info(f"loading_documents count={len(SOURCE_URLS)}")

    docs = WebBaseLoader(
        web_paths=SOURCE_URLS,
        bs_kwargs={"parse_only": SoupStrainer("article")},
    ).load()

    for doc in docs:
        source_url = str(doc.metadata.get("source", ""))
        doc.metadata["source_id"] = SOURCE_ID_BY_URL[source_url]

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    splits = splitter.split_documents(docs)
    source_chunk_counts: dict[str, int] = {}
    for split in splits:
        source_id = str(split.metadata["source_id"])
        chunk_index = source_chunk_counts.get(source_id, 0)
        split.metadata["doc_id"] = f"{source_id}#chunk-{chunk_index:03d}"
        source_chunk_counts[source_id] = chunk_index + 1
    logger.info(
        f"split_complete chunks={len(splits)} "
        f"chunk_size={settings.chunk_size} overlap={settings.chunk_overlap}"
    )

    if not splits:
        logger.error("no_chunks_produced 检查语料来源是否可访问")
        return 1

    logger.info("embedding_and_persisting 这一步可能要几十秒到几分钟...")
    vectors = get_embedding_model().embed_documents([doc.page_content for doc in splits])
    if exists:
        client.drop_collection(collection_name=settings.milvus_collection)
    create_index(client, settings.milvus_collection, splits, vectors, settings.milvus_analyzer)
    logger.info(
        "milvus_index_built chunks=%s collection=%s", len(splits), settings.milvus_collection
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 / 重建向量索引")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="删除现有索引后重建。改了 chunk_size / embedding 模型后【必须】用这个。",
    )
    args = parser.parse_args()

    setup_logging(get_settings().log_level)
    try:
        return build_index(rebuild=args.rebuild)
    except Exception:
        logger.exception("ingest_failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
