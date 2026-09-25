"""Milvus 建索引与检索；BM25 和 RRF 由 Milvus 执行。"""

from __future__ import annotations

from langchain_core.documents import Document

OUTPUT_FIELDS = ["doc_id", "text", "source", "source_id"]


def new_client(uri: str, token: str = ""):
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise RuntimeError(f"PyMilvus 或其依赖无法导入：{exc}") from exc
    kwargs = {"uri": uri}
    if token.strip():
        kwargs["token"] = token
    return MilvusClient(**kwargs)


def require_index(client, collection: str) -> None:
    if not client.has_collection(collection_name=collection):
        raise RuntimeError(f"Milvus 集合不存在：{collection}。请先运行 python ingest.py")
    first = client.query(
        collection_name=collection,
        filter="",
        output_fields=["doc_id"],
        limit=1,
    )
    if not first:
        raise RuntimeError(f"Milvus 集合为空：{collection}。请重新运行建索引步骤")


def hits_to_documents(result: list[list[dict]], k: int) -> list[Document]:
    """Milvus 的结果包一层列表；按排名取出、去重，再交给现有 Agent。"""
    docs: list[Document] = []
    seen: set[str] = set()
    for hit in result[0] if result else []:
        entity = hit.get("entity", {})
        doc_id = str(entity.get("doc_id", hit.get("id", "")))
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        docs.append(
            Document(
                page_content=str(entity.get("text", "")),
                metadata={
                    "doc_id": doc_id,
                    "source": str(entity.get("source", "")),
                    "source_id": str(entity.get("source_id", "")),
                },
            )
        )
        if len(docs) == k:
            break
    return docs


def search_milvus(client, query: str, k: int, settings, embed_query) -> list[Document]:
    mode = settings.milvus_search_mode
    common = {
        "collection_name": settings.milvus_collection,
        "output_fields": OUTPUT_FIELDS,
    }
    if mode == "dense":
        result = client.search(
            **common,
            data=[embed_query(query)],
            anns_field="dense",
            limit=k,
        )
    elif mode == "bm25":
        result = client.search(
            **common,
            data=[query],
            anns_field="sparse",
            limit=k,
        )
    elif mode == "hybrid":
        from pymilvus import AnnSearchRequest, RRFRanker

        candidate_k = max(k, settings.milvus_candidate_k)
        dense = AnnSearchRequest(
            data=[embed_query(query)],
            anns_field="dense",
            param={},
            limit=candidate_k,
        )
        bm25 = AnnSearchRequest(
            data=[query],
            anns_field="sparse",
            param={},
            limit=candidate_k,
        )
        result = client.hybrid_search(
            **common,
            reqs=[dense, bm25],
            ranker=RRFRanker(k=settings.milvus_rrf_k),
            limit=k,
        )
    else:
        raise ValueError(f"未知 Milvus 检索方式：{mode}")
    return hits_to_documents(result, k)


def create_index(
    client, collection: str, docs: list[Document], vectors: list[list[float]], analyzer: str
) -> None:
    """每块的 ID、原文和原向量一起写入；稀疏向量由 Milvus 的 BM25 生成。"""
    if not docs or len(docs) != len(vectors) or not vectors[0]:
        raise ValueError("文档与向量不能为空，且必须一一对应")
    from pymilvus import DataType, Function, FunctionType

    dim = len(vectors[0])
    if any(len(vector) != dim for vector in vectors):
        raise ValueError("所有向量的维度必须相同")
    schema = client.create_schema(auto_id=False)
    schema.add_field("doc_id", DataType.VARCHAR, is_primary=True, max_length=512)
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=True,
        analyzer_params={"type": analyzer},
    )
    schema.add_field("source", DataType.VARCHAR, max_length=2048)
    schema.add_field("source_id", DataType.VARCHAR, max_length=512)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(
        Function(
            name="text_bm25",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
    )
    indexes = client.prepare_index_params()

    indexes.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="L2")
    indexes.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"bm25_k1": 1.2, "bm25_b": 0.75},
    )
    client.create_collection(collection_name=collection, schema=schema, index_params=indexes)
    try:
        for start in range(0, len(docs), 100):
            rows = []
            for doc, vector in zip(
                docs[start : start + 100], vectors[start : start + 100], strict=True
            ):
                rows.append(
                    {
                        "doc_id": str(doc.metadata["doc_id"]),
                        "text": doc.page_content,
                        "source": str(doc.metadata.get("source", "")),
                        "source_id": str(doc.metadata.get("source_id", "")),
                        "dense": list(vector),
                    }
                )
            client.insert(collection_name=collection, data=rows)
        client.flush(collection_name=collection)
    except Exception:
        client.drop_collection(collection_name=collection)
        raise
