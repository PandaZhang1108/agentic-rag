"""Langfuse tracing integration kept optional and failure-isolated."""

from __future__ import annotations

import logging
from typing import Any

from config import get_settings

logger = logging.getLogger(__name__)


def traced_config(
    config: dict[str, Any],
    *,
    session_id: str,
    request_id: str,
    run_name: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a LangGraph config with a per-request Langfuse callback.

    Observability must never become a new reason for the user request to fail.
    Startup validation catches missing credentials; transient collector/SDK
    failures are logged and the graph continues without the callback.
    """

    settings = get_settings()
    if not settings.langfuse_enabled:
        return config

    try:
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler()
    except Exception:
        logger.exception("langfuse_callback_init_failed")
        return config

    result = dict(config)
    result["callbacks"] = [*result.get("callbacks", []), handler]
    result["run_name"] = run_name

    metadata = dict(result.get("metadata", {}))
    metadata.update(
        {
            "langfuse_session_id": session_id,
            "langfuse_tags": ["agentic-rag", settings.milvus_search_mode],
            "request_id": request_id,
            "retrieval_mode": settings.milvus_search_mode,
            "milvus_collection": settings.milvus_collection,
        }
    )
    if extra_metadata:
        metadata.update(extra_metadata)
    result["metadata"] = metadata
    return result


def flush_langfuse() -> None:
    """Flush queued events during process shutdown or short-lived eval runs."""

    if not get_settings().langfuse_enabled:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        logger.exception("langfuse_flush_failed")
