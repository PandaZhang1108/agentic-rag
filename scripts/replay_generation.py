# ruff: noqa: E402
"""Replay only the final generation step with a fixed messages payload."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain.chat_models import init_chat_model

from config import get_settings
from core.llm import extract_usage
from observability import flush_langfuse, traced_config


async def run(messages_path: Path, tag: str) -> dict:
    messages = json.loads(messages_path.read_text(encoding="utf-8"))
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 文件必须是非空 JSON 数组")

    settings = get_settings()
    model = init_chat_model(
        settings.llm_model,
        temperature=0,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        api_key=settings.deepseek_api_key,
    )
    run_id = str(uuid4())
    config = traced_config(
        {},
        session_id=f"replay-{tag}",
        request_id=run_id,
        run_name="generation-replay",
        extra_metadata={
            "run_kind": "generation_replay",
            "replay_tag": tag,
            "messages_file": str(messages_path),
        },
    )

    started = time.perf_counter()
    try:
        response = await asyncio.wait_for(
            model.ainvoke(messages, config=config),
            timeout=settings.llm_timeout_seconds + 5,
        )
    finally:
        await asyncio.to_thread(flush_langfuse)
    latency_ms = (time.perf_counter() - started) * 1000
    input_tokens, output_tokens = extract_usage(response)
    estimated_cost_usd = None
    if input_tokens is not None and output_tokens is not None:
        estimated_cost_usd = (input_tokens * 0.30 + output_tokens * 1.20) / 1_000_000

    return {
        "tag": tag,
        "source_messages": str(messages_path),
        "answer": response.content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": round(latency_ms, 1),
        "estimated_cost_usd": estimated_cost_usd,
        "trace_session_id": f"replay-{tag}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    result = asyncio.run(run(args.messages, args.tag))
    output = PROJECT_ROOT / "eval_results/replays" / f"{args.tag}-result.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"已保存：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
