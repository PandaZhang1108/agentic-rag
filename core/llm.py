from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMCall:
    """一次模型调用的记录。"""

    label: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_ms: float = 0.0
    ok: bool = True
    error: str | None = None


@dataclass
class UsageLedger:
    calls: list[LLMCall] = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int | None:
        if any(c.input_tokens is None for c in self.calls):
            return None
        return sum(c.input_tokens or 0 for c in self.calls)

    @property
    def total_output_tokens(self) -> int | None:
        if any(c.output_tokens is None for c in self.calls):
            return None
        return sum(c.output_tokens or 0 for c in self.calls)

    @property
    def total_ms(self) -> float:
        return sum(c.duration_ms for c in self.calls)

    def cost(self, price_in: float, price_out: float) -> float | None:
        """
        price_in / price_out 单位：元 / 百万 token。
        各家定价不同，所以作为参数传进来 —— 这就是"骨架 vs 插件"的分界：
        计算方式是骨架，具体单价是插件。
        """
        input_tokens = self.total_input_tokens
        output_tokens = self.total_output_tokens
        if input_tokens is None or output_tokens is None:
            return None
        return (input_tokens * price_in + output_tokens * price_out) / 1_000_000

    def summary(self) -> dict[str, Any]:
        """一次请求结束时打这一条日志，你就有了成本和延迟数据。"""
        return {
            "llm_calls": len(self.calls),
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "llm_ms": round(self.total_ms, 1),
            "failed_calls": sum(1 for c in self.calls if not c.ok),
            "token_usage_calls": sum(
                1 for c in self.calls if c.input_tokens is not None and c.output_tokens is not None
            ),
        }


_ledger_var: contextvars.ContextVar[UsageLedger | None] = contextvars.ContextVar(
    "usage_ledger", default=None
)


def start_ledger() -> UsageLedger:
    """在请求开始时调用（放在中间件里）。"""
    ledger = UsageLedger()
    _ledger_var.set(ledger)
    return ledger


def current_ledger() -> UsageLedger | None:
    return _ledger_var.get()


def clear_ledger() -> None:
    """Remove the request-local ledger after streaming/evaluation completes."""
    _ledger_var.set(None)


def extract_usage(response: Any) -> tuple[int | None, int | None]:
    """
    不同 provider 把 token 数放在不同地方，这里做兼容。

    【这就是"屏蔽 provider 差异"的具体含义】——
    脏活集中在一个函数里，上层代码不用关心你用的是哪家。
    以后换模型发现取不到 token 数，只改这一个函数。
    """

    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        return (
            int(input_tokens) if input_tokens is not None else None,
            int(output_tokens) if output_tokens is not None else None,
        )

    meta = getattr(response, "response_metadata", None) or {}
    token_usage = meta.get("token_usage") or meta.get("usage") or {}
    input_tokens = token_usage.get("prompt_tokens")
    output_tokens = token_usage.get("completion_tokens")
    return (
        int(input_tokens) if input_tokens is not None else None,
        int(output_tokens) if output_tokens is not None else None,
    )


def record_call(record: LLMCall) -> None:
    """Append one model call to the current request ledger and structured log."""
    ledger = current_ledger()
    if ledger is not None:
        ledger.calls.append(record)
    logger.info(
        "llm_call label=%s model=%s in=%s out=%s ms=%.0f ok=%s",
        record.label,
        record.model,
        record.input_tokens,
        record.output_tokens,
        record.duration_ms,
        record.ok,
    )


class ModelClient:
    """
    所有模型调用都走这里。

    提供三件事：
        - 硬超时（不信任 SDK 自己的 timeout）
        - 失败降级到备用模型
        - 自动记账
    """

    def __init__(
        self,
        primary,
        fallback=None,
        *,
        timeout: float = 30.0,
        model_name: str = "primary",
        fallback_name: str = "fallback",
    ):
        self.primary = primary
        self.fallback = fallback
        self.timeout = timeout
        self.model_name = model_name
        self.fallback_name = fallback_name

    async def ainvoke(self, messages, *, label: str, model=None):
        """
        label 是这次调用的用途标签（比如 "grade" / "generate_answer"）。

        【为什么 label 是必填的 keyword-only 参数】
        因为记账时最有用的不是"总共花了多少"，而是
        "哪个节点最烧钱"。没有 label，账本就是一堆匿名数字，
        没法回答"我该优化哪一步"。
        """
        target = model or self.primary
        name = self.model_name if target is self.primary else self.fallback_name

        start = time.perf_counter()
        record = LLMCall(label=label, model=name)

        try:
            response = await asyncio.wait_for(target.ainvoke(messages), timeout=self.timeout)
            record.input_tokens, record.output_tokens = extract_usage(response)
            return response

        except Exception as exc:
            record.ok = False
            record.error = type(exc).__name__

            if self.fallback is not None and target is self.primary:
                logger.warning("llm_failed_falling_back label=%s error=%s", label, record.error)
                record.duration_ms = (time.perf_counter() - start) * 1000
                record_call(record)
                return await self.ainvoke(messages, label=label, model=self.fallback)

            logger.error("llm_failed label=%s error=%s", label, record.error)
            raise

        finally:
            if record.ok:
                record.duration_ms = (time.perf_counter() - start) * 1000
                record_call(record)
