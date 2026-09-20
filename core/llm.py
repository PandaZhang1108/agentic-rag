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
    label: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0
    ok: bool = True
    error: str | None = None


@dataclass
class UsageLedger:
    calls: list[LLMCall] = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_ms(self) -> float:
        return sum(c.duration_ms for c in self.calls)

    def cost(self, price_in: float, price_out: float) -> float:

        return (
            self.total_input_tokens * price_in + self.total_output_tokens * price_out
        ) / 1_000_000

    def summary(self) -> dict[str, Any]:

        return {
            "llm_calls": len(self.calls),
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "llm_ms": round(self.total_ms, 1),
            "failed_calls": sum(1 for c in self.calls if not c.ok),
        }


_ledger_var: contextvars.ContextVar[UsageLedger | None] = contextvars.ContextVar(
    "usage_ledger", default=None
)


def start_ledger() -> UsageLedger:

    ledger = UsageLedger()
    _ledger_var.set(ledger)
    return ledger


def current_ledger() -> UsageLedger | None:
    return _ledger_var.get()


def _extract_usage(response: Any) -> tuple[int, int]:

    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))

    meta = getattr(response, "response_metadata", None) or {}
    token_usage = meta.get("token_usage") or meta.get("usage") or {}
    return (
        int(token_usage.get("prompt_tokens", 0)),
        int(token_usage.get("completion_tokens", 0)),
    )


class ModelClient:
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

        target = model or self.primary
        name = self.model_name if target is self.primary else self.fallback_name

        start = time.perf_counter()
        record = LLMCall(label=label, model=name)

        try:
            response = await asyncio.wait_for(target.ainvoke(messages), timeout=self.timeout)
            record.input_tokens, record.output_tokens = _extract_usage(response)
            return response

        except Exception as exc:
            record.ok = False
            record.error = type(exc).__name__

            if self.fallback is not None and target is self.primary:
                logger.warning("llm_failed_falling_back label=%s error=%s", label, record.error)
                record.duration_ms = (time.perf_counter() - start) * 1000
                self._record(record)
                return await self.ainvoke(messages, label=label, model=self.fallback)

            logger.error("llm_failed label=%s error=%s", label, record.error)
            raise

        finally:
            if record.ok:
                record.duration_ms = (time.perf_counter() - start) * 1000
                self._record(record)

    @staticmethod
    def _record(record: LLMCall) -> None:
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
