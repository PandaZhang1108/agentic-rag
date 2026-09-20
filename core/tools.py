from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class Risk(str, Enum):
    READ = "read"
    WRITE = "write"
    DANGEROUS = "dangerous"


@dataclass
class ToolSpec:
    tool: Any
    risk: Risk = Risk.READ

    is_retrieval: bool = False

    requires_approval: bool = False
    tags: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.tool.name


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(
        self,
        tool: Any,
        *,
        risk: Risk = Risk.READ,
        is_retrieval: bool = False,
        requires_approval: bool | None = None,
        tags: tuple[str, ...] = (),
    ) -> None:

        if requires_approval is None:
            requires_approval = risk is Risk.DANGEROUS

        spec = ToolSpec(
            tool=tool,
            risk=risk,
            is_retrieval=is_retrieval,
            requires_approval=requires_approval,
            tags=tags,
        )

        if spec.name in self._specs:
            raise ValueError(f"工具名重复：{spec.name}（模型靠名字区分工具，不能重名）")

        self._check_description(tool)
        self._specs[spec.name] = spec
        logger.info("tool_registered name=%s risk=%s", spec.name, risk.value)

    @staticmethod
    def _check_description(tool: Any) -> None:

        desc = (getattr(tool, "description", "") or "").strip()
        if len(desc) < 30:
            raise ValueError(
                f"工具 {getattr(tool, 'name', '?')} 的描述太短（{len(desc)} 字符）。\n"
                f"这段描述是【给模型看的说明书】，不是注释。至少要说清楚：\n"
                f"  - 这个工具能做什么\n"
                f"  - 什么情况该用它\n"
                f"  - 什么情况【不】该用它（这一条最容易漏，也最有用）"
            )

    def tools(
        self,
        *,
        max_risk: Risk = Risk.DANGEROUS,
        tags: tuple[str, ...] = (),
    ) -> list[Any]:

        order = {Risk.READ: 0, Risk.WRITE: 1, Risk.DANGEROUS: 2}
        limit = order[max_risk]

        result = []
        for spec in self._specs.values():
            if order[spec.risk] > limit:
                continue
            if tags and not set(tags) & set(spec.tags):
                continue
            result.append(spec.tool)
        return result

    def retrieval_tool_names(self) -> set[str]:

        return {s.name for s in self._specs.values() if s.is_retrieval}

    def needs_approval(self, tool_name: str) -> bool:
        spec = self._specs.get(tool_name)
        return bool(spec and spec.requires_approval)

    def spec(self, tool_name: str) -> ToolSpec | None:
        return self._specs.get(tool_name)

    def audit_table(self) -> list[dict[str, Any]]:

        return [
            {
                "name": s.name,
                "risk": s.risk.value,
                "requires_approval": s.requires_approval,
                "is_retrieval": s.is_retrieval,
                "description": (getattr(s.tool, "description", "") or "")[:120],
            }
            for s in self._specs.values()
        ]


async def safe_tool_result(
    fn: Callable,
    *args,
    tool_name: str,
    max_chars: int = 8000,
    **kwargs,
) -> str:

    import inspect

    try:
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        text = result if isinstance(result, str) else str(result)

        if len(text) > max_chars:
            logger.warning("tool_result_truncated tool=%s len=%s", tool_name, len(text))
            text = text[:max_chars] + f"\n\n[内容过长，已截断，原长度 {len(text)} 字符]"

        logger.info("tool_ok tool=%s len=%s", tool_name, len(text))
        return text

    except Exception as exc:
        logger.exception("tool_failed tool=%s", tool_name)

        return (
            f"工具 {tool_name} 执行失败：{type(exc).__name__}。"
            "请换一种方式，或告知用户此功能暂不可用。"
        )
