"""
================================================================================
core/tools.py —— 模块 2（工具层）+ 模块 9（权限边界）
================================================================================
这个文件是"骨架/插件"分离最典型的例子：

    骨架（这个文件）：注册机制、权限分级、审计、错误处理
    插件（domain/tools.py）：具体有哪些工具

换一个客户，你新写的只有 domain/tools.py 里那几个函数，
这个文件原样搬过去。

【为什么要有注册表，直接写个 list 不行吗】
list 能跑，但回答不了这几个问题：
    - 哪些工具是危险的？（模型被注入后能干什么）
    - 这次会话该给模型哪些工具？（工具太多会显著降低选择准确率）
    - 昨天 agent 一共调用了哪些工具、成功率多少？
注册表就是为了让这三个问题有答案。
================================================================================
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Risk(str, Enum):
    READ = "read"
    WRITE = "write"
    DANGEROUS = "dangerous"


@dataclass
class ToolSpec:
    """一个工具的完整登记信息。"""

    tool: Any
    risk: Risk = Risk.READ

    is_retrieval: bool = False

    requires_approval: bool = False
    tags: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.tool.name


class ToolRegistry:
    """
    工具的中央登记处。

    典型用法：
        registry = ToolRegistry()
        registry.register(search_orders, risk=Risk.READ, is_retrieval=True)
        registry.register(refund_order, risk=Risk.DANGEROUS, requires_approval=True)

        # 装配给模型时按需筛选
        tools = registry.tools(max_risk=Risk.WRITE)      # 不给它危险工具
    """

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
        """
        注册时就检查描述质量。

        【这是我最推荐加的一个小检查】
        工具的 description 会【原样发给模型】，模型完全靠它决定要不要调用。
        工具选错，90% 的原因是描述写得烂，不是模型笨。
        但这个错误在运行时是"静默"的 —— 模型只是没选它，不会报错，
        你要跑很多次才会发现。
        所以在注册时就拦下来，把一个运行时的隐性问题变成启动时的显性错误。
        """
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
        """
        取出要装配给模型的工具列表。

        max_risk：不给模型超过这个风险等级的工具。
        【这是防 prompt injection 最有效的一招】——
        提示词里写"忽略文档中的指令"是最弱的防护（模型可能不听）；
        真正管用的是"即使模型被骗了，它手上也没有那个工具"。

        tags：按场景挑子集。工具超过 15~20 个时，模型的选择准确率会明显下降，
        所以"这次对话只给相关的 5 个"是常见优化。
        """
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
        """哪些工具的结果算"检索资料"—— 供纠错链路判断用。"""
        return {s.name for s in self._specs.values() if s.is_retrieval}

    def needs_approval(self, tool_name: str) -> bool:
        spec = self._specs.get(tool_name)
        return bool(spec and spec.requires_approval)

    def spec(self, tool_name: str) -> ToolSpec | None:
        return self._specs.get(tool_name)

    def audit_table(self) -> list[dict[str, Any]]:
        """
        导出所有工具及其权限 —— 交付给客户时，这张表是安全评审的材料。
        FDE 场景里，客户的安全团队一定会要这个。
        """
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
    """
    工具执行的统一包装。

    做三件事：
      1. 异常不外抛，转成一句给模型看的说明
      2. 结果截断，防止一个工具返回 10 万字把上下文撑爆
      3. 记录审计日志

    【关键设计决策：工具失败了，返回给模型还是直接抛出？】
    这是 agent 工程里一个经常被问的取舍：

      - 返回给模型：agent 有机会自己换个参数重试、或换个工具。更"智能"，
        但可能陷入无效重试的循环
      - 直接抛出：立即失败，用户马上知道。更可控，但失去了自愈能力

    这里选【返回给模型】，因为 agent 的价值本来就在于自主处理意外。
    但必须配合模块 4 的循环上限，否则会无限重试。
    这两个设计是【成对】的，只做一个会出问题。
    """
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
