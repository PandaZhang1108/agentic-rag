"""
================================================================================
core/evals.py —— 模块 8（评测）
================================================================================
这是我认为对你找工作最值钱的一个文件。

【为什么】
"我做了个 RAG"在 2026 年是简历里最泛滥的一句话。
"我用 recall@5 对比了四组 chunk_size，从 0.42 提到 0.71，代价是成本涨 40%"
——这句话极少见，而且它一句话同时证明了三件事：
你会量化、你懂取舍、你真的跑过数据。

这个文件是骨架，`domain/evalset.jsonl` 是插件。
换业务只换那个 jsonl 文件。

用法：
    python -m core.evals run --dataset domain/evalset.jsonl --tag baseline
    python -m core.evals compare baseline tuned
================================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("eval_results")


@dataclass
class EvalCase:
    """
    一条评测样本。

    【每个字段为什么存在】
      id             出问题时能定位到具体哪条
      question       输入
      relevant_ids   期望命中的文档 id —— 算 recall/MRR 用
      reference      参考答案 —— 算端到端正确率用
      expected_tools 期望调用的工具 —— 算工具选择准确率用（步骤级评测）
      difficulty     【这一栏最有用】标注"这条为什么难"。
                     跑完之后按 difficulty 分组看分数，你就知道
                     系统的失败模式集中在哪类问题上 ——
                     这比一个笼统的总分有用十倍。
    """

    id: str
    question: str
    relevant_ids: list[str] = field(default_factory=list)
    reference: str | None = None
    expected_tools: list[str] = field(default_factory=list)
    difficulty: str = "normal"
    rubric: str | None = None
    split: str = "dev"

    history: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.history, list):
            raise ValueError("history 必须是消息列表")
        for message in self.history:
            if (
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
                or message["role"] not in {"user", "assistant"}
                or not isinstance(message["content"], str)
                or not message["content"].strip()
            ):
                raise ValueError("历史消息需要 user/assistant role 和非空 content")

    @staticmethod
    def load(path: str | Path) -> list[EvalCase]:
        """
        读 jsonl（一行一个 JSON 对象）。

        【为什么用 jsonl 不用 json 数组】
        1. 可以一行行流式读，几万条也不占内存
        2. git diff 友好 —— 加一条样本只多一行，review 时一眼看清
        3. 追加只要 append 一行，不用解析整个文件
        评测集是会长期演进的资产，格式要选对。
        """
        cases = []
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                try:
                    cases.append(EvalCase(**json.loads(line)))
                except Exception as exc:
                    raise ValueError(f"{path} 第 {line_no} 行解析失败：{exc}") from exc
        ids = [case.id for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("评测样本 id 不能重复")
        if any(case.split not in {"dev", "test"} for case in cases):
            raise ValueError("split 只能是 dev 或 test")
        return cases


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """前 k 条里，命中了该找到的多少比例。"""
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & set(relevant))
    return hits / len(relevant)


def mrr_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    relevant_set = set(relevant)
    for idx, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant_set:
            return 1.0 / idx
    return 0.0


def tool_selection_accuracy(called: list[str], expected: list[str]) -> float:
    """
    步骤级指标：模型选对工具了吗。

    用 Jaccard 相似度（交集/并集）而不是简单的相等判断，
    因为模型可能多调了一个无害的工具，那不该算全错。
    """
    if not expected and not called:
        return 1.0
    if not expected or not called:
        return 0.0
    a, b = set(called), set(expected)
    return len(a & b) / len(a | b)


@dataclass
class CaseResult:
    id: str
    difficulty: str
    retrieved_ids: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    recall: float = 0.0
    mrr: float = 0.0
    tool_acc: float = 0.0
    latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    answer: str = ""
    answer_status: str = "not_evaluated"
    answer_score: float | None = None
    answer_reason: str | None = None
    answer_evaluator: str | None = None
    evaluation_error: str | None = None
    error: str | None = None


@dataclass
class AnswerGrade:
    """老师的评分：0 到 1 的分数，加上理由和老师的身份。"""

    score: float
    reason: str
    evaluator: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(self.score)
            or not 0 <= self.score <= 1
        ):
            raise ValueError("答案评分必须是 0 到 1 的有限数字")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("答案评分必须给出理由")
        if not isinstance(self.evaluator, str) or not self.evaluator.strip():
            raise ValueError("答案评分必须标记评分器身份")


AnswerEvaluator = Callable[[EvalCase, str], Awaitable[AnswerGrade | None]]


def answer_fingerprint(case: EvalCase, answer: str) -> str:
    """给试题、标准和完整回答盖一个指纹章，防止误用旧评分。"""
    case_data = asdict(case)
    if not case.history:
        case_data.pop("history")
    text = json.dumps({"case": case_data, "answer": answer}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manual_evaluator(path: str | Path) -> AnswerEvaluator:
    """读取人工评分 JSON；未标注的题返回未评测，而不是默认给满分。"""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("人工评分文件必须是 JSON 数组")
    grades = {}
    for record in records:
        case_id = record["id"]
        if case_id in grades:
            raise ValueError(f"人工评分 id 重复：{case_id}")
        grade = AnswerGrade(record["score"], record["reason"], record["evaluator"])
        fingerprint = record["fingerprint"]
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("人工评分必须提供 fingerprint")
        grades[case_id] = (fingerprint, grade)

    async def evaluate(case: EvalCase, answer: str) -> AnswerGrade | None:
        entry = grades.get(case.id)
        if entry is None:
            return None
        fingerprint, grade = entry
        if fingerprint != answer_fingerprint(case, answer):
            raise ValueError("题目、评分标准或回答已改变，不能沿用这份人工评分")
        return grade

    return evaluate


def _known_tokens(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Token 用量必须是非负整数或未知(None)")
    return value


AgentFn = Callable[[str], Awaitable[dict[str, Any]]]
CaseAgentFn = Callable[[EvalCase], Awaitable[dict[str, Any]]]


async def run_eval(
    agent: AgentFn,
    cases: list[EvalCase],
    *,
    k: int = 5,
    concurrency: int = 4,
    answer_evaluator: AnswerEvaluator | None = None,
    case_agent: CaseAgentFn | None = None,
) -> list[CaseResult]:
    if k < 1 or concurrency < 1:
        raise ValueError("k 和 concurrency 必须大于 0")
    sem = asyncio.Semaphore(concurrency)

    async def run_one(case: EvalCase) -> CaseResult:
        async with sem:
            result = CaseResult(id=case.id, difficulty=case.difficulty)
            start = time.perf_counter()
            try:
                if case_agent is not None:
                    out = await case_agent(case)
                elif case.history:
                    raise ValueError("多轮题必须使用 case_agent，不能丢弃 history")
                else:
                    out = await agent(case.question)
                retrieved = out.get("retrieved_ids", [])
                result.retrieved_ids = [str(identifier) for identifier in retrieved]
                result.retrieved_doc_ids = [
                    str(identifier) for identifier in out.get("retrieved_doc_ids", [])
                ]
                result.recall = recall_at_k(retrieved, case.relevant_ids, k)
                result.mrr = mrr_at_k(retrieved, case.relevant_ids, k)
                result.tool_acc = tool_selection_accuracy(
                    out.get("tools_called", []), case.expected_tools
                )
                result.answer = str(out.get("answer", ""))
                result.input_tokens = _known_tokens(out.get("input_tokens"))
                result.output_tokens = _known_tokens(out.get("output_tokens"))
            except Exception as exc:
                result.error = f"{type(exc).__name__}: {exc}"
            finally:
                result.latency_ms = (time.perf_counter() - start) * 1000

            if result.error is None and answer_evaluator is not None:
                try:
                    grade = await answer_evaluator(case, result.answer)
                    if grade is not None:
                        if not isinstance(grade, AnswerGrade):
                            raise TypeError("评分器必须返回 AnswerGrade 或 None")
                        grade.__post_init__()
                        result.answer_status = "scored"
                        result.answer_score = grade.score
                        result.answer_reason = grade.reason
                        result.answer_evaluator = grade.evaluator
                except Exception as exc:
                    result.answer_status = "error"
                    result.evaluation_error = f"{type(exc).__name__}: {exc}"
            return result

    return await asyncio.gather(*(run_one(c) for c in cases))


def summarize(results: list[CaseResult], k: int = 5) -> dict[str, Any]:
    ok = [r for r in results if r.error is None]
    if not ok:
        return {"error": "全部失败", "total": len(results)}

    latencies = sorted(r.latency_ms for r in ok)

    def pct(p: float) -> float:
        """
        百分位。p95 = 95% 的请求比这个快。

        【为什么看 p95 不看平均值】
        平均延迟会被少数快请求拉低，掩盖尾部体验。
        用户感知到的"这系统好慢"，来自那 5% 的慢请求。
        企业采购时问的也是 p95，不是平均。
        """
        if not latencies:
            return 0.0
        idx = min(int(len(latencies) * p), len(latencies) - 1)
        return round(latencies[idx], 1)

    by_difficulty: dict[str, list[float]] = {}
    for r in ok:
        by_difficulty.setdefault(r.difficulty, []).append(r.recall)

    scored = [r for r in ok if r.answer_status == "scored" and r.answer_score is not None]
    inputs = [r.input_tokens for r in ok if r.input_tokens is not None]
    outputs = [r.output_tokens for r in ok if r.output_tokens is not None]

    return {
        "total": len(results),
        "succeeded": len(ok),
        "failed": len(results) - len(ok),
        f"recall@{k}": round(statistics.mean(r.recall for r in ok), 4),
        f"mrr@{k}": round(statistics.mean(r.mrr for r in ok), 4),
        "tool_accuracy": round(statistics.mean(r.tool_acc for r in ok), 4),
        "latency_p50_ms": pct(0.50),
        "latency_p95_ms": pct(0.95),
        "avg_input_tokens": round(statistics.mean(inputs), 1) if inputs else None,
        "avg_output_tokens": round(statistics.mean(outputs), 1) if outputs else None,
        "input_token_coverage": round(len(inputs) / len(ok), 4),
        "output_token_coverage": round(len(outputs) / len(ok), 4),
        "answer_score_mean": round(statistics.mean(r.answer_score for r in scored), 4)
        if scored
        else None,
        "answer_scored": len(scored),
        "answer_not_evaluated": sum(r.answer_status == "not_evaluated" for r in results),
        "answer_evaluation_failed": sum(r.answer_status == "error" for r in results),
        "answer_score_coverage": round(len(scored) / len(results), 4),
        "recall_by_difficulty": {
            d: round(statistics.mean(v), 4) for d, v in sorted(by_difficulty.items())
        },
    }


def save_run(
    tag: str,
    summary: dict,
    results: list[CaseResult],
    config: dict,
    *,
    cases: list[EvalCase] | None = None,
    dataset_path: str | Path | None = None,
    split: str = "all",
) -> Path:
    """
    每次跑完存一份带时间戳和配置的结果。

    【config 一定要存】
    半个月后你看到 recall 从 0.42 涨到 0.71，但想不起来当时改了什么，
    这次实验就白做了。把 chunk_size / k / 模型名一起存进去，
    结果才是可复现、可追溯的。这是实验记录的基本纪律。
    """
    if not tag or tag in {".", ".."} or Path(tag).name != tag or "\\" in tag:
        raise ValueError("实验名字只能是文件名，不能包含目录")
    RESULTS_DIR.mkdir(exist_ok=True)
    snapshots = [asdict(case) for case in cases] if cases is not None else None
    snapshot_text = json.dumps(snapshots, ensure_ascii=False, sort_keys=True)
    payload = {
        "tag": tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "dataset": {
            "file_sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
            if dataset_path is not None
            else None,
            "selected_sha256": hashlib.sha256(snapshot_text.encode("utf-8")).hexdigest()
            if snapshots is not None
            else None,
            "split": split,
            "samples": snapshots,
        },
        "summary": summary,
        "cases": [asdict(r) for r in results],
    }
    path = RESULTS_DIR / f"{tag}.json"

    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def compare(tag_a: str, tag_b: str) -> str:
    """
    对比两次运行，输出 markdown 表格 —— 直接贴进 README。

    【回归检测：这个函数最重要的部分在最后】
    总分涨了不代表没有变差的用例。"整体 recall 涨了 0.1，
    但有 3 条原来能答对的现在答错了" —— 这种信息只有逐条对比才看得到，
    而它往往比总分更重要（尤其是客户最在意的那几条）。
    """
    a = json.loads((RESULTS_DIR / f"{tag_a}.json").read_text(encoding="utf-8"))
    b = json.loads((RESULTS_DIR / f"{tag_b}.json").read_text(encoding="utf-8"))

    a_version = a.get("dataset", {}).get("selected_sha256")
    b_version = b.get("dataset", {}).get("selected_sha256")
    if a_version and b_version and a_version != b_version:
        raise ValueError("选中的评测样本或标准不同，不能直接当作同一组对照实验")

    lines = [
        f"## {tag_a} vs {tag_b}\n",
        "| 指标 | " + tag_a + " | " + tag_b + " | 变化 |",
        "|---|---|---|---|",
    ]
    for key in a["summary"]:
        va, vb = a["summary"].get(key), b["summary"].get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = vb - va
            arrow = "🟢" if delta > 0 else ("🔴" if delta < 0 else "⚪")
            lines.append(f"| {key} | {va} | {vb} | {arrow} {delta:+.4f} |")

    if not a_version or not b_version:
        lines.append("\n⚠️ 历史结果缺少数据版本，比较结论需人工核对。")

    a_cases = {c["id"]: c for c in a["cases"]}
    regressed = [
        c["id"]
        for c in b["cases"]
        if c["id"] in a_cases and c["recall"] < a_cases[c["id"]]["recall"]
    ]
    answer_regressed = [
        c["id"]
        for c in b["cases"]
        if c["id"] in a_cases
        and c.get("answer_score") is not None
        and a_cases[c["id"]].get("answer_score") is not None
        and c.get("answer_evaluator") == a_cases[c["id"]].get("answer_evaluator")
        and c["answer_score"] < a_cases[c["id"]]["answer_score"]
    ]
    if answer_regressed:
        lines.append("\n⚠️ 答案评分变差的用例：" + ", ".join(answer_regressed[:10]))
    if regressed:
        lines.append(f"\n⚠️ **变差的用例（{len(regressed)} 条）**：{', '.join(regressed[:10])}")
    else:
        lines.append("\n✅ 没有发现召回指标变差；未评分答案不在正确性检查范围内。")

    return "\n".join(lines)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Agent 评测")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--tag", required=True, help="本次实验的名字，如 baseline / chunk500")
    p_run.add_argument("--k", type=int, default=5, help="指标截断位置；不改变实际检索数量")
    p_run.add_argument("--split", choices=["all", "dev", "test"], default="all")
    p_run.add_argument("--scores", help="人工评分 JSON 文件；默认不评分")
    p_run.add_argument(
        "--mode",
        choices=["retrieval", "agent"],
        default="retrieval",
        help="retrieval 只测本地检索；agent 会运行完整 Agent 并调用模型",
    )

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("tag_a")
    p_cmp.add_argument("tag_b")

    args = parser.parse_args()

    if args.cmd == "compare":
        print(compare(args.tag_a, args.tag_b))
        return 0

    cases = EvalCase.load(args.dataset)
    if args.split != "all":
        cases = [case for case in cases if case.split == args.split]
    if args.mode == "retrieval":
        cases = [case for case in cases if case.relevant_ids]
    if not cases:
        parser.error("选中的评测样本为空")
    if args.mode == "retrieval" and any(case.history for case in cases):
        parser.error("多轮题请使用 --mode agent；纯检索模式不处理对话历史")
    from domain.eval_adapter import agent_case_fn, agent_fn, current_config, retrieval_fn

    target = retrieval_fn if args.mode == "retrieval" else agent_fn
    case_agent = agent_case_fn if args.mode == "agent" else None
    evaluator = load_manual_evaluator(args.scores) if args.scores else None
    print(f"跑 {len(cases)} 条用例...")
    results = asyncio.run(
        run_eval(target, cases, k=args.k, answer_evaluator=evaluator, case_agent=case_agent)
    )
    summary = summarize(results, k=args.k)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    path = save_run(
        args.tag,
        summary,
        results,
        current_config(),
        cases=cases,
        dataset_path=args.dataset,
        split=args.split,
    )
    print(f"\n已保存：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
