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
ANSWER_DIMENSIONS = ("correctness", "completeness", "faithfulness", "relevance")


# ==============================================================================
# 一、数据格式
# ==============================================================================
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
    difficulty: str = "normal"      # 例如 multi_hop / ambiguous / out_of_scope
    rubric: str | None = None       # 必答事实、允许的表达、错误与部分得分规则
    split: str = "dev"              # 调参用 dev；最终验证用 test

    history: list[dict[str, str]] = field(default_factory=list)  # 当前问题之前的模拟对话

    def __post_init__(self) -> None:
        if not isinstance(self.history, list):
            raise ValueError("history 必须是消息列表")
        for message in self.history:
            if (not isinstance(message, dict)
                    or set(message) != {"role", "content"}
                    or message["role"] not in {"user", "assistant"}
                    or not isinstance(message["content"], str)
                    or not message["content"].strip()):
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


# ==============================================================================
# 二、指标
# ==============================================================================
def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """前 k 条里，命中了该找到的多少比例。"""
    if not relevant:
        return 0.0          # 边界：标注"没有正确答案"时不该除零
    hits = len(set(retrieved[:k]) & set(relevant))
    return hits / len(relevant)


def mrr_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    第一个命中项排在第几位的倒数。排第 1 得 1.0，排第 3 得 1/3。

    【recall 和 MRR 的区别，面试会问】
      recall 只关心"找没找到"，不关心排在第几
      MRR 只关心"最靠前的那个命中排第几"，不关心一共找到几个
    所以两个要一起看：recall 高但 MRR 低 = 相关内容找到了但排在后面，
    应先检查候选内容和排名，再实验比较重排、切片等方案，不能仅凭两个总分归因。
    """
    relevant_set = set(relevant)
    for idx, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant_set:
            return 1.0 / idx
    return 0.0


def recall_at_k_per_call(
    retrieval_batches: list[list[str]], relevant: list[str], k: int
) -> float:
    """每次检索调用各看前 k 条，再计算本轮 Agent 实际获得的整体召回。"""
    visible = [doc_id for batch in retrieval_batches for doc_id in batch[:k]]
    return recall_at_k(visible, relevant, len(visible))


def mrr_at_k_per_call(
    retrieval_batches: list[list[str]], relevant: list[str], k: int
) -> float:
    """并行检索调用没有先后排名；取各批次中最佳的首个相关结果排名。"""
    return max(
        (mrr_at_k(batch, relevant, k) for batch in retrieval_batches),
        default=0.0,
    )


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


# ==============================================================================
# 三、运行器
# ==============================================================================
@dataclass
class CaseResult:
    id: str
    difficulty: str
    retrieved_ids: list[str] = field(default_factory=list)
    retrieval_batches: list[list[str]] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieved_contexts: list[str] = field(default_factory=list)
    retrieval_evaluable: bool = True
    recall: float = 0.0
    mrr: float = 0.0
    tool_acc: float = 0.0
    latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    answer: str = ""
    answer_status: str = "not_evaluated"
    answer_score: float | None = None
    answer_dimensions: dict[str, float] | None = None
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
    dimensions: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if (isinstance(self.score, bool) or not isinstance(self.score, (int, float))
                or not math.isfinite(self.score) or not 0 <= self.score <= 1):
            raise ValueError("答案评分必须是 0 到 1 的有限数字")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("答案评分必须给出理由")
        if not isinstance(self.evaluator, str) or not self.evaluator.strip():
            raise ValueError("答案评分必须标记评分器身份")
        if self.dimensions is not None:
            if set(self.dimensions) != set(ANSWER_DIMENSIONS):
                raise ValueError(
                    "答案维度必须完整包含 "
                    "correctness/completeness/faithfulness/relevance"
                )
            for name, value in self.dimensions.items():
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError(f"答案维度 {name} 必须是 0 到 1 的有限数字")


AnswerEvaluator = Callable[[EvalCase, str], Awaitable[AnswerGrade | None]]


def answer_fingerprint(case: EvalCase, answer: str) -> str:
    """给试题、标准和完整回答盖一个指纹章，防止误用旧评分。"""
    case_data = asdict(case)
    if not case.history:
        # 保留旧单轮题的指纹；多轮题的前文则必须参与指纹计算。
        case_data.pop("history")
    text = json.dumps({"case": case_data, "answer": answer},
                      ensure_ascii=False, sort_keys=True)
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
        raw_dimensions = record.get("dimensions")
        dimensions = None
        if raw_dimensions is not None:
            if not isinstance(raw_dimensions, dict):
                raise ValueError("dimensions 必须是对象")
            if set(raw_dimensions) != set(ANSWER_DIMENSIONS):
                raise ValueError(f"{case_id} 的四个评分维度不完整")
            for name, value in raw_dimensions.items():
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError(f"{case_id} 的 {name} 尚未填写 0 到 1 的分数")
            dimensions = dict(raw_dimensions)
            # 四项分数填写完后，程序替人算平均数，避免手算出错。
            calculated_score = statistics.mean(dimensions.values())
            raw_score = record.get("score")
            if raw_score is None:
                raw_score = calculated_score
            elif (not isinstance(raw_score, (int, float))
                  or not math.isclose(raw_score, calculated_score, abs_tol=1e-9)):
                raise ValueError(f"{case_id} 的 score 与四个维度平均分不一致")
        else:
            raw_score = record.get("score")
            if raw_score is None:
                raise ValueError(f"{case_id} 必须填写 score 或 dimensions")
        grade = AnswerGrade(raw_score, record["reason"], record["evaluator"], dimensions)
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


# 类型别名：被测系统必须长这个样子。
# 传进来一个问题，返回一个 dict，至少含 retrieved_ids / tools_called / answer。
# 【这个抽象是复用的关键】—— 只要你的 agent 能包装成这个签名，
# 这套评测代码就能直接用，不管它内部是 LangGraph 还是别的。
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
    """
    批量跑评测。

    语法讲解 —— asyncio.Semaphore 控制并发：
        不加限制地 gather 100 个请求，会瞬间打爆 API 的速率限制，
        而且延迟数据会失真（互相排队）。
        Semaphore(4) = 同时最多 4 个在跑。

    语法讲解 —— asyncio.gather(*tasks)：
        并发执行所有协程，按【传入顺序】返回结果（不是完成顺序）。
        所以下面的 results 和 cases 是一一对应的。
    """
    if k < 1 or concurrency < 1:
        raise ValueError("k 和 concurrency 必须大于 0")
    sem = asyncio.Semaphore(concurrency)

    async def run_one(case: EvalCase) -> CaseResult:
        async with sem:
            result = CaseResult(id=case.id, difficulty=case.difficulty)
            # 没有 relevant_ids 的题不具备本地检索标准答案，不能当作检索失败。
            result.retrieval_evaluable = bool(case.relevant_ids)
            start = time.perf_counter()
            try:
                if case_agent is not None:
                    out = await case_agent(case)
                elif case.history:
                    raise ValueError("多轮题必须使用 case_agent，不能丢弃 history")
                else:
                    out = await agent(case.question)
                retrieved = out.get("retrieved_ids", [])
                raw_batches = out.get("retrieval_batches")
                if raw_batches is None:
                    # 兼容尚未返回分批结果的旧 Agent 适配器。
                    result.retrieval_batches = [
                        [str(identifier) for identifier in retrieved]
                    ] if retrieved else []
                else:
                    result.retrieval_batches = [
                        [str(identifier) for identifier in batch]
                        for batch in raw_batches
                    ]
                result.retrieved_ids = [
                    identifier
                    for batch in result.retrieval_batches
                    for identifier in batch
                ]
                result.retrieved_doc_ids = [
                    str(identifier) for identifier in out.get("retrieved_doc_ids", [])
                ]
                result.retrieved_contexts = [
                    str(context) for context in out.get("retrieved_contexts", [])
                ]
                result.recall = recall_at_k_per_call(
                    result.retrieval_batches, case.relevant_ids, k
                )
                result.mrr = mrr_at_k_per_call(
                    result.retrieval_batches, case.relevant_ids, k
                )
                result.tool_acc = tool_selection_accuracy(
                    out.get("tools_called", []), case.expected_tools
                )
                result.answer = str(out.get("answer", ""))
                result.input_tokens = _known_tokens(out.get("input_tokens"))
                result.output_tokens = _known_tokens(out.get("output_tokens"))
            except Exception as exc:
                # 【重要】单条失败不能中断整批。
                # 跑 30 条评测跑到第 7 条崩了，前面 6 条的结果也白费——
                # 这种事发生一次你就明白为什么要 try 在这一层。
                result.error = f"{type(exc).__name__}: {exc}"
            finally:
                result.latency_ms = (time.perf_counter() - start) * 1000
            # 评分耗时不混进被测系统耗时；评分失败也不冒充 Agent 执行失败。
            if result.error is None and answer_evaluator is not None:
                try:
                    grade = await answer_evaluator(case, result.answer)
                    if grade is not None:
                        if not isinstance(grade, AnswerGrade):
                            raise TypeError("评分器必须返回 AnswerGrade 或 None")
                        grade.__post_init__()
                        result.answer_status = "scored"
                        result.answer_score = grade.score
                        result.answer_dimensions = grade.dimensions
                        result.answer_reason = grade.reason
                        result.answer_evaluator = grade.evaluator
                except Exception as exc:
                    result.answer_status = "error"
                    result.evaluation_error = f"{type(exc).__name__}: {exc}"
            return result

    return await asyncio.gather(*(run_one(c) for c in cases))


# ==============================================================================
# 四、汇总与落盘
# ==============================================================================
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

    retrieval_scored = [r for r in ok if r.retrieval_evaluable]
    by_difficulty: dict[str, list[float]] = {}
    for r in retrieval_scored:
        by_difficulty.setdefault(r.difficulty, []).append(r.recall)

    scored = [r for r in ok if r.answer_status == "scored" and r.answer_score is not None]
    inputs = [r.input_tokens for r in ok if r.input_tokens is not None]
    outputs = [r.output_tokens for r in ok if r.output_tokens is not None]

    return {
        "total": len(results),
        "succeeded": len(ok),
        "failed": len(results) - len(ok),
        f"recall@{k}": round(statistics.mean(r.recall for r in retrieval_scored), 4)
                        if retrieval_scored else None,
        f"mrr@{k}": round(statistics.mean(r.mrr for r in retrieval_scored), 4)
                     if retrieval_scored else None,
        "retrieval_metric_coverage": round(len(retrieval_scored) / len(ok), 4),
        "tool_accuracy": round(statistics.mean(r.tool_acc for r in ok), 4),
        "latency_p50_ms": pct(0.50),
        "latency_p95_ms": pct(0.95),
        "avg_input_tokens": round(statistics.mean(inputs), 1) if inputs else None,
        "avg_output_tokens": round(statistics.mean(outputs), 1) if outputs else None,
        "input_token_coverage": round(len(inputs) / len(ok), 4),
        "output_token_coverage": round(len(outputs) / len(ok), 4),
        "answer_score_mean": round(statistics.mean(r.answer_score for r in scored), 4)
                             if scored else None,
        "answer_score_by_dimension": {
            dimension: round(statistics.mean(values), 4)
            for dimension in ANSWER_DIMENSIONS
            if (values := [
                r.answer_dimensions[dimension]
                for r in scored
                if r.answer_dimensions is not None
            ])
        },
        "answer_dimension_coverage": {
            dimension: round(sum(
                r.answer_dimensions is not None and dimension in r.answer_dimensions
                for r in ok
            ) / len(ok), 4)
            for dimension in ANSWER_DIMENSIONS
        },
        "answer_scored": len(scored),
        "answer_not_evaluated": sum(r.answer_status == "not_evaluated" for r in results),
        "answer_evaluation_failed": sum(r.answer_status == "error" for r in results),
        "answer_score_coverage": round(len(scored) / len(results), 4),
        # 按难度分组的 recall —— 这一栏告诉你该往哪个方向优化
        "recall_by_difficulty": {
            d: round(statistics.mean(v), 4) for d, v in sorted(by_difficulty.items())
        },
    }


def save_run(tag: str, summary: dict, results: list[CaseResult], config: dict,
             *, cases: list[EvalCase] | None = None,
             dataset_path: str | Path | None = None, split: str = "all") -> Path:
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
                           if dataset_path is not None else None,
            "selected_sha256": hashlib.sha256(snapshot_text.encode("utf-8")).hexdigest()
                               if snapshots is not None else None,
            "split": split,
            "samples": snapshots,
        },
        "summary": summary,
        "cases": [asdict(r) for r in results],
    }
    path = RESULTS_DIR / f"{tag}.json"
    # 不覆盖旧实验；换一个 tag，才能保留修复前后的证据。
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def export_review_template(run_path: str | Path, output_path: str | Path) -> Path:
    """把一次冻结的运行导出成人工可批改的 JSON，不重新调用 Agent。"""
    run_path = Path(run_path)
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    samples = payload.get("dataset", {}).get("samples")
    if not isinstance(samples, list):
        raise ValueError("运行结果缺少 dataset.samples，无法还原题目和评分标准")
    sample_by_id = {sample["id"]: EvalCase(**sample) for sample in samples}
    records = []
    for result in payload.get("cases", []):
        case_id = result["id"]
        if case_id not in sample_by_id:
            raise ValueError(f"运行结果中的 {case_id} 没有对应题目快照")
        case = sample_by_id[case_id]
        answer = str(result.get("answer", ""))
        records.append({
            "id": case_id,
            "fingerprint": answer_fingerprint(case, answer),
            "question": case.question,
            "reference": case.reference,
            "rubric": case.rubric,
            "retrieved_ids": result.get("retrieved_ids", []),
            "retrieved_contexts": result.get("retrieved_contexts", []),
            "answer": answer,
            "run_error": result.get("error"),
            "dimensions": {dimension: None for dimension in ANSWER_DIMENSIONS},
            "score": None,
            "reason": "",
            "evaluator": "human",
        })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(records, ensure_ascii=False, indent=2))
    return output_path


def _metric_k(summary: dict[str, Any]) -> int:
    """从旧报告的 recall@5 这类字段里取回当时使用的 k。"""
    for key in summary:
        if key.startswith("recall@") and key.removeprefix("recall@").isdigit():
            return int(key.removeprefix("recall@"))
    return 5


async def _apply_manual_scores(
    payload: dict[str, Any], evaluator: AnswerEvaluator
) -> list[CaseResult]:
    samples = payload.get("dataset", {}).get("samples")
    if not isinstance(samples, list):
        raise ValueError("运行结果缺少 dataset.samples，无法校验人工评分")
    sample_by_id = {sample["id"]: EvalCase(**sample) for sample in samples}
    results = []
    for raw_result in payload.get("cases", []):
        result = CaseResult(**raw_result)
        case = sample_by_id.get(result.id)
        if case is None:
            raise ValueError(f"运行结果中的 {result.id} 没有对应题目快照")
        # 兼容修正前保存的旧报告：按冻结题目重新确定这题能否计算检索指标。
        result.retrieval_evaluable = bool(case.relevant_ids)
        grade = await evaluator(case, result.answer)
        if grade is not None:
            grade.__post_init__()
            result.answer_status = "scored"
            result.answer_score = grade.score
            result.answer_dimensions = grade.dimensions
            result.answer_reason = grade.reason
            result.answer_evaluator = grade.evaluator
            result.evaluation_error = None
        results.append(result)
    return results


def apply_scores_to_run(
    run_path: str | Path, scores_path: str | Path, output_tag: str
) -> Path:
    """把人工评分应用到已保存的回答；生成新报告，绝不覆盖原始运行。"""
    if (not output_tag or output_tag in {".", ".."}
            or Path(output_tag).name != output_tag or "\\" in output_tag):
        raise ValueError("实验名字只能是文件名，不能包含目录")
    run_path = Path(run_path)
    scores_path = Path(scores_path)
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    evaluator = load_manual_evaluator(scores_path)
    results = asyncio.run(_apply_manual_scores(payload, evaluator))
    summary = summarize(results, k=_metric_k(payload.get("summary", {})))

    scored_payload = dict(payload)
    scored_payload.update({
        "tag": output_tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "cases": [asdict(result) for result in results],
        "scoring": {
            "source_run": run_path.name,
            "scores_file_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        },
    })
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / f"{output_tag}.json"
    with output_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(scored_payload, ensure_ascii=False, indent=2))
    return output_path


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
            # 召回、正确率等指标越大越好，但延迟恰好相反。原来的通用判断
            # 会把“P95 从 8 秒涨到 23 秒”错误标成绿色，容易让实验结论反过来。
            lower_is_better = key.startswith("latency_")
            if delta == 0:
                arrow = "⚪"
            elif lower_is_better:
                arrow = "🟢" if delta < 0 else "🔴"
            else:
                arrow = "🟢" if delta > 0 else "🔴"
            lines.append(f"| {key} | {va} | {vb} | {arrow} {delta:+.4f} |")

    if not a_version or not b_version:
        lines.append("\n⚠️ 历史结果缺少数据版本，比较结论需人工核对。")
    # 逐条回归检测；答案分数只比较同一评分器标注的样本。
    a_cases = {c["id"]: c for c in a["cases"]}
    regressed = [
        c["id"] for c in b["cases"]
        if c["id"] in a_cases and c["recall"] < a_cases[c["id"]]["recall"]
    ]
    answer_regressed = [
        c["id"] for c in b["cases"]
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


# ==============================================================================
# CLI
# ==============================================================================
def _main() -> int:
    parser = argparse.ArgumentParser(description="Agent 评测")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--tag", required=True, help="本次实验的名字，如 baseline / chunk500")
    p_run.add_argument(
        "--k",
        type=int,
        default=5,
        help="每次检索工具调用的指标截断位置；不改变实际检索数量",
    )
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

    p_review = sub.add_parser("review-template")
    p_review.add_argument("--run", required=True, help="已经冻结的运行结果 JSON")
    p_review.add_argument("--output", required=True, help="待填写的人工评分 JSON")

    p_apply = sub.add_parser("apply-scores")
    p_apply.add_argument("--run", required=True, help="已经冻结的运行结果 JSON")
    p_apply.add_argument("--scores", required=True, help="填写完成的人工评分 JSON")
    p_apply.add_argument("--tag", required=True, help="评分后新报告的名字")

    args = parser.parse_args()

    if args.cmd == "compare":
        print(compare(args.tag_a, args.tag_b))
        return 0
    if args.cmd == "review-template":
        print(f"已导出：{export_review_template(args.run, args.output)}")
        return 0
    if args.cmd == "apply-scores":
        print(f"已保存：{apply_scores_to_run(args.run, args.scores, args.tag)}")
        return 0

    # 把你的 agent 包装成 AgentFn 的形状，接在这里
    cases = EvalCase.load(args.dataset)
    if args.split != "all":
        cases = [case for case in cases if case.split == args.split]
    if args.mode == "retrieval":
        # 超范围题没有正确本地资料，不参与 recall/MRR；它们留给完整 Agent 模式
        # 检查工具选择和拒答行为。
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
    results = asyncio.run(run_eval(target, cases, k=args.k,
                                   answer_evaluator=evaluator, case_agent=case_agent))
    summary = summarize(results, k=args.k)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    path = save_run(args.tag, summary, results, current_config(), cases=cases,
                    dataset_path=args.dataset, split=args.split)
    print(f"\n已保存：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


# ==============================================================================
# domain/evalset.jsonl 长什么样（每行一个 JSON）
# ==============================================================================
# {"id":"q001","question":"FastAPI 默认文档在哪？","relevant_ids":["first_steps"],
#  "reference":"Swagger UI 默认位于 /docs...","expected_tools":["retrieve_fastapi_docs"],
#  "difficulty":"normal"}
#
# {"id":"q002","question":"今天美股怎么样？","relevant_ids":[],
#  "expected_tools":["web_search"],"difficulty":"out_of_scope"}
#   ↑ 这条测的是"该走联网而不是走本地检索"—— 负样本和正样本一样重要
#
# {"id":"q003","question":"幻觉和 reward hacking 有什么关系？",
#  "relevant_ids":["hallucination#3","reward-hacking#5"],"difficulty":"multi_hop"}
#   ↑ 需要跨两篇文档，最容易失败的一类
#
# 【建评测集的实操建议】
#   - 20~30 条就能开始，别追求一步到位
#   - 一定要有 out_of_scope 的负样本（测"知道自己不知道"）
#   - 一定要有 multi_hop（这类最能暴露 chunk_size 设太小的问题）
#   - 让业务方帮你标 30 条，价值远超你自己编 300 条
#   - 每次线上遇到答错的问题，就补进评测集 —— 评测集会自己长大
# ==============================================================================
