"""
evaluators/judge.py —— LLM-as-judge 答案评估器，实现 core.interfaces.Evaluator。

为什么现在才上（answer.py 当初 defer 的两个理由，这里逐一解决）：
  1. 自评偏袒：答案用 DeepSeek 生成 → 裁判换 Moonshot kimi，跨厂商裁判，
     不让"运动员兼裁判"。裁判模型可注入 / 可配，默认与 generator 不同源。
  2. 非确定：同一输入多次采样(n_samples)，对每个维度取均值，并**报告方差**——
     方差本身是信号：某题方差大 = 裁判对它判得不稳，该题的分数不可尽信。

评分维度（一次调用、结构化 JSON 返回两个分，量纲 0/1/2）：
  correctness  对照 expected_answer：0 答错 / 1 部分对 / 2 答对。补 AnswerCoverage
               字面匹配的语义盲区（"12个主题"vs"十二个主题"字面漏判，judge 不漏）。
  groundedness 答案是否扎在检索到的上下文里：0 无据 / 1 部分有据 / 2 完全有据。
               RAG 命门指标(反幻觉)，与 CitationPrecision 互补：Precision 查"引用编号
               合不合法"，groundedness 查"答案内容到底有没有出处"。

诚实边界：
  - LLM-judge 仍是近似，必须**人工校准**才可信 → 提供 calibrate()，用一小批人工分
    算 judge-人工一致度(MAE + 完全一致率)。校准过、报告了一致度，这个数才敢往简历写。
  - 解析失败 / 调用失败 → 跳过该次采样、不崩；某题全部采样失败计入 n_failed，
    绝不静默把它当满分虚高。

不进 pipeline：Evaluator 是离线背书层，scripts 直接实例化（守承重墙）。
client 可注入（沙箱传 FakeClient，零网络验证聚合逻辑）。
"""
from __future__ import annotations

import json
import re
import statistics
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # 自足加载 .env：judge 可能在不建 LLM pipeline 的场景被单用，不能依赖别人先 load

from core.interfaces import Evaluator, RAGResult, EvalSample

# 去掉 LLM 可能给 JSON 包的 ```json ... ``` 围栏（很多模型改不掉这习惯）
_FENCE_RE = re.compile(r"```(?:json)?|```")

JUDGE_SYSTEM = (
    "你是严格的 RAG 答案评审。依据给定的【问题】【参考答案】【检索上下文】【待评答案】，"
    "对待评答案打两个分，每个分都是 0/1/2 的整数：\n"
    "correctness（对照参考答案，看事实对不对，允许同义改写，不看措辞）："
    "0=答错或答非所问；1=部分正确或不完整；2=正确且完整。\n"
    "groundedness（答案内容是否由检索上下文支撑，不看对错只看有没有出处）："
    "0=上下文里找不到依据/像凭空说的；1=部分有据；2=完全可由上下文支撑。\n"
    "只输出 JSON，不要任何解释或围栏，格式严格为："
    '{"correctness": <0|1|2>, "groundedness": <0|1|2>}'
)


def _build_user_prompt(question: str, reference: str, answer: str, context: str) -> str:
    ref = reference.strip() or "（无参考答案，仅凭上下文判断 correctness 是否答非所问）"
    return (
        f"【问题】\n{question}\n\n"
        f"【参考答案】\n{ref}\n\n"
        f"【检索上下文】\n{context}\n\n"
        f"【待评答案】\n{answer.strip()}\n\n"
        "请按 system 要求只输出 JSON。"
    )


class LLMJudgeEvaluator(Evaluator):
    """LLM-as-judge 答案评估器。裁判默认 Moonshot kimi（与生成用的 DeepSeek 异源）。"""

    def __init__(
        self,
        client=None,                                   # 可注入：沙箱传 FakeClient
        model: str = "kimi-k2.6",
        base_url: str = "https://api.moonshot.cn/v1",
        api_key_env: str = "MOONSHOT_API_KEY",
        n_samples: int = 3,                            # 同题采样次数：治非确定
        temperature: float = 1.0,  # kimi-k2.6 仅允许 temperature=1；>0 即可让多次采样有分布
        max_context_chars: int = 2000,                 # 上下文截断，控 token
    ):
        self._client = client
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.n_samples = max(1, n_samples)
        self.temperature = temperature
        self.max_context_chars = max_context_chars

    def _client_lazy(self):
        if self._client is None:
            import os
            from openai import OpenAI
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"环境变量 {self.api_key_env} 未设置，无法调用裁判模型。")
            self._client = OpenAI(api_key=key, base_url=self.base_url,
                                  timeout=60.0, max_retries=3)
        return self._client

    @staticmethod
    def _context_of(r: RAGResult, limit: int) -> str:
        """把检索块拼成带编号的上下文，截断到 limit 字符。"""
        parts = [f"[{i}] {ret.chunk.text.strip()}" for i, ret in enumerate(r.retrieved, 1)]
        return "\n".join(parts)[:limit]

    def _judge_once(self, question: str, reference: str,
                    answer: str, context: str) -> Optional[dict]:
        """单次裁判调用 → {'correctness':int,'groundedness':int}；任何失败返回 None。"""
        try:
            resp = self._client_lazy().chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": _build_user_prompt(
                        question, reference, answer, context)},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            raw = _FENCE_RE.sub("", raw).strip()
            obj = json.loads(raw)
            c = int(obj["correctness"])
            g = int(obj["groundedness"])
            # 钳到 [0,2]，挡住裁判偶尔越界打分
            return {"correctness": max(0, min(2, c)), "groundedness": max(0, min(2, g))}
        except Exception as e:
            print(f"[judge] 跳过一次采样: {e}")
            return None

    def evaluate(self, results: list[RAGResult],
                 samples: list[EvalSample]) -> dict[str, float]:
        by_query = {r.query: r for r in results}

        # 每题先得到「该题多次采样的均值」，再对所有题求全局均值
        corr_means: list[float] = []
        grnd_means: list[float] = []
        corr_vars: list[float] = []     # 每题 correctness 的采样方差（不稳定度）
        n_judged = 0
        n_failed = 0                    # 全部采样都失败的题数

        for s in samples:
            r = by_query.get(s.query)
            if r is None:
                continue

            context = self._context_of(r, self.max_context_chars)
            corr_samples: list[int] = []
            grnd_samples: list[int] = []
            for _ in range(self.n_samples):
                v = self._judge_once(s.query, s.expected_answer, r.answer, context)
                if v is None:
                    continue
                corr_samples.append(v["correctness"])
                grnd_samples.append(v["groundedness"])

            if not corr_samples:        # 这题彻底没判成
                n_failed += 1
                continue

            n_judged += 1
            corr_means.append(statistics.mean(corr_samples))
            grnd_means.append(statistics.mean(grnd_samples))
            # 单题方差：仅 1 次采样时定义为 0
            # statistics.pvariance 是 Python 标准库中 statistics 模块的一个函数，用于计算总体方差。
            corr_vars.append(statistics.pvariance(corr_samples) if len(corr_samples) > 1 else 0.0)

        if n_judged == 0:
            return {"JudgeCorrectness": 0.0, "JudgeGroundedness": 0.0,
                    "JudgeStability": 0.0, "n_judged": 0.0, "n_failed": float(n_failed)}

        # 归一到 [0,1] 便于和 Recall/Precision 这些率读在一起（原始量纲 0-2，除以 2）
        return {
            "JudgeCorrectness": round(statistics.mean(corr_means) / 2, 4),
            "JudgeGroundedness": round(statistics.mean(grnd_means) / 2, 4),
            # 稳定度：1 - 平均采样方差/最大可能方差(1.0)。越接近 1 越可信
            "JudgeStability": round(1 - statistics.mean(corr_vars) / 1.0, 4),
            "n_judged": float(n_judged),
            "n_failed": float(n_failed),
        }

    def calibrate(self, results: list[RAGResult], samples: list[EvalSample],
                  human: dict[str, int]) -> dict[str, float]:
        """人工校准：human = {query: 人工 correctness(0/1/2)}。
        对每个有人工分的题，跑一遍 judge 取均值(四舍五入到 0/1/2)，与人工分比对。
        返回 MAE(平均绝对误差，越小越好) 与 ExactMatch(完全一致率，越大越好)。
        —— 没校准过的 judge 分不可信；这个方法就是把"可信"做成可报告的数字。"""
        by_query = {r.query: r for r in results}
        abs_errs: list[float] = []
        exact = 0
        n = 0
        for s in samples:
            if s.query not in human:
                continue
            r = by_query.get(s.query)
            if r is None:
                continue

            context = self._context_of(r, self.max_context_chars)
            corr_samples = []

            for _ in range(self.n_samples):
                v = self._judge_once(s.query, s.expected_answer, r.answer, context)
                if v is not None:
                    corr_samples.append(v["correctness"])

            if not corr_samples:
                continue
            judge_score = round(statistics.mean(corr_samples))
            h = human[s.query]
            abs_errs.append(abs(judge_score - h))
            exact += 1 if judge_score == h else 0
            n += 1

        if n == 0:
            return {"MAE": 0.0, "ExactMatch": 0.0, "n_calibrated": 0.0}
        return {
            "MAE": round(statistics.mean(abs_errs), 4),
            "ExactMatch": round(exact / n, 4),
            "n_calibrated": float(n),
        }