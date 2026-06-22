"""对生产配置的答案质量做评估。python scripts/eval_answer.py

跑法：跑**完整 pipeline.run**(检索→生成)，真调 DeepSeek/kimi，拿带答案的
RAGResult 再评。评的是用户真实体验那套配置(rerank+semantic+vlm)。

题目从 evaluators/golden.jsonl 读（单一真相源），用 expected_answer 字段。

两层评估：
  确定性(免费、快、常驻)：CitationPrecision(抓幻觉引用) + AnswerCoverage(字面命中，弱信号)。
  LLM-as-judge(贵、准、opt-in)：设环境变量 RUN_JUDGE=1 启用，跨厂商裁判+多采样+方差。
    —— 默认关，避免普通跑 eval 意外烧 API。
"""
import os

from core.config import RAGConfig, build_pipeline
from core.interfaces import RAGResult
from evaluators.answer import AnswerEvaluator
from evaluators.golden import load_golden

samples = load_golden()      # 答案侧：每题都有 expected_answer

cfg = RAGConfig(retriever="rerank", chunker="semantic",
                generator="llm", collection_name="docqa_v2")
pipeline = build_pipeline(cfg)

print(f"答案评估 | {len(samples)} 题 | 配置: {cfg.retriever}+{cfg.chunker}+{cfg.generator}\n")

results: list[RAGResult] = []
for s in samples:
    r = pipeline.run(s.query, k=4)
    results.append(r)
    print(f"Q: {s.query}")
    print(f"   A: {r.answer.strip()[:80]}{'...' if len(r.answer.strip())>80 else ''}")

ev = AnswerEvaluator()
metrics = ev.evaluate(results, samples)
print("\n=== 确定性指标 ===")
for k, v in metrics.items():
    print(f"  {k}: {v}")

import re
_CITE = re.compile(r"\[(\d+)\]")
print("\n--- per-query 明细 ---")
by_q = {s.query: s for s in samples}
for r in results:
    s = by_q[r.query]
    cited = [int(n) for n in _CITE.findall(r.answer)]
    n_ret = len(r.retrieved)
    invalid = [n for n in cited if not (1 <= n <= n_ret)]
    cite_flag = f"⚠ 幻觉引用{invalid}" if invalid else ("无引用" if not cited else "引用OK")
    points = [p.strip() for p in s.expected_answer.split(";") if p.strip()]
    missed = [p for p in points if p not in r.answer]
    miss_flag = f"漏掉关键点{missed}" if missed else "覆盖全"
    print(f"  {s.query[:18]:20} 检索{n_ret}块 引用{cited} {cite_flag} | {miss_flag}")


# ---- LLM-as-judge：opt-in，真调 API ----
if os.environ.get("RUN_JUDGE") == "1":
    from evaluators.judge import LLMJudgeEvaluator
    judge = LLMJudgeEvaluator(n_samples=3)
    print(f"\n=== LLM-as-judge (裁判={judge.model}, n_samples={judge.n_samples}, 真调 API) ===")
    jm = judge.evaluate(results, samples)
    for k, v in jm.items():
        print(f"  {k}: {v}")
    print("  注：JudgeStability 越接近 1 越可信；分数需经 calibrate() 与人工校准后方可对外引用。")
else:
    print("\n（设 RUN_JUDGE=1 启用 LLM 裁判评估；会真调 kimi、按题×n_samples 次、有成本）")