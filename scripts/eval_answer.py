"""
对生产配置的答案质量做评估。python scripts/eval_answer.py

跑法与检索评估不同：这里跑**完整 pipeline.run**（检索→生成），会真调
DeepSeek/kimi（有 API 成本、6 题可接受），拿到带答案的 RAGResult 再评。
评的是用户真实体验到的那套配置（rerank + semantic + vlm）。

指标（见 evaluators/answer.py）：
- CitationPrecision：答案标的 [n] 引用有几成有效（抓幻觉引用）
- AnswerCoverage   ：expected_answer 关键点的字面命中率（弱信号）

⚠️ expected_answer 需人工标注：每题写成 "关键点1;关键点2"（分号分隔），
   基于 PDF 真实内容填。关键点是答案里**该出现的核心事实词**，用于字面匹配，
   故选词要稳（用文档原词，避免同义改写导致字面漏判）。
"""
from core.config import RAGConfig, build_pipeline
from core.interfaces import RAGResult, EvalSample
from evaluators.answer import AnswerEvaluator

SAMPLES = [
    ("本研究将政策工具分为哪三类", "供给型;需求型;环境型"),
    ("最终确定的主题数是多少",     "12"),
    ("困惑度曲线说明了什么趋势",   "急剧下降;趋于平缓"),       # 关键点取最小核心词，避免前缀导致字面漏判
    ("本文采用了什么研究方法",     "LDA;社会网络分析"),
    ("本文分析了多少份政策文本",   "43"),
    # 下题关键点为论文表3的真实结论；当前答案答的是侧面(LDA可信度)，故 coverage 会偏低——
    # 这是评估该暴露的真问题(检索/生成对该题覆盖不足)，不为了好看而迁就答案改关键点。
    ("社会网络分析得出了什么结论", "12个主要节点;各节点贡献度接近;强制性治理"),
]

samples = [EvalSample(query=q, expected_answer=ea) for q, ea in SAMPLES]

cfg = RAGConfig(retriever="rerank", chunker="semantic",
                generator="llm", collection_name="docqa_v2")
pipeline = build_pipeline(cfg)

print(f"答案评估 | {len(SAMPLES)} 题 | 配置: {cfg.retriever}+{cfg.chunker}+{cfg.generator}\n")

results: list[RAGResult] = []
for q, _ in SAMPLES:
    r = pipeline.run(q, k=4)
    results.append(r)
    print(f"Q: {q}")
    print(f"   A: {r.answer.strip()[:80]}{'...' if len(r.answer.strip())>80 else ''}")

ev = AnswerEvaluator()
metrics = ev.evaluate(results, samples)
print("\n=== 指标 ===")
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

    # 终端日志：CitationPrecision: 1.0
    # 这意味着什么：大模型没有撒谎。在所有 6 道题里，无论是标了单个 [1]，还是连续标了 [1, 2, 2, 3]，这些数字全部都在真正检索到的文档块范围内。