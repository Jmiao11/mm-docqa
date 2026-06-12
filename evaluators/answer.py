"""
evaluators/answer.py —— 答案质量指标，实现 core.interfaces.Evaluator。

设计取舍：第一版只做**确定性**指标，不上 LLM-as-judge。
理由：评估要可复现（同输入同输出）才能可信地背书改进；LLM-judge 非确定、
且"用 DeepSeek 答、再用 DeepSeek 评"有自评偏袒。LLM-judge 列入 roadmap。

两个指标（均不依赖大模型）：

CitationPrecision : 答案标注的 [n] 引用里，有几成 n 落在检索结果编号范围(1..N)内。
                    直接抓"幻觉引用"（标了不存在的来源）——背书本项目"带引用来源"卖点。
                    与 generators.llm.collect_citations 用同一正则解析 [n]，标准一致。
                    无引用的答案不计入精确率（precision 对它没发言权），但单独计 n_no_citation。
                    注：只做 Precision；Recall(该引未引)缺 ground truth、需语义标注，不做。

AnswerCoverage    : 弱信号。expected_answer 按 ';' 切成关键点，逐点子串匹配生成答案。
                    coverage = 命中关键点数 / 总关键点数。
                    明确是**字面子串匹配**，不等于"正确率"——"12个主题"vs"十二个主题"会漏判。

results 与 samples 按 query 文本配对。
- Precision 评所有有引用的样本；Coverage 只评 expected_answer 非空的样本。
"""
from __future__ import annotations

import re

from core.interfaces import Evaluator, RAGResult, EvalSample

_CITE_RE = re.compile(r"\[(\d+)\]")        # 与 collect_citations 同款，解析口径一致


class AnswerEvaluator(Evaluator):
    def evaluate(self, results: list[RAGResult],
                 samples: list[EvalSample]) -> dict[str, float]:
        by_query = {r.query: r for r in results}

        # --- Citation Precision ---
        prec_sum = 0.0
        n_prec = 0              # 参与精确率统计的样本数（有引用的）
        n_no_citation = 0      # 答案一个引用都没标的样本数（单独记，不混入精确率）

        # --- Answer Coverage ---
        cov_sum = 0.0
        n_cov = 0              # 有 expected_answer 的样本数

        for s in samples:
            r = by_query.get(s.query)
            if r is None:
                continue

            # ---- Citation Precision ----
            cited = [int(n) for n in _CITE_RE.findall(r.answer)]
            n_retrieved = len(r.retrieved)        # 有效编号范围 1..N
            if not cited:
                n_no_citation += 1                # 无引用：不参与精确率
            else:
                valid = sum(1 for n in cited if 1 <= n <= n_retrieved)
                # 解析：r.retrieved 是这次检索真正找出来的文本块。如果找出来 3 块，那合法的引用编号只能是 1、2、3。这里的 n_retrieved 就是 3。valid一定在这之间
                prec_sum += valid / len(cited)
                n_prec += 1

            # ---- Answer Coverage（弱信号，字面子串匹配）----
            if s.expected_answer:
                points = [p.strip() for p in s.expected_answer.split(";") if p.strip()]
                # 解析：把标准答案 "数据层; 接口层 ; " 用分号切开，并用 strip() 去掉前后的空格，最后把空字符串过滤掉。得到一个干净的列表：['数据层', '接口层']。
                if points:
                    hit = sum(1 for p in points if p in r.answer)
                    cov_sum += hit / len(points)
                    n_cov += 1

        return {
            "CitationPrecision": round(prec_sum / n_prec, 4) if n_prec else 0.0,
            "AnswerCoverage": round(cov_sum / n_cov, 4) if n_cov else 0.0,
            "n_cited": float(n_prec),          # 参与精确率统计的样本数
            "n_no_citation": float(n_no_citation),
            "n_coverage": float(n_cov),        # 参与覆盖统计的样本数
        }