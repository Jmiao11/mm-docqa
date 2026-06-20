"""
evaluators/golden.py —— 黄金评估集的单一真相源加载器。

为什么外置成 JSONL（而非 Python 字面量散在三个脚本里）：
  - 此前同一批 query 在 eval_run / eval_answer / eval_label 各抄一份，扩集要改三处、
    极易漂移。外置后**数据与代码解耦**：扩到 n>6 只追加一行、脚本一行不改。
  - JSONL 是真·数据集格式（一行一题、可被非代码工具编辑/版本化/diff）。

每行一个 JSON 对象，schema：
  query              str          必填。问题文本，也是 results↔samples 的配对键。
  expected_answer    str          可选。答案侧用：";" 分隔的关键点，供 AnswerEvaluator
                                  字面匹配 + LLMJudge 对照。
  relevant_chunk_ids list[str]    可选。检索侧用：人工标注的相关块 id，供 RetrievalEvaluator。

失败即报错（不静默丢题）：黄金集是背书基准，少一题都让指标失真，所以坏行带行号抛出。
重复 query 只告警不拦：evaluators 内部按 query 建字典，重复会悄悄相互覆盖，提前点名。
"""
from __future__ import annotations

import json
from pathlib import Path

from core.interfaces import EvalSample
from core.paths import GOLDEN_PATH


def load_golden(path: str | Path = GOLDEN_PATH) -> list[EvalSample]:
    """读 JSONL 黄金集 → list[EvalSample]。坏行带行号抛 ValueError；重复 query 告警。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"黄金集不存在: {path}")

    samples: list[EvalSample] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:                      # 跳过空行
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"黄金集第 {lineno} 行不是合法 JSON: {e}") from e

            query = obj.get("query")
            if not query:
                raise ValueError(f"黄金集第 {lineno} 行缺 query 字段")
            if query in seen:
                print(f"[golden] ⚠ 重复 query（第 {lineno} 行）将在按-query 配对时相互覆盖: {query}")
            seen.add(query)

            samples.append(EvalSample(
                query=query,
                expected_answer=obj.get("expected_answer", ""),
                relevant_chunk_ids=obj.get("relevant_chunk_ids", []),
            ))

    if not samples:
        raise ValueError(f"黄金集为空: {path}")
    return samples


def load_queries(path: str | Path = GOLDEN_PATH) -> list[str]:
    """只取 query 列表（eval_label 标注时只需问题、不需答案/标注）。"""
    return [s.query for s in load_golden(path)]