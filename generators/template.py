"""占位生成器：不调 LLM，把命中的上下文拼成带来源的模板答案。
只为验证"检索结果能流到生成层、引用信息没丢"。真 LLM 在 Phase 1 的 generators/llm.py。"""
from __future__ import annotations

from core.interfaces import Generator, Retrieved


class TemplateGenerator(Generator):
    def generate(self, query: str, contexts: list[Retrieved]) -> str:
        if not contexts:
            return f"【未检索到相关内容】问题：{query}"
        lines = [f"问题：{query}", "", "根据检索到的内容："]
        for i, r in enumerate(contexts, 1):
            snippet = r.chunk.text.strip().replace("\n", " ")
            if len(snippet) > 80:
                snippet = snippet[:80] + "…"
            lines.append(f"  [{i}] (来源 {r.chunk.source} #{r.chunk.id}, "
                         f"score={r.score:.3f}) {snippet}")
        return "\n".join(lines)
