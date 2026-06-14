"""
rewriters/noop.py —— 不改写，原样返回查询。

用途：关闭多轮（退化成单轮）/ 评估基线 / 测试。
是 QueryRewriter 抽象的"零实现"，证明这层可插拔——换 LLMRewriter 只改 config 分支。
"""
from __future__ import annotations

from core.interfaces import QueryRewriter


class NoOpRewriter(QueryRewriter):
    """直通：无论有无历史，都原样返回 query。"""

    def rewrite(self, history: list[dict], query: str) -> str:
        return query