"""
rewriters/llm.py —— 用 DeepSeek 做"历史感知查询改写"（condense question）。

把依赖上文的追问（"那它怎么用？"、"上面第一点展开"）改写成自洽的独立检索 query，
再交给现有 pipeline.run —— 改写吸收历史依赖，下游 retrieve/generate 保持无状态单轮。

复用 LLMGenerator 同一套 OpenAI 兼容客户端约定（DeepSeek base_url + DEEPSEEK_API_KEY +
延迟创建），不另起一套。client 可注入，便于沙箱 FakeClient 单测、不依赖网络。

两条铁律（见 QueryRewriter 接口）：
- 首轮（history 为空）→ 原样返回，不调 LLM（省延迟/成本）。
- 改写失败 / 返回空 → 回退原句，绝不拖垮检索。
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv()

from core.interfaces import QueryRewriter

REWRITE_SYSTEM = (
    "你是一个查询改写助手。根据【对话历史】把用户的【追问】改写成一个"
    "语义自洽、可独立用于检索的查询：把指代（它/这个/上面）和省略补全成明确名词。\n"
    "严格遵守：\n"
    "1. 只输出改写后的查询本身，不要任何解释、前缀或标点包裹。\n"
    "2. 若追问本身已自洽、不依赖历史，原样输出。\n"
    "3. 不要回答问题，只做改写。"
)


class LLMRewriter(QueryRewriter):
    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        api_key_env: str = "DEEPSEEK_API_KEY",
        temperature: float = 0.0,   # 改写要确定性，温度拉到 0
        max_turns: int = 5,         # 只取最近 N 条历史，限 prompt 体积与成本
        client=None,                # 可注入：沙箱 FakeClient 单测用
    ):
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_turns = max_turns
        self._client = client       # None 则延迟创建真客户端

    def _client_lazy(self):
        if self._client is None:
            from openai import OpenAI
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(
                    f"环境变量 {self.api_key_env} 未设置。请先在本地配置 DeepSeek API key。"
                )
            self._client = OpenAI(api_key=key, base_url=self.base_url)
        return self._client

    def _build_messages(self, history: list[dict], query: str) -> list[dict]:
        recent = history[-self.max_turns:]
        lines = []
        for m in recent:
            role = "用户" if m.get("role") == "user" else "助手"
            content = (m.get("content") or "").strip()
            if content:
                lines.append(f"{role}：{content}")
        history_text = "\n".join(lines) if lines else "（无）"
        user = (
            f"【对话历史】\n{history_text}\n\n"
            f"【追问】\n{query}\n\n"
            f"请输出改写后的独立查询："
        )
        return [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": user},
        ]

    def rewrite(self, history: list[dict], query: str) -> str:
        if not history:                       # 首轮：不调 LLM
            return query
        messages = self._build_messages(history, query)
        try:
            resp = self._client_lazy().chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature,
            )
            rewritten = (resp.choices[0].message.content or "").strip()
            return rewritten or query          # 空回退原句
        except Exception as e:                 # 改写失败不能拖垮检索 → 降级单轮
            print(f"[rewrite] 改写失败，回退原句：{e}")
            return query