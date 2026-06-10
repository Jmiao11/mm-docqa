"""
LLMGenerator：用 DeepSeek 做检索增强的文本生成，实现 core.interfaces.Generator。
可直接替换占位的 TemplateGenerator，pipeline 不改。

设计三要点（对照 1.2 原理）：
1. 系统提示把模型从"闭卷"逼成"开卷"：只依据提供的资料答，没有就说不知道 → 抗幻觉。
2. 上下文按编号 [1][2][3] 拼接，要求模型在句末标编号；代码再把编号映射回真实来源。
3. generate 收的是 list[Retrieved]（非纯文本），所以能同时拿到正文和来源 → 引用链成立。

注意：DeepSeek 模型名会变（旧的 deepseek-chat/reasoner 2026-07-24 停用）。
当前用 deepseek-v4-flash（便宜快，适合文档问答）。换模型只改 model 参数。
如报"model not found"，去 https://api-docs.deepseek.com 查最新模型名。
"""
from __future__ import annotations

import os

from dotenv import load_dotenv  # <--- 新增这行
load_dotenv()                   # <--- 新增这行：这行一执行，.env 里的 Key 就会瞬间注入到系统环境里

from core.interfaces import Generator, Retrieved

SYSTEM_PROMPT = (
    "你是一个严谨的文档问答助手。请严格遵守：\n"
    "1. 只能依据下面提供的【资料】回答问题，不得使用资料之外的知识或自行编造。\n"
    "2. 如果资料中没有足以回答问题的信息，直接回答：根据现有资料无法回答该问题。\n"
    "3. 在用到某条资料时，在相应句子末尾标注其编号，如 [1]、[2]；可同时引用多条，如 [1][3]。\n"
    "4. 回答简洁、准确，不要复述问题，不要输出无关内容。"
)

class LLMGenerator(Generator):
    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        api_key_env: str = "DEEPSEEK_API_KEY",
        max_context_chars: int = 3000,      # 上下文拼接总长上限，防撑爆
        temperature: float = 0.2,           # 问答要稳，温度调低
    ):
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.max_context_chars = max_context_chars
        self.temperature = temperature
        self._client = None                 # 延迟创建：拼 prompt/解析不需要它


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


    # ---- 1) 拼上下文：编号 -> 真实来源的映射在这里建立（可单测，无网络） ----
    def build_context(self, contexts: list[Retrieved]) -> tuple[str, dict[int, Retrieved]]:
        blocks, mapping, used = [], {}, 0
        for i, r in enumerate(contexts, 1):
            text = r.chunk.text.strip()
            if used + len(text) > self.max_context_chars and blocks:
                break

            blocks.append(f"[{i}]{text}")
            mapping[i] = r
            used += len(text)

            # mapping 是一个字典 {
            #     1: Retrieved(chunk=Chunk(text="LDA主题数为12...", source="论文.pdf", ...), score=0.86),
            #     2: Retrieved(chunk=Chunk(text="困惑度曲线显示...", source="论文.pdf", ...), score=0.74),
            #     3: Retrieved(chunk=Chunk(text="[图caption]社会网络图...",
            #                              metadata={"kind": "image", "image_path": "/data/images/fig3.png"}, ...),
            #                  score=0.61),
            # }

        return "\n\n".join(blocks), mapping


    def build_messages(self, query: str, context_text: str) -> list[dict]:
        user = (
            f"【资料】\n{context_text}\n\n"
            f"【问题】\n{query}\n\n"
            f"请依据上述资料作答，并标注引用编号。"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]


    # ---- 2) 解析模型输出里用到的编号 -> 整理成来源清单（可单测，无网络） ----
    @staticmethod
    def collect_citations(answer: str, mapping: dict[int, Retrieved]) -> list[dict]:
        import re
        used_nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
        sources = []
        for n in used_nums:
            r = mapping.get(n)
            if r is not None:
                sources.append({
                    "n": n, "id": r.chunk.id,
                    "source": r.chunk.source, "score": round(r.score, 3),
                })
        return sources

    # ---- 3) 接口实现：拼 -> 调 -> 答（这一步需要真 API key） ----
    def generate(self, query: str, contexts: list[Retrieved]) -> str:
        if not contexts:
            return "根据现有资料无法回答该问题。"

        context_text, mapping = self.build_context(contexts)
        messages = self.build_messages(query, context_text)

        resp = self._client_lazy().chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature,
        )
        answer = resp.choices[0].message.content.strip()

        return answer
