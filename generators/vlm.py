"""
generators/vlm.py —— VLMGenerator：检索命中图块时把图喂 kimi-k2.6 看图作答。

接口隔离兑现：generate 签名不变（query, contexts: list[Retrieved]）；继承
LLMGenerator 复用 build_context(编号→来源映射) 与 collect_citations(引用解析)，
所以 routes 的 build_context/collect_citations 调用零改动、引用链照常成立。

分流：
- 无图块 → 委托父类 super().generate 走 DeepSeek（便宜快）。
- 命中图块(metadata.kind == "image") → 编号文本上下文 + 按编号附对应图(base64)，
  一次 kimi-k2.6 多模态调用看图作答。

注意：kimi-k2.6 看图调用只接受 temperature=1.0（否则 400），默认即设 1.0。
"""
from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

from core.interfaces import Retrieved
from generators.llm import LLMGenerator

VLM_SYSTEM_PROMPT = (
    "你是一个严谨的多模态文档问答助手。提供的【资料】包含文字与论文插图。请严格遵守：\n"
    "1. 只能依据【资料】(含图)回答，不得使用资料之外的知识或自行编造；"
    "信息不足时直接回答：根据现有资料无法回答该问题。\n"
    "2. 图以\"图[n]\"给出，n 对应资料里的编号；看图作答时在相应句子末尾标注该编号，如 [2]。\n"
    "3. 回答简洁、准确，不复述问题，不输出无关内容。"
)


class VLMGenerator(LLMGenerator):
    def __init__(
        self,
        vlm_model: str = "kimi-k2.6",
        vlm_base_url: str = "https://api.moonshot.cn/v1",
        vlm_api_key_env: str = "MOONSHOT_API_KEY",
        vlm_temperature: float = 1.0,   # kimi-k2.6 看图只接受 1.0
        vlm_client=None,                # 可注入：测试时传假 client
        **kwargs,                       # 透传给父类：model/base_url/temperature 等文本路径参数
    ):
        super().__init__(**kwargs)      # 文本路径仍走 DeepSeek
        self.vlm_model = vlm_model
        self.vlm_base_url = vlm_base_url
        self.vlm_api_key_env = vlm_api_key_env
        self.vlm_temperature = vlm_temperature
        self._vlm_client = vlm_client

    def _vlm_lazy(self):
        if self._vlm_client is None:
            from openai import OpenAI
            key = os.environ.get(self.vlm_api_key_env)
            if not key:
                raise RuntimeError(
                    f"环境变量 {self.vlm_api_key_env} 未设置，无法调用 VLM。"
                )
            self._vlm_client = OpenAI(
                api_key=key, base_url=self.vlm_base_url,
                timeout=60.0, max_retries=3,
            )
        return self._vlm_client

    @staticmethod
    def _image_contexts(mapping: dict[int, Retrieved]) -> list[tuple[int, str]]:
        """从编号映射里挑出"是图块且文件存在"的，返回 [(编号, 路径), ...]。"""
        out = []
        for n, r in mapping.items():
            md = r.chunk.metadata or {}
            p = md.get("image_path")
            if md.get("kind") == "image" and p and Path(p).exists():
                out.append((n, p))
        return out

    @staticmethod
    def _data_url(path: str) -> str:
        mime = mimetypes.guess_type(path)[0] or "image/png"
        #  mimetypes 是 Python 内置库，它会根据传入路径的后缀名（比如 .jpg、.webp）自动推断出标准的网络媒体类型（如 image/jpeg）
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
        # .decode()：把 Base64 的 bytes 类型解码成 Python 的普通字符串 str 类型，方便下一步做字符串拼接
        return f"data:{mime};base64,{b64}"

    def generate(self, query: str, contexts: list[Retrieved]) -> str:
        if not contexts:
            return "根据现有资料无法回答该问题。"

        # 复用父类编号映射 → 引用编号与纯文本路径完全一致
        context_text, mapping = self.build_context(contexts)
        imgs = self._image_contexts(mapping)

        if not imgs:  # 无图 → 父类 DeepSeek，便宜快
            print("[vlm] 命中图块 0 张 → 回退 DeepSeek 纯文本作答")  # ← 新增
            return super().generate(query, contexts)

        print(f"[vlm] 命中图块 {len(imgs)} 张(编号 {[n for n, _ in imgs]}) "  # ← 新增
              f"→ {self.vlm_model} 多模态看图作答")  # ← 新增


        # 命中图块：编号文本上下文 + 按编号附图，一次多模态调用
        content = [{"type": "text", "text": f"【资料】\n{context_text}"}]
        for n, path in imgs:
            content.append({"type": "text", "text": f"图[{n}]："})
            content.append({"type": "image_url",
                            "image_url": {"url": self._data_url(path)}})
        content.append({"type": "text",
                        "text": f"\n【问题】\n{query}\n请依据上述资料(含图)作答，并标注引用编号。"})

        resp = self._vlm_lazy().chat.completions.create(
            model=self.vlm_model,
            temperature=self.vlm_temperature,
            messages=[
                {"role": "system", "content": VLM_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        )
        return resp.choices[0].message.content.strip()