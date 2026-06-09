"""
ingest/captioner.py —— 用 VLM(Moonshot kimi-k2.6) 给图表生成中文 caption。

caption-then-embed 的关键：图本身进不了文本向量空间，所以让 VLM 把每张图
描述成一段中文文字（图类型 + 关键维度 + 关键数值/趋势），这段 caption 作为
"文本"参与 bge 嵌入检索 —— 于是"困惑度曲线说明了什么"这类问题能语义召回到图。

OpenAI 兼容调用，和 llm.py 调 DeepSeek 同一套；client 可注入便于测试。

_safe_caption          → 把"可能失败"变成"返回 None"
ThreadPoolExecutor     → 把"串行慢"变成"并行快"
if not cap: continue   → 把"None"过滤掉，只保留成功结果
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

_PROMPT = (
    "这是一篇学术论文里的插图。用2-3句中文描述："
    "①图的类型（如折线图/社会网络图/气泡图/主题距离地图/柱状图）；"
    "②横纵轴或主要维度；③图中体现的关键信息、数值或趋势（有极值/拐点请点明）。"
    "只输出客观描述，不要客套话、不要猜测论文结论。"
)


class Captioner:
    def __init__(self, client=None, model: str = "kimi-k2.6",
                 api_key_env: str = "MOONSHOT_API_KEY",
                 base_url: str = "https://api.moonshot.cn/v1",
                 prompt: str = _PROMPT, temperature: float = 1.0):
        self._client = client            # 注入：测试时传假 client
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.prompt = prompt
        self.temperature = temperature

    def _client_lazy(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ[self.api_key_env],
                                  base_url=self.base_url,
                                  timeout=60.0, max_retries=3)  # 429 自动退避重试
        return self._client
    def caption(self, image_path: str | Path) -> str:
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        data_url = f"data:image/png;base64,{b64}"
        resp = self._client_lazy().chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": self.prompt},
            ]}],
        )
        return resp.choices[0].message.content.strip()


def _safe_caption(captioner: "Captioner", f: dict):
    """单张图 caption，失败返回 None（不拖垮整篇入库）。"""
    try:
        return f, captioner.caption(f["path"])
    except Exception as e:
        print(f"[captioner] 跳过图 {f.get('path')}: {e}")
        return f, None

# 前向引用
def build_image_chunks(figures: list[dict], captioner: "Captioner", source: str,
                       max_workers: int = 2) -> list:
    """图 → 可检索图块。并发 caption（默认4并行）缩短入库时间；单张失败跳过、不拖垮全篇。"""
    from concurrent.futures import ThreadPoolExecutor
    from core.interfaces import Chunk

    if not figures:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(lambda f: _safe_caption(captioner, f), figures))

    chunks = []
    for f, cap in results:
        if not cap:
            continue
        chunks.append(Chunk(
            id=Chunk.make_id(source + "#img", f["page"], f["xref"]),
            text=cap, source=source, start=0, end=0,
            metadata={"image_path": f["path"], "page": f["page"],
                      "xref": f["xref"], "kind": "image"},
        ))
    return chunks