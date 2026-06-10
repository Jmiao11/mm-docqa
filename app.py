"""
app.py —— Gradio 界面，通过 HTTP 调 FastAPI 后端（前后端分离）。
运行前先启动后端：  python -m uvicorn api.main:app
再运行本文件：      python app.py
然后浏览器开 http://127.0.0.1:7860
"""
from __future__ import annotations

import time

import gradio as gr
import requests

API = "http://127.0.0.1:8000"      # 后端地址
SESSION = "gradio-session"          # 简单起见，固定一个会话


# ---------- 调后端的函数 ----------
def upload_and_wait(file):
    """上传 PDF，然后轮询状态直到入库完成。用 yield 实时更新界面进度。"""
    if file is None:
        yield "请先选择一个 PDF 文件。"
        return

    # 1) 上传 → POST /documents
    try:
        with open(file.name, "rb") as f:
            resp = requests.post(
                f"{API}/documents",
                files={"file": (file.name.split("/")[-1].split("\\")[-1], f, "application/pdf")},
                timeout=30,
            )
        resp.raise_for_status()
    except Exception as e:
        yield f"上传失败：{e}\n（后端启动了吗？检查 {API} 是否可访问）"
        return

    doc_id = resp.json()["doc_id"]
    yield f"已上传，正在后台处理：{doc_id}\n状态：pending …"

    # 2) 轮询 → GET /documents/{id}/status
    for _ in range(300):                      # 最多等 ~5 分钟（含图 caption）
        time.sleep(1)
        s = requests.get(f"{API}/documents/{doc_id}/status", timeout=10).json()
        status = s["status"]
        if status == "indexed":
            yield (f"✅ 入库完成：{s['source']}\n"
                   f"   页数 {s['n_pages']}，切块 {s['n_chunks']} 块。现在可以提问了。")
            return
        if status == "failed":
            yield f"❌ 入库失败：{s.get('error', '未知错误')}"
            return
        yield f"处理中… 状态：{status}"

    yield "⏱ 处理超时，请检查后端日志。"


def refresh_docs():
    """GET /documents → 渲染成 Markdown 表格。"""
    try:
        docs = requests.get(f"{API}/documents", timeout=10).json()
    except Exception as e:
        return f"无法获取文档列表：{e}"
    if not docs:
        return "知识库为空，请先上传 PDF。"
    rows = ["| 文件 | 状态 | 页数 | 块数 |", "|---|---|---|---|"]
    for d in docs:
        rows.append(f"| {d['source']} | {d['status']} | {d['n_pages']} | {d['n_chunks']} |")
    return "\n".join(rows)


def ask(question, k):
    """POST /query → 返回 (答案markdown, gallery更新)。命中图走单独 Gallery 展示。"""
    if not question.strip():
        return "请输入问题。", gr.update(value=[], visible=False)
    try:
        resp = requests.post(
            f"{API}/query",
            json={"question": question, "k": int(k), "session_id": SESSION},
            timeout=120,
        )
        resp.raise_for_status()
    except Exception as e:
        return f"提问失败：{e}", gr.update(value=[], visible=False)

    data = resp.json()
    answer = data["answer"]
    citations = data.get("citations", [])
    images = data.get("images", [])

    out = [answer]
    if citations:
        out.append("\n---\n**引用来源：**")
        for c in citations:
            out.append(f"- [{c['n']}] {c['source']}  "
                       f"`#{c['id']}`  (相关度 {c['score']})")
    out.append(f"\n*（检索命中 {data.get('n_retrieved', 0)} 块）*")

    # images = [
    #     {"path": "/data/images/fig3.png", "n": 1},
    #     {"path": "/data/images/fig5.png", "n": 2},
    # ]
    # 命中图 → Gallery；同机直接读后端返回的本机绝对路径
    gallery = [(img["path"], f"图 [{img['n']}]") for img in images]
    # 结果：
    # [
    #     ("/data/images/fig3.png", "图 [1]"),
    #     ("/data/images/fig5.png", "图 [2]"),
    # ]
    return "\n".join(out), gr.update(value=gallery, visible=bool(gallery))
    # Gradio Gallery 要求传入 (图片路径, 标题) 的元组列表


# ---------- 界面布局 ----------
with gr.Blocks(title="mm-docqa 文档问答助手") as demo:
    gr.Markdown("# 📄 mm-docqa 多模态文档问答助手\n上传 PDF，就内容提问，得到带引用来源的答案。")

    with gr.Row():
        # 左栏：上传 + 文档列表
        with gr.Column(scale=1):
            gr.Markdown("### 知识库")
            file_in = gr.File(label="上传 PDF", file_types=[".pdf"])
            upload_btn = gr.Button("上传并入库", variant="primary")
            upload_status = gr.Textbox(label="处理状态", lines=4, interactive=False)
            refresh_btn = gr.Button("刷新文档列表")
            doc_list = gr.Markdown("知识库为空。")

        # 右栏：问答
        with gr.Column(scale=2):
            gr.Markdown("### 提问")
            question = gr.Textbox(label="你的问题", lines=2,
                                  placeholder="例如：这篇论文最终确定了多少个主题？是怎么确定的？")
            k_slider = gr.Slider(1, 20, value=4, step=1, label="检索块数 k")
            ask_btn = gr.Button("提问", variant="primary")
            answer_out = gr.Markdown(label="答案")
            gallery_out = gr.Gallery(label="命中插图（VLM 看图作答的图）",
                                     columns=2, object_fit="contain",
                                     height="auto", visible=False)

    # 绑定事件
    upload_btn.click(upload_and_wait, inputs=file_in, outputs=upload_status) \
              .then(refresh_docs, outputs=doc_list)        # 入库完自动刷新列表

    refresh_btn.click(refresh_docs, outputs=doc_list)
    ask_btn.click(ask, inputs=[question, k_slider], outputs=[answer_out, gallery_out])
    demo.load(refresh_docs, outputs=doc_list)              # 打开页面就加载列表


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)