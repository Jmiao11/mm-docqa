"""
app.py —— Gradio 界面，通过 HTTP 调 FastAPI 后端（前后端分离）。
运行前先启动后端：  python -m uvicorn api.main:app
再运行本文件：      python app.py
然后浏览器开 http://127.0.0.1:7860
"""
from __future__ import annotations

import html as _html
import inspect
import time
import uuid

import gradio as gr
import requests

API = "http://127.0.0.1:8000"      # 后端地址
# 会话标识改由前端 gr.State 持有（每次"新会话"换一个 uuid）；
# 前端 Chatbot 仅作展示，后端 SQLite messages 才是改写用的历史真相源，session_id 对齐两者。


# ---------- 调后端的函数 ----------
def upload_and_wait(file):
    """上传文档，然后轮询状态直到入库完成。用 yield 实时更新界面进度。"""
    if file is None:
        yield "请先选择一个文件（PDF / txt / md）。"
        return

    # 1) 上传 → POST /documents
    try:
        fname = file.name.split("/")[-1].split("\\")[-1]
        import mimetypes
        mime = mimetypes.guess_type(fname)[0] or "application/octet-stream"
        with open(file.name, "rb") as f:
            resp = requests.post(
                f"{API}/documents",
                files={"file": (fname, f, mime)},
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
        return "知识库为空，请先上传文档（PDF / txt / md）。"
    rows = ["| 文件 | 状态 | 页数 | 块数 |", "|---|---|---|---|"]
    for d in docs:
        rows.append(f"| {d['source']} | {d['status']} | {d['n_pages']} | {d['n_chunks']} |")
    return "\n".join(rows)


def refresh_doc_choices():
    """GET /documents → 取 doc_id 列表，更新删除下拉框的可选项。
    用 gr.update(choices=...) 才能动态改下拉框选项（同 Gallery 的 gr.update 机制）。"""
    try:
        docs = requests.get(f"{API}/documents", timeout=10).json()
    except Exception:
        return gr.update(choices=[], value=None)
    choices = [d["doc_id"] for d in docs]
    return gr.update(choices=choices, value=None)


def delete_doc(doc_id):
    """DELETE /documents/{doc_id} → 删某文档（向量+图片+元数据）。
    doc_id 含中文，requests 会自动 URL 编码，无需手动 quote。"""
    if not doc_id:
        return "请先在下拉框选择要删除的文档。"
    try:
        resp = requests.delete(f"{API}/documents/{doc_id}", timeout=30)
        resp.raise_for_status()
    except Exception as e:
        return f"删除失败：{e}"
    d = resp.json()
    if not d.get("deleted"):
        return f"未删除（文档不存在？）：{doc_id}"
    return (f"🗑 已删除：{doc_id}\n"
            f"   清理向量块 {d.get('n_chunks', 0)} 块，图片文件 {d.get('n_images', 0)} 个。")


def user_submit(question, history):
    """第一步（瞬时）：立刻把用户气泡加进对话 + 清空输入框，不等后端。
    第三个返回值把【原始问题字符串】存进 pending State，供第二步直接使用——
    不从 Chatbot 历史里取，因为 Gradio 会把 message 的 content 规整成 list（取出来非 str）。
    返回 (新history, 清空输入框, pending问题)。空问题不动、pending 置空。"""
    history = list(history or [])
    if not question.strip():
        return history, question, ""
    history.append({"role": "user", "content": question})
    return history, "", question


def _citation_details(c: dict) -> str:
    """把一条引用渲染成 HTML <details> 折叠块：摘要=引用行，展开=检索到的原文（可溯源）。
    用标准安全标签（details/summary/blockquote），Gradio sanitize 默认保留、浏览器原生折叠、零 JS。
    原文做 HTML 转义防注入/破版，换行转 <br>。"""
    head = _html.escape(f"[{c['n']}] {c['source']}  #{c['id']}  (相关度 {c.get('score','')})")
    body = _html.escape((c.get("text") or "").strip()) or "（无原文）"
    body = body.replace("\n", "<br>")
    tag = "🖼 图片说明（VLM caption）" if c.get("kind") == "image" else "📄 检索原文"
    return (f"<details><summary>{head}</summary>"
            f"<blockquote><b>{tag}</b><br>{body}</blockquote></details>")


def bot_respond(history, k, session_id, pending):
    """第二步（慢）：用 pending 里的原始问题打后端，追加助手气泡。
    返回 (新history, gallery更新)。pending 为空（空提交）则不动。"""
    history = list(history or [])
    question = (pending or "").strip()
    if not question:
        return history, gr.update()

    try:
        resp = requests.post(
            f"{API}/query",
            json={"question": question, "k": int(k), "session_id": session_id},
            timeout=120,
        )
        resp.raise_for_status()
    except Exception as e:
        history.append({"role": "assistant", "content": f"提问失败：{e}\n（后端启动了吗？{API}）"})
        return history, gr.update(value=[], visible=False)

    data = resp.json()
    answer = data["answer"]
    citations = data.get("citations", [])
    images = data.get("images", [])
    standalone = (data.get("standalone_query") or "").strip()

    parts = []
    # 改写可见：仅当历史感知改写真的改了句子时，把后端 [rewrite] 搬到台前
    if standalone and standalone != question:
        parts.append(f"> 🔁 检索用：{standalone}\n")
    parts.append(answer)
    if citations:
        parts.append("\n---\n**引用来源（点击展开原文溯源）：**\n")
        for c in citations:
            parts.append(_citation_details(c))
    parts.append(f"\n*（检索命中 {data.get('n_retrieved', 0)} 块）*")

    history.append({"role": "assistant", "content": "\n".join(parts)})
    gallery = [(img["path"], f"图 [{img['n']}]") for img in images]
    return history, gr.update(value=gallery, visible=bool(gallery))


def new_session():
    """开新会话：换一个 session_id（后端历史按新 key 起、旧历史不再被读回）+ 清空界面。
    旧消息留在 SQLite 但永不再读 → 翻篇且零删除，避免跨主题污染改写。
    返回 (新session_id, 清空Chatbot, 清空输入框, 隐藏Gallery)。"""
    return str(uuid.uuid4()), [], "", gr.update(value=[], visible=False)


def refresh_sessions(current_sid):
    """拉取会话列表填充下拉框；若当前会话已在列表中则保持选中（程序化设值不触发 .select，无循环）。"""
    try:
        resp = requests.get(f"{API}/sessions", timeout=30)
        resp.raise_for_status()
        sessions = resp.json()
    except Exception:
        return gr.update(choices=[])
    choices = [(f"{(s['title'] or '（无标题）')[:24]}（{s['n']}）", s["session_id"]) for s in sessions]
    val = current_sid if any(sid == current_sid for _, sid in choices) else None
    return gr.update(choices=choices, value=val)


def switch_session(session_id):
    """切换到某历史会话：拉它的消息、重建对话显示（用存好的 sources 复原可溯源 <details>）。
    session_state 设为该 id → 下一句提问的改写器自动读这个会话历史 → 自然续上。
    返回 (重建的history, session_state, 清空pending, 隐藏Gallery)。"""
    if not session_id:
        return [], str(uuid.uuid4()), "", gr.update(value=[], visible=False)
    try:
        resp = requests.get(f"{API}/sessions/{session_id}/messages", timeout=30)
        resp.raise_for_status()
        msgs = resp.json()
    except Exception as e:
        return ([{"role": "assistant", "content": f"加载会话失败：{e}"}],
                session_id, "", gr.update(value=[], visible=False))

    history = []
    for m in msgs:
        if m.get("role") == "user":
            history.append({"role": "user", "content": m.get("content", "")})
        else:
            parts = [m.get("content", "")]
            srcs = m.get("sources") or []
            if srcs:
                parts.append("\n---\n**引用来源（点击展开原文溯源）：**\n")
                for c in srcs:
                    parts.append(_citation_details(c))
            history.append({"role": "assistant", "content": "\n".join(parts)})
    return history, session_id, "", gr.update(value=[], visible=False)


# 约束命中插图缩略图：等比缩放 + 限高，避免 object_fit 在某些 gradio 版本压不住高度导致裁切。
# 点击放大的全屏预览是独立浮层、不受此 CSS 影响，仍看全图。
GALLERY_CSS = """
#hit-gallery img { object-fit: contain !important; max-height: 200px !important; }
#hit-gallery .grid-wrap, #hit-gallery .grid-container { max-height: 220px !important; }
/* 会话列表：限高 + 滚轮到边界不传导到页面，避免与页面滚动冲突 */
#session-list { max-height: 200px; overflow-y: auto; overscroll-behavior: contain; }
/* 每个选项独占一行 → 把默认的横向药丸网格变成干净的竖直列表（侧栏式） */
#session-list label { width: 100% !important; box-sizing: border-box !important;
    margin: 2px 0 !important; font-size: 0.9em !important; }
"""

# ---------- 界面布局 ----------
with gr.Blocks(title="mm-docqa 文档问答助手", css=GALLERY_CSS) as demo:
    gr.Markdown("# 📄 mm-docqa 多模态文档问答助手\n上传文档（PDF / txt / md），就内容提问，得到带引用来源的答案。")

    with gr.Row():
        # 左栏：上传 + 文档列表
        with gr.Column(scale=1):
            gr.Markdown("### 知识库")
            file_in = gr.File(label="上传文档（PDF / txt / md）",
                              file_types=[".pdf", ".txt", ".md"])
            upload_btn = gr.Button("上传并入库", variant="primary")
            upload_status = gr.Textbox(label="处理状态", lines=4, interactive=False)
            refresh_btn = gr.Button("刷新文档列表")
            doc_list = gr.Markdown("知识库为空。")

            del_dropdown = gr.Dropdown(label="选择要删除的文档", choices=[],
                                       interactive=True)
            del_btn = gr.Button("删除选中文档", variant="stop")
            del_status = gr.Textbox(label="删除结果", lines=2, interactive=False)

        # 右栏：多轮对话
        with gr.Column(scale=2):
            with gr.Row():
                gr.Markdown("### 提问")
                new_chat_btn = gr.Button("🆕 新会话", scale=0, min_width=100)
            # 会话列表用 Radio（竖直单选列表）而非 Dropdown：Dropdown 闭合时仍捕获滚轮、
            # 与页面滚动抢事件；Radio 不因滚轮改值，配 CSS overscroll-behavior 隔离滚动。
            session_dropdown = gr.Radio(label="历史会话（点击切换 / 继续）",
                                        choices=[], value=None, interactive=True,
                                        elem_id="session-list")
            # session_id 存在前端 State；每次新会话换 uuid。后端按此 key 取历史做改写。
            session_state = gr.State(str(uuid.uuid4()))
            pending = gr.State("")   # 两步链之间传递“待答的原始问题字符串”
            # 消息格式跨版本兼容：gradio ≥6 已默认 messages 且移除 type 参数；
            # 4.27–5.x 需显式 type="messages"。按签名探测，避免赌死某个版本。
            _cb_kw = {"label": "对话", "height": 600}
            if "type" in inspect.signature(gr.Chatbot.__init__).parameters:
                _cb_kw["type"] = "messages"
            # gr.Group 把对话框+输入框+发送键包成一个连续容器（去掉组件间分隔），
            # 仿网页大模型的一体式聊天面板。输入框 container=False 去掉自身边框、融进容器。
            with gr.Group():
                chatbot = gr.Chatbot(**_cb_kw)
                with gr.Row(equal_height=True):    # 输入框与发送键等高对齐，避免按钮溢出裁切
                    question = gr.Textbox(
                        show_label=False, container=False, lines=1, max_lines=6, scale=8,
                        placeholder="输入问题，回车发送、Shift+回车换行…（例：这篇论文确定了多少个主题？）",
                    )
                    ask_btn = gr.Button("提问", variant="primary", scale=1, min_width=88)
            k_slider = gr.Slider(1, 20, value=4, step=1, label="检索块数 k")
            gallery_out = gr.Gallery(label="命中插图（点击放大看原图）",
                                     elem_id="hit-gallery",
                                     columns=3, object_fit="contain",
                                     height=240, allow_preview=True, visible=False)   # 固定高度→缩略图；点击可放大看原图

        # 绑定事件
        upload_btn.click(upload_and_wait, inputs=file_in, outputs=upload_status) \
            .then(refresh_docs, outputs=doc_list) \
            .then(refresh_doc_choices, outputs=del_dropdown)  # 入库完同步列表+下拉框

        refresh_btn.click(refresh_docs, outputs=doc_list) \
            .then(refresh_doc_choices, outputs=del_dropdown)

        # 删除：删完同步刷新 Markdown 列表 + 下拉框（两处派生视图都要更新）
        del_btn.click(delete_doc, inputs=del_dropdown, outputs=del_status) \
            .then(refresh_docs, outputs=doc_list) \
            .then(refresh_doc_choices, outputs=del_dropdown)

        # 提问两步链：① user_submit 瞬时显示用户气泡+清空输入框+存 pending（queue=False 不排队）
        #            ② bot_respond 用 pending 打后端、追加助手气泡（慢，排队）③ 刷新会话列表
        _u_io = dict(inputs=[question, chatbot], outputs=[chatbot, question, pending])
        _b_io = dict(inputs=[chatbot, k_slider, session_state, pending], outputs=[chatbot, gallery_out])
        ask_btn.click(user_submit, **_u_io, queue=False).then(bot_respond, **_b_io) \
            .then(refresh_sessions, inputs=[session_state], outputs=[session_dropdown])
        question.submit(user_submit, **_u_io, queue=False).then(bot_respond, **_b_io) \
            .then(refresh_sessions, inputs=[session_state], outputs=[session_dropdown])

        # 切换会话：用户在下拉里选某会话 → 加载其历史、续上（.select 仅用户操作触发，刷新列表不会误触）
        session_dropdown.select(switch_session, inputs=[session_dropdown],
                                outputs=[chatbot, session_state, pending, gallery_out])

        # 新会话：换 session_id + 清空界面，再刷新列表（上一会话若已有消息会出现在列表里）
        new_chat_btn.click(new_session,
                           outputs=[session_state, chatbot, question, gallery_out]) \
            .then(refresh_sessions, inputs=[session_state], outputs=[session_dropdown])

        demo.load(refresh_docs, outputs=doc_list) \
            .then(refresh_doc_choices, outputs=del_dropdown) \
            .then(refresh_sessions, inputs=[session_state], outputs=[session_dropdown])  # 打开页面加载文档列表+删除下拉+会话列表

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)