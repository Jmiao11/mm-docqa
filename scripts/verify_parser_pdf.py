"""
2.1 验证：把真实 PDF 解析成 Document，检查页数、文本质量、页码反查。
运行：  python verify_parser_pdf.py
"""
from ingest.parser import parse_pdf, page_of

# 你的 PDF 在项目根目录。文件名很长，用变量存，避免手抖打错。
PDF = "../data/uploads/政策工具视域下我国省级数字经济政策文本的量化分析——基于LDA的主题社会网络分析_陈美.pdf"


def main():
    doc = parse_pdf(PDF)
    meta = doc.metadata

    print(f"source     : {doc.source}")
    print(f"总页数     : {meta['n_pages']}")
    print(f"全文字符数 : {len(doc.text)}")
    print(f"每页起始位置(前5个): {meta['page_offsets'][:5]}")

    print("\n=== 全文前 300 字 ===")
    print(doc.text[:300])

    print("\n=== 全文第 1000~1300 字（看正文质量）===")
    print(doc.text[1000:1300])

    # 页码反查：取几个字符位置，看落在第几页
    print("\n=== 页码反查 ===")
    for off in [0, 500, 2000, len(doc.text) - 1]:
        print(f"  字符位置 {off:>6} → 第 {page_of(off, meta['page_offsets'])} 页")


if __name__ == "__main__":
    main()