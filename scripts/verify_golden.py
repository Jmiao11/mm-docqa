"""沙箱验证黄金集 loader：真集解析 + 坏行带行号报错 + 缺 query 报错 + 重复告警。
python scripts/verify_golden.py（零网络）"""
import json
import tempfile
from pathlib import Path

from evaluators.golden import load_golden, load_queries
from core.paths import GOLDEN_PATH


def _write(lines: list[str]) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    f.write("\n".join(lines) + "\n")
    f.close()
    return Path(f.name)


def main():
    # ① 真集解析：6 题、字段齐、无重复
    real = load_golden(GOLDEN_PATH)
    assert len(real) == 6, f"真集应 6 题，实得 {len(real)}"
    q1 = next(s for s in real if s.query == "本研究将政策工具分为哪三类")
    assert q1.expected_answer == "供给型;需求型;环境型", q1.expected_answer
    assert q1.relevant_chunk_ids == ["9b6fa194ac750cef", "501ae4da6dd4944f"], q1.relevant_chunk_ids
    assert load_queries(GOLDEN_PATH) == [s.query for s in real]
    print(f"[OK] 真集解析 {len(real)} 题，字段齐")

    # ② round-trip + 缺省字段（只给 query）
    p = _write([
        json.dumps({"query": "只有问题"}, ensure_ascii=False),
        "",  # 空行应被跳过
        json.dumps({"query": "带答案", "expected_answer": "a;b"}, ensure_ascii=False),
    ])
    rt = load_golden(p)
    assert len(rt) == 2, rt
    assert rt[0].expected_answer == "" and rt[0].relevant_chunk_ids == [], rt[0]
    print("[OK] round-trip + 缺省字段 + 跳空行")

    # ③ 坏 JSON 带行号报错
    bad = _write([json.dumps({"query": "好"}, ensure_ascii=False), "{不是json"])
    try:
        load_golden(bad)
        assert False, "坏行应抛 ValueError"
    except ValueError as e:
        assert "第 2 行" in str(e), e
    print(f"[OK] 坏 JSON 带行号报错")

    # ④ 缺 query 报错
    noq = _write([json.dumps({"expected_answer": "x"}, ensure_ascii=False)])
    try:
        load_golden(noq)
        assert False, "缺 query 应抛 ValueError"
    except ValueError as e:
        assert "缺 query" in str(e), e
    print("[OK] 缺 query 报错")

    # ⑤ 重复 query 告警（不拦截，仍返回）
    dup = _write([json.dumps({"query": "重复"}, ensure_ascii=False),
                  json.dumps({"query": "重复"}, ensure_ascii=False)])
    print("  （下面应出现一条 [golden] ⚠ 重复告警）")
    res = load_golden(dup)
    assert len(res) == 2, res
    print("[OK] 重复 query 告警但不拦截")

    print("\n全部通过 ✅")


if __name__ == "__main__":
    main()