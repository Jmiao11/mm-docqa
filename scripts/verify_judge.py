"""沙箱验证 LLM-as-judge 聚合逻辑，用假裁判 client，零网络。python scripts/verify_judge.py

覆盖：①稳定题(方差0) ②不稳题(采样分歧→方差>0→Stability<1) ③某题全部采样解析失败
(→n_failed、整题跳过) ④混合(一次坏解析+两次正常→仍判成) ⑤calibrate 的 MAE/一致率。
"""
from core.interfaces import Chunk, Retrieved, RAGResult, EvalSample
from evaluators.judge import LLMJudgeEvaluator


# ---- 假裁判 client：模仿 openai 的 client.chat.completions.create ----
class _Msg:
    def __init__(self, content): self.message = type("M", (), {"content": content})()
class _Resp:
    def __init__(self, content): self.choices = [_Msg(content)]
class _Completions:
    def __init__(self, outer): self.outer = outer
    def create(self, model, temperature, messages):
        user = messages[-1]["content"]
        # 用待评答案里的标记决定返回什么；按标记计数，模拟多次采样的分布
        for marker, script in SCRIPTS.items():
            if marker in user:
                i = self.outer.counter.get(marker, 0)
                self.outer.counter[marker] = i + 1
                return _Resp(script[i % len(script)])
        return _Resp('{"correctness": 0, "groundedness": 0}')
class FakeJudgeClient:
    def __init__(self):
        self.counter = {}
        self.chat = type("C", (), {"completions": _Completions(self)})()


# 每个标记对应「三次采样依次返回什么」
SCRIPTS = {
    "ANS_GOOD":    ['{"correctness": 2, "groundedness": 2}'] * 3,          # 稳定满分
    "ANS_PARTIAL": ['{"correctness": 2, "groundedness": 2}',               # 采样分歧
                    '{"correctness": 1, "groundedness": 2}',
                    '{"correctness": 1, "groundedness": 1}'],
    "ANS_BADPARSE":['这不是JSON', 'still not json', '抱歉我无法'],          # 全失败 → 跳过
    "ANS_MIXED":   ['坏的', '{"correctness": 2, "groundedness": 1}',        # 一坏两好 → 仍判成
                    '{"correctness": 2, "groundedness": 1}'],
}


def _mk(query, answer_marker, expected=""):
    """造一个带 1 个检索块的 RAGResult + 对应 EvalSample。"""
    ch = Chunk(id="x", text="检索上下文占位", source="t", start=0, end=0)
    r = RAGResult(query=query, retrieved=[Retrieved(ch, 1.0)],
                  answer=f"{answer_marker} 的答案正文")
    s = EvalSample(query=query, expected_answer=expected)
    return r, s


def main():
    pairs = [
        _mk("Q好",   "ANS_GOOD",    "供给型;需求型;环境型"),
        _mk("Q部分", "ANS_PARTIAL", "12"),
        _mk("Q全坏", "ANS_BADPARSE","LDA"),
        _mk("Q混合", "ANS_MIXED",   "43"),
    ]
    results = [r for r, _ in pairs]
    samples = [s for _, s in pairs]

    ev = LLMJudgeEvaluator(client=FakeJudgeClient(), n_samples=3)
    m = ev.evaluate(results, samples)
    print("=== evaluate 指标 ===")
    for k, v in m.items():
        print(f"  {k}: {v}")

    # ---- 断言聚合数学 ----
    # 判成 3 题(好/部分/混合)，全坏 1 题 → n_failed=1
    assert m["n_judged"] == 3.0, m
    assert m["n_failed"] == 1.0, m
    # correctness 每题均值: 好=2, 部分=(2+1+1)/3=1.333, 混合=(2+2)/2=2 → 全局均值=(2+1.333+2)/3=1.778 → /2=0.8889
    assert abs(m["JudgeCorrectness"] - 0.8889) < 1e-3, m["JudgeCorrectness"]
    # groundedness: 好=2, 部分=(2+2+1)/3=1.667, 混合=(1+1)/2=1 → (2+1.667+1)/3=1.556 → /2=0.7778
    assert abs(m["JudgeGroundedness"] - 0.7778) < 1e-3, m["JudgeGroundedness"]
    # Stability=1-平均(单题correctness方差): 好 var0, 部分 pvar([2,1,1])=0.2222, 混合 pvar([2,2])=0
    #   平均=(0+0.2222+0)/3=0.0741 → Stability=0.9259
    assert abs(m["JudgeStability"] - 0.9259) < 1e-3, m["JudgeStability"]
    print("\n[OK] evaluate 聚合 / 方差 / 失败计数 全部正确")

    # ---- calibrate：人工给「部分题=1、混合题=2」，judge 均值四舍五入后比对 ----
    ev2 = LLMJudgeEvaluator(client=FakeJudgeClient(), n_samples=3)
    human = {"Q部分": 1, "Q混合": 2, "Q好": 2}
    cal = ev2.calibrate(results, samples, human)
    print("\n=== calibrate ===")
    for k, v in cal.items():
        print(f"  {k}: {v}")
    # judge: 好 round(2)=2 vs 人工2 ✓; 部分 round(1.333)=1 vs 1 ✓; 混合 round(2)=2 vs 2 ✓
    assert cal["n_calibrated"] == 3.0, cal
    assert cal["MAE"] == 0.0 and cal["ExactMatch"] == 1.0, cal
    print("\n[OK] calibrate MAE/一致率 正确")
    print("\n全部通过 ✅")


if __name__ == "__main__":
    main()