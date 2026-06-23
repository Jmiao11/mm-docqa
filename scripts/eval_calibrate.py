"""人工校准 LLM-judge：输出 judge 与人工 correctness 的一致度(MAE / ExactMatch)。

  python scripts/eval_calibrate.py

为什么需要：没校准过的 judge 分不可信、不许对外引用。本脚本对每道有人工分的题
真跑一遍 pipeline.run → judge 判 n_samples 次取整，与你的人工分比对，把"可信"做成
可报告的数字。题目从 evaluators/golden.jsonl 读(单一真相源)。

人工分由你独立判定(0=错 / 1=部分 / 2=对)，填在下方 HUMAN。这是简历那句的底气，
不能照着 judge 反推；对照你上一轮 eval_answer 打印出的每题答案 vs expected_answer 自行判。

成本：对 HUMAN 里每题真调 DeepSeek 生成 + n_samples 次 kimi 判，慢且会烧 API。
"""
import sys

from core.config import RAGConfig, build_pipeline
from core.interfaces import RAGResult
from evaluators.golden import load_golden
from evaluators.judge import LLMJudgeEvaluator

# 人工 correctness 分(0/1/2)。11 题预填我的明显读数(答案与 expected 完全吻合 → 2)，
# 请逐一对照上一轮答案核实；Q6 是唯一需要你真正判定的题(见下方注释)。
# 不想纳入校准的题删掉该行即可。
HUMAN: dict[str, int] = {
    "本研究将政策工具分为哪三类": 2,
    "最终确定的主题数是多少": 2,
    "困惑度曲线说明了什么趋势": 2,
    "本文采用了什么研究方法": 2,
    "本文分析了多少份政策文本": 2,
    # Q6：答案给的是"SNA 证实 LDA 可信度"——论文里真实存在的另一个结论，但不是
    # expected 的 节点/贡献度/治理 三点。你判：0=答非所问 / 1=部分相关。按你自己的标准定。
    "社会网络分析得出了什么结论": 1,   # ← 必填：0 或 1
    "本文分析的政策文本覆盖全国多少个省份": 2,
    "本文用哪两种方法综合确定最佳主题数": 2,
    "本文借助什么软件构建主题特征词的社会网络": 2,
    "本文分析的政策文本现行有效的时间截至何时": 2,
    "针对政策工具失衡本文建议倡导哪种治理理念": 2,
    "本文发现我国数字经济政策过度依赖哪一类政策工具": 2,
}


def main() -> None:
    missing = [q for q, v in HUMAN.items() if v is None]
    if missing:
        print("以下题的人工分还没填(值为 None)，请按你的判断填 0/1/2 后再跑：",
              file=sys.stderr)
        for q in missing:
            print(f"  - {q}", file=sys.stderr)
        sys.exit(1)

    samples = load_golden()
    cfg = RAGConfig(retriever="rerank", chunker="semantic",
                    generator="llm", collection_name="docqa_v2")
    pipeline = build_pipeline(cfg)

    todo = [s for s in samples if s.query in HUMAN]
    print(f"校准 | {len(todo)} 题 | 配置: {cfg.retriever}+{cfg.chunker}+{cfg.generator}\n",
          file=sys.stderr)

    results: list[RAGResult] = []
    for s in todo:
        r = pipeline.run(s.query, k=4)
        results.append(r)

    judge = LLMJudgeEvaluator(n_samples=3)
    report = judge.calibrate(results, todo, HUMAN)

    print("\n=== judge 人工校准 ===")
    print(f"  MAE:          {report['MAE']}      (平均绝对误差，越小越好)")
    print(f"  ExactMatch:   {report['ExactMatch']}   (judge 与人工完全一致的比例)")
    print(f"  n_calibrated: {report['n_calibrated']}")
    print("  注：报告了一致度后，judge 分方可对外引用(如简历)。")

    # 产出物解读：MAE 与 ExactMatch
    # 跑完后，judge.calibrate() 会返回两个核心指标：
    #
    # MAE (Mean Absolute Error, 平均绝对误差)：如果你判 2 分（全对），机器判 1 分（部分对），绝对误差就是 1。MAE 是所有题误差的平均值。越接近 0，说明机器裁判的尺度跟你越像。
    #
    # ExactMatch (完全一致率)：机器裁判给出的分数（取整后），跟你给的分数一模一样的题的占比。
    #
    # 总结：这个脚本是一个强有力的“信任锚点”。它把主观的“我觉得这个模型答得还行”，转化成了客观的、可复现的误差数值（MAE）。这就是工程化思维的直接体现。


if __name__ == "__main__":
    main()