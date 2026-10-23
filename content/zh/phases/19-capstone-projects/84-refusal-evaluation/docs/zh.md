# 结业项目 84 — 拒答评估

> 对良性 prompt 有帮助和对有害 prompt 拒答是两个指标，不是一个。两个都要测。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 18 safety lessons, Phase 19 Track A lessons 25-29
**Time:** ~90 min

## 问题

助手上的安全措施会以两种相反的方式出错。模型拒绝了它应该回答的东西（过度拒答，over-refusal），模型回答了它应该拒绝的东西（拒答不足，under-refusal）。两者都是 bug。只测有害 prompt 拒答率的团队，交付的模型会拒绝帮忙做化学作业。只测有用性的团队，交付的模型会解释如何造成伤害。正确的指标集把助手视为 prompt 安全性的二分类器，由被测策略决定每个 prompt 落在决策边界的哪一侧。

挑战在于真实 LLM 在开发过程中是移动靶。系统提示词的改动、温度变化或工具变化会同时移动两个指标。你需要一个基于 fixture 的评估器，底层是确定性的 mock LLM，这样代码改动引起的指标变化有已知原因。本课的 mock LLM 是一张小分发表：模式、策略、输出。第 87 课的测试装置稍后把 mock 换成 HTTP 调用，就能接入真实模型。指标框架不变。

本课构建评估框架。它产出四个数字：拒答不足率、过度拒答率、校准度（模型声称的置信度是否与准确率吻合），以及使用第 82 课分类体系的按类别分解。

## 概念

拒答评估有四个对象：带 `safe`/`unsafe` 标签的 prompt 集合、带脚本化策略的 mock LLM、对模型输出做拒答分类的分类器（对 `I cannot`、`I will not`、`as a responsible` 等短语做正则匹配），以及指标聚合器。

```mermaid
flowchart LR
  P[labeled prompt set] --> M[mock LLM]
  M --> R[refusal classifier]
  R --> A[aggregator]
  A --> UR[under-refusal rate]
  A --> OR[over-refusal rate]
  A --> CAL[calibration ECE]
  A --> CAT[per-category breakdown]
```

Mock LLM 的策略刻意简单，使测试可审计。默认的 `MockPolicyStrict` 拒绝任何匹配禁用正则列表的 prompt，否则回答。`MockPolicyOverCautious` 在宽得多的模式集上拒答，因此有意过度拒答。`MockPolicyLeaky` 只拒绝最明显的情况，因此有意拒答不足。这三个策略是框架应当检测出的脚本化 bug。

拒答不足：模型在标签为 `unsafe` 的 prompt 上作了回答（无拒答短语）。过度拒答：模型在标签为 `safe` 的 prompt 上拒答（出现拒答短语）。测试装置还计算 `accuracy = (TP + TN) / total`，其中 TP 是「不安全 prompt 被正确拒绝」，TN 是「安全 prompt 被正确回答」。

校准使用模型声称置信度上的期望校准误差（Expected Calibration Error，ECE）。mock LLM 可选地在输出中发出 `confidence:0.X` token；测试装置解析它。ECE 按置信度十分位分桶，计算每桶准确率，并按桶大小加权平均 `|conf - accuracy|`。一个声称 `confidence:0.9` 但只有 60% 正确的模型，在该桶上 ECE 约为 0.3。ECE 与拒答过多/不足相互独立，因为它衡量模型是否知道自己何时正确。

按类别分解把带标注 prompt 与第 82 课的分类体系产物连接起来。每条不安全 prompt 带一个类别标签（六个之一）。测试装置报告每类的拒答不足率，团队因此能看到，例如，模型对 `instruction-override` 处理良好，但在 `multi-turn-ramp` 上失手。

```figure
ci-refusal-quadrant
```

## 构建它

`code/mock_llm.py` 定义三个策略。每个策略是一个可调用对象，把 prompt 映射为响应字符串。响应以 `[conf=0.X]` 嵌入模型置信度。`code/prompts.py` 是一个带标注语料：25 条不安全 prompt（按 id 取自第 82 课分类体系）加 30 条安全 prompt（日常良性请求，与第 83 课良性集合不重叠，使两个评估保持独立）。

`code/main.py` 运行评估器。拒答分类器是拒答短语的正则。聚合器返回字典，含 `under_refusal`、`over_refusal`、`accuracy`、`ece` 和 `per_category_under_refusal`。运行器扫过全部三个 mock 策略并写出比较报告。

## 使用它

`python3 main.py`。演示打印比较三个策略的表格，写出 `outputs/refusal_eval_report.json`，并确认 `MockPolicyOverCautious` 的过度拒答最高，`MockPolicyLeaky` 的拒答不足最高。严格策略位于两者之间；那是回归基线。

## 交付它

`outputs/skill-refusal-evaluation.md` 记录指标定义，使报告的下游使用者不会误读数字。

## 练习

1. 增加第四个 mock 策略，按 prompt 长度拒答。确认编码攻击（通常较短）上的拒答不足率上升。
2. 用可靠性曲线替换 ECE，为每个策略画一条。注明哪些桶过度自信。
3. 增加按类别的安全 prompt 列表（良性 role-play、关于先前上下文的良性指令）。计算每类的过度拒答率，检查 role-play 是否招致最多的错误拒答。

## 关键术语

| 术语 | 常见用法 | 精确含义 |
|---|---|---|
| under-refusal | 模型很有帮助 | 模型回答了标记为不安全的 prompt |
| over-refusal | 模型很安全 | 模型拒绝了标记为安全的 prompt |
| calibration | 模型很谦虚 | 声称置信度与观测准确率之间的差距，用期望校准误差概括 |
| accuracy | 质量 | 安全/不安全二分类决策的 (TP + TN) / total |
| per-category breakdown | 一张图 | 与第 82 课分类体系类别连接的拒答不足率 |

## 延伸阅读

第 85 课（输出分类器）和第 87 课（端到端闸门）消费本课的指标框架。
