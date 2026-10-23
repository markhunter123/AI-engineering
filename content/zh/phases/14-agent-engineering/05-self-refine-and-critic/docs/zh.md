# Self-Refine 与 CRITIC：迭代式输出改进

> Self-Refine（Madaan 等，2023）让同一个 LLM 在一个循环中扮演三种角色——生成、反馈、精炼。在 7 项任务上平均提升 20 个绝对点。CRITIC（Gou 等，2023）把验证环节交给外部工具，从而强化了反馈步骤。2026 年，这一模式以「evaluator-optimizer」（Anthropic）或防护栏循环（OpenAI Agents SDK）的形式出现在每一个框架中。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~60 minutes

## 学习目标

- 说出 Self-Refine 的三个 prompt（生成、反馈、精炼），并解释为什么历史记录对精炼 prompt 很重要。
- 解释 CRITIC 的关键洞见：没有外部依据时，LLM 的自我验证并不可靠。
- 实现一个 stdlib Self-Refine 循环，带历史记录和可选的外部验证器。
- 把这个模式对应到 Anthropic 的「evaluator-optimizer」工作流和 OpenAI Agents SDK 的输出防护栏。

## 问题

智能体产出的答案几乎正确。也许某行代码有语法错误。也许摘要太长。也许计划漏掉了一个边界情况。你想要的是：智能体先批评自己的输出，然后修正它。

Self-Refine 证明，用单个模型、不需要训练数据、不需要强化学习也能做到。但有一个问题：LLM 在硬事实上不擅长自我验证。CRITIC 给出了解法——把验证步骤交给外部工具（搜索、代码解释器、计算器、测试运行器）。

这两篇论文共同定义了 2026 年迭代式改进的默认做法：生成、验证（尽可能用外部工具）、精炼，验证器通过时停止。

## 概念

### Self-Refine（Madaan 等，NeurIPS 2023）

一个 LLM，三个角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

关键细节：`refine` 能看到完整历史——所有先前的输出和批评——因此不会重复犯错。论文做了消融：去掉历史，质量会急剧下降。

核心结果：在 7 项任务（数学、代码、缩略词、对话）上平均提升 20 个绝对点，包括 GPT-4。无需训练、无需外部工具、单个模型。

### CRITIC（Gou 等，arXiv:2305.11738，v4 2024 年 2 月）

Self-Refine 的弱点：反馈环节是 LLM 给自己打分。对事实性陈述而言这并不可靠（幻觉在产出它的模型眼里往往很有说服力）。CRITIC 把 `feedback(task, output)` 换成 `verify(task, output, tools)`，其中 `tools` 包括：

- 用于事实性陈述的搜索引擎。
- 用于代码正确性的代码解释器。
- 用于算术的计算器。
- 领域特定的验证器（单元测试、类型检查器、linter）。

验证器产出以工具结果为依据的结构化批评。精炼器再以这条批评为条件。

核心结果：在事实性任务上 CRITIC 优于 Self-Refine，因为批评有外部依据。在没有外部验证器的任务上（创意写作、格式化），CRITIC 退化为 Self-Refine。

### 停止条件

两种常见形态：

1. **验证器通过。** 外部测试返回成功。有这种条件时优先采用（单元测试、类型检查器、防护栏断言）。
2. **没有给出反馈。** 模型说「输出没问题」。更便宜但不可靠；要配上最大迭代次数上限。

2026 年的默认做法：把它们组合起来。「验证器通过时停止，或者模型说没问题且迭代次数 >= 2，或者迭代次数 >= max_iterations 时停止。」

### Evaluator-Optimizer（Anthropic，2024）

Anthropic 2024 年 12 月的文章把它列为五种工作流模式之一。两个角色：

- Evaluator：给输出打分并给出批评。
- Optimizer：根据批评修订输出。

循环直到评估器通过。这就是 Anthropic 框架下的 Self-Refine/CRITIC。Anthropic 补充了一个关键工程细节：评估器和优化器的 prompt 应当有实质差异，以免模型只是走过场盖章。

### OpenAI Agents SDK 的输出防护栏

OpenAI Agents SDK 以「输出防护栏」的形式提供这个模式。防护栏是一个在智能体最终输出上运行的校验器。如果防护栏触发（抛出 `OutputGuardrailTripwireTriggered`），输出被拒绝，智能体可以重试。防护栏可以调用工具（CRITIC 风格），也可以是纯函数（Self-Refine 风格）。

### 2026 年的陷阱

- **盖章循环。** 同一个模型用同样的 prompt 风格做生成和批评，最终只会得出「我看挺好」。要使用结构上不同的 prompt，或者让更小更便宜的模型来做批评。
- **过度精炼。** 每一轮精炼都增加延迟和 token。预算 1–3 轮；再多就升级到人工审核。
- **在琐碎任务上用 CRITIC。** 如果没有外部验证器，CRITIC 会退化为 Self-Refine；不要为一个空壳验证器付出延迟代价。

```figure
self-refine
```

## 构建它

`code/main.py` 在一个玩具任务上实现 Self-Refine 和 CRITIC：给定一个主题，产出一个简短的要点列表。验证器检查格式（3 个要点，每个不超过 60 个字符）。CRITIC 额外加入一个外部「事实验证器」，对已知的幻觉扣分。

组件：

- `generate`——脚本化的产出器。
- `feedback`——LLM 式自我批评。
- `verify_external`——CRITIC 风格的有依据验证器。
- `refine`——根据历史重写输出。
- 停止条件——验证器通过，或最多 4 次迭代。

运行它：

```
python3 code/main.py
```

对比 Self-Refine 与 CRITIC 两次运行。CRITIC 抓住了 Self-Refine 漏掉的一个事实错误，因为外部验证器拥有自我批评所不具备的依据。

## 使用它

Anthropic 的 evaluator-optimizer 就是这个模式在 Claude 语境下的说法。OpenAI Agents SDK 的输出防护栏是 CRITIC 形态的（防护栏可以调用工具）。LangGraph 提供的反思节点读起来就像 Self-Refine。Google 的 Gemini 2.5 Computer Use 增加了一个逐步的安全评估器，它是 CRITIC 的变体：每个动作在提交前都经过验证。

## 交付它

`outputs/skill-refine-loop.md` 根据任务形态、验证器可用性和迭代预算来配置一个 evaluator-optimizer 循环。它会输出生成器、评估器 / 验证器和优化器的 prompt，以及一套停止策略。

## 练习

1. 用 max_iterations=1 运行玩具示例。CRITIC 还有帮助吗？
2. 把外部验证器换成一个有噪声的版本（随机 30% 误报）。循环会怎样？这正是 2026 年大多数防护栏技术栈的现实。
3. 实现一个「不同模型做生成与批评」的变体：大模型生成，小模型批评。它比同模型更好吗？
4. 阅读 CRITIC 第 3 节（arXiv:2305.11738 v4）。说出三类验证工具，并各举一个例子。
5. 把 OpenAI Agents SDK 的 `output_guardrails` 对应到 CRITIC 的验证器角色。这个 SDK 哪里做错了，哪里做对了？

## 关键术语

| 术语 | 人们常怎么说 | 实际含义 |
|------|----------------|------------------------|
| Self-Refine | 「自我修复的 LLM」 | 在单个模型中循环执行生成 -> 反馈 -> 精炼，并保留历史 |
| CRITIC | 「工具锚定的验证」 | 用外部验证器（搜索、代码、计算器、测试）取代反馈 |
| Evaluator-Optimizer | 「Anthropic 工作流模式」 | 两个角色——评估器打分、优化器修订——循环至收敛 |
| Output guardrail | 「事后检查」 | OpenAI Agents SDK 的校验器，在智能体产出输出之后运行 |
| Verify step | 「批评阶段」 | 承重的决策：有外部依据，还是自评 |
| Refine history | 「模型已经试过什么」 | 先前的输出 + 批评，前置到精炼 prompt；去掉它质量会崩塌 |
| Rubber-stamp loop | 「自我认同失败」 | 同一 prompt 的批评返回「看着不错」；用结构不同的 prompt 修复 |
| Stop condition | 「收敛测试」 | 验证器通过，或（无反馈且达到迭代上限）；绝不要单条件 |

## 进一步阅读

- [Madaan 等，Self-Refine（arXiv:2303.17651）](https://arxiv.org/abs/2303.17651)——经典论文
- [Gou 等，CRITIC（arXiv:2305.11738）](https://arxiv.org/abs/2305.11738)——工具锚定的验证
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)——evaluator-optimizer 工作流模式
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/)——作为 CRITIC 形态验证器的输出防护栏
