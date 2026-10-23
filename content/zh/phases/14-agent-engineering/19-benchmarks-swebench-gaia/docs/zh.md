# 基准测试：SWE-bench、GAIA、AgentBench

> 2026 年，有三个基准测试为 Agent 评估提供了锚点。SWE-bench 测试代码补丁能力。GAIA 测试通用工具使用能力。AgentBench 测试多环境推理能力。你需要了解它们的构成、它们的污染故事，以及它们没有衡量的东西。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 06 (Tool Use)
**Time:** ~60 minutes

## 学习目标

- 说出 SWE-bench 的测试框架（FAIL_TO_PASS），并解释为什么它以单元测试作为门槛。
- 解释 SWE-bench Verified（OpenAI，500 个任务）为何存在，以及它移除了什么。
- 描述 GAIA 的设计：对人类简单，对 AI 困难；三个难度等级。
- 说出 AgentBench 的八个环境，以及它对开源 LLM 的主要阻碍。
- 总结 SWE-bench+ 的污染发现及其影响。

## 问题

排行榜告诉你哪个模型在某个基准上获胜。它们不会告诉你：

- 基准是否被污染（训练数据中包含解法、测试泄漏）。
- 基准是否衡量了你关心的内容（代码、浏览还是通用能力）。
- 评估器是否稳健（AST 匹配、状态检查、人工审核）。

在引用任何数字之前，先了解这三个锚点基准及其失效模式。

## 概念

### SWE-bench（Jimenez 等，ICLR 2024 oral）

- 来自 12 个热门 Python 仓库的 2,294 个真实 GitHub issue。
- Agent 收到：修复前提交的代码库 + 自然语言 issue 描述。
- Agent 产出：一个补丁。
- 评估器：应用补丁，运行仓库的测试套件。补丁必须让 FAIL_TO_PASS 测试翻转（原先失败、现在通过），同时不破坏 PASS_TO_PASS 测试。

SWE-agent（Yang 等，2024）在发布时达到 12.5%，靠的是强调 Agent-计算机接口（文件编辑器命令、模型能理解的搜索语法）。

### SWE-bench Verified

OpenAI，2024 年 8 月。人工策展的 500 个任务子集。移除了有歧义的 issue、不可靠的测试，以及修复方案不明确的任务。这是回答「你的 Agent 能否交付真实补丁？」的主要基准。

### 污染

- 超过 94% 的 SWE-bench issue 早于大多数模型的截止时间。
- **SWE-bench+** 发现 32.67% 的成功补丁在 issue 文本中泄漏了解法（模型在描述里看到了修复方案），另有 31.08% 因测试覆盖薄弱而可疑。
- Verified 更干净，但并非没有污染。

实际含义：一个在 SWE-bench 上得 50% 的模型，在 SWE-bench+ 上可能只有 35%。如果你声称有 SWE-bench 性能，请始终同时报告两者。

### GAIA（Mialon 等，2023 年 11 月）

- 466 个问题；其中 300 个保留给 huggingface.co/gaia-benchmark 上的私榜。
- 设计理念：「对人类概念上简单（92%），但对 AI 困难（GPT-4 加插件：15%）。」
- 测试推理、多模态、网页、工具使用。
- 三个难度等级；Level 3 需要跨模态的长工具链。

GAIA 是你衡量「通用能力」时要跑的基准。不要与代码专用基准混淆。

### AgentBench（Liu 等，ICLR 2024）

- 8 个环境，覆盖代码（Bash、DB、KG）、游戏（Alfworld、LTP）、网页（WebShop、Mind2Web）和开放式生成。
- 多轮，每个 split 约 4k–13k 轮。
- 主要发现：长期推理、决策和指令遵循是开源 LLM 追赶商业模型的主要阻碍。

### 这些基准没有衡量的东西

- 真实世界的运营成本（token、实际耗时）。
- 对抗条件下的安全行为。
- 在你所在领域的表现（使用你自己的评估，第 30 课）。
- 尾部失败（基准取平均值；生产运维关心最差的 1%）。

### 基准测试常见的错误

- **单一数字执念。** SWE-bench 50% 告诉你的信息，少于 P50/P75/P95 成本 + 步数分布。
- **污染下的声明。** 报告 SWE-bench 却不提 Verified 或 SWE-bench+，具有误导性。
- **把基准当开发目标。** 为基准优化会偏离生产实用性。

```figure
ae-swebench-gate
```

## 构建它

`code/main.py` 实现一个玩具式的类 SWE-bench 框架：

- 合成的 bug 修复任务（3 个）。
- 一个脚本化的「Agent」提出补丁。
- 一个测试运行器，检查 FAIL_TO_PASS（bug 现已修复）和 PASS_TO_PASS（没有破坏其他东西）。
- 一个 GAIA 风格的难度分类器，基于问题分解深度。

运行它：

```
python3 code/main.py
```

输出显示每个任务、每个难度的解决率，并让评估器规则变得具体。

## 使用它

- **SWE-bench Verified** 用于代码 Agent。始终报告 Verified 分数。
- **GAIA** 用于通用 Agent。使用私榜 split。
- **AgentBench** 用于多环境比较。
- **自定义评估**（第 30 课）用于你产品的实际形态。

## 交付它

`outputs/skill-benchmark-harness.md` 为任意代码库-任务对构建一个 SWE-bench 风格的框架，带 FAIL_TO_PASS / PASS_TO_PASS 门槛。

## 练习

1. 把这个玩具框架移植到真实仓库上（选你自己的一个）。为已知 bug 编写 3 个 FAIL_TO_PASS 测试。
2. 增加一个步数指标。在你的 3 个任务上，每次解决平均需要多少 Agent 步骤？
3. 阅读 SWE-bench+ 论文。实现一个解法泄漏检查（把 issue 文本与 diff 做模式匹配）。
4. 从公开 split 下载一个 GAIA 问题。追踪一个 GPT-4 级 Agent 会怎么做。它需要哪些工具？
5. 阅读 AgentBench 的分环境拆解。哪个环境最像你的产品界面？那里的「SOTA」是什么样？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| SWE-bench | 「代码 Agent 基准」 | 2,294 个 GitHub issue；补丁必须翻转 FAIL_TO_PASS 测试 |
| SWE-bench Verified | 「干净的 SWE-bench」 | 500 个人工策展任务，OpenAI |
| FAIL_TO_PASS | 「修复门槛」 | 原先失败、打补丁后必须通过的测试 |
| PASS_TO_PASS | 「无回归门槛」 | 原先通过、现在必须仍然通过的测试 |
| GAIA | 「通用基准」 | 466 个人类简单、AI 困难的多工具问题 |
| AgentBench | 「多环境基准」 | 8 个环境；长程多轮 |
| 污染 | 「训练集泄漏」 | 基准任务出现在模型训练数据中 |
| SWE-bench+ | 「污染审计」 | 在成功的 SWE-bench 补丁中发现 32.67% 的解法泄漏 |

## 延伸阅读

- [Jimenez 等，SWE-bench（arXiv:2310.06770）](https://arxiv.org/abs/2310.06770) — 原始基准
- [OpenAI，SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 策展子集
- [Mialon 等，GAIA（arXiv:2311.12983）](https://arxiv.org/abs/2311.12983) — 通用基准
- [Liu 等，AgentBench（arXiv:2308.03688）](https://arxiv.org/abs/2308.03688) — 多环境套件
