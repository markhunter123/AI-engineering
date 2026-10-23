# 自主编码智能体版图（2026）

> SWE-bench Verified 在不到三年里从 4% 升到 80.9%。同一个 Claude Sonnet 4.5 在 SWE-agent v1 上得分 43.2%，在 Cline autonomous 上得分 59.8%——如今模型周围的脚手架与模型本身同样重要。OpenHands（原 OpenDevin）是活跃度最高的 MIT 许可平台，它的 CodeAct 循环直接在沙盒中执行 Python 动作，而不是发出 JSON 工具调用。这些头条数字掩盖了一个方法论问题：SWE-bench Verified 的 500 个任务中有 161 个只需要 1–2 行改动，而对同样的前沿模型，SWE-bench Pro（10 行以上任务）的得分只有 23%–59%。

**Type:** Learn
**Languages:** Python (stdlib, CodeAct vs JSON tool-call comparison)
**Prerequisites:** Phase 14 · 07 (Tool use), Phase 15 · 01 (Long-horizon agents)
**Time:** ~45 minutes

## 问题

「哪个编码智能体最好」是个错误的问题。正确的问题是：在与我的工作相匹配的任务分布上，用我将在生产环境中运行的脚手架，我能得到怎样的端到端可靠性？

2022 年到 2026 年间，这个领域认识到脚手架——检索层、规划器、沙盒、编辑-验证循环、反馈格式——是承重结构。Claude Sonnet 4.5 在 SWE-agent v1 上于 SWE-bench Verified 得分 43.2%；同一模型在 Cline 的自主脚手架内得分 59.8%。同样的权重，16.6 个百分点的绝对差距。基础模型只是一个组件，循环才是产品。

另一个问题是基准饱和会掩盖退步。SWE-bench Verified 已接近饱和，简单任务的长尾（500 个任务中有 161 个只需 ≤2 行改动）抬高了最高分。真实世界质量更适合在 SWE-bench Pro（10 行以上改动）这样的分布上衡量——同样的领先者在那里仍然只有 23%–59%。

## 概念

### SWE-bench，一段话说清

SWE-bench（Jimenez 等）取真实的 GitHub issue 及其标准答案补丁，要求智能体产出一个能让测试套件通过的补丁。SWE-bench Verified（OpenAI，2024）是人工筛选的 500 任务子集，去掉了含糊和有问题的任务。SWE-bench Pro 是更难的继任者——任务需要 10 行以上改动，当前前沿智能体的得分在 23%–59% 之间。

### 2022 → 2026 曲线到底说明了什么

- **2022 年**：研究模型在原始 SWE-bench 上约 4%。
- **2024 年**：GPT-4 + Devin 式脚手架约 14%；SWE-agent 约 12%。
- **2025 年**：Aider 和 SWE-agent 中的 Claude 3.5/3.7 Sonnet 推进到 40%–55% 区间。
- **2026 年**：Claude Sonnet 4.5 与前沿竞争者们在 SWE-bench Verified 上达到 70%–80% 以上。Epoch AI 的排行榜实时追踪这一数据。

这条陡峭曲线来自三个复合来源：更好的基础模型、更好的脚手架（CodeAct、反思、验证器循环），以及更好的基准（Verified 去除了噪声）。

### CodeAct 与 JSON 工具调用

OpenHands（All-Hands-AI，arXiv:2407.16741，原 OpenDevin）做了一个具体的架构押注：模型不发出由宿主解码并执行的 JSON 工具调用，而是发出 Python 代码，由 Jupyter 风格的内核在沙盒中运行。智能体可以在单个动作内遍历文件、串联工具并捕获自己的异常。

权衡如下：

- **JSON 工具调用**：每个动作是一轮；易于审计；组合性有限；默认安全，因为每次调用都经过显式验证器。
- **CodeAct**：一个动作可以是一整个程序；具备组合性；需要强化的沙盒（OpenHands 使用 Docker 隔离）；失效模式包括沙盒运行时允许的任何行为。

两种架构都已投产。CodeAct 在开放平台（OpenHands、smolagents）中占主导。JSON 工具调用在托管服务（Anthropic Managed Agents、OpenAI Assistants）中仍然占主导，因为提供商控制着执行器。

### 2026 年版图中的脚手架

| 脚手架 | 许可证 | 执行模型 | 显著特性 |
|---|---|---|---|
| OpenHands（OpenDevin） | MIT | Docker 中的 CodeAct | 最活跃的开放平台；事件流可重放 |
| SWE-agent | MIT | 智能体-计算机接口（ACI） | 首个端到端 SWE-bench 脚手架 |
| Aider | Apache-2 | 在本地仓库中通过 diff 编辑 | 极简脚手架，回归稳定性强 |
| Cline | Apache-2 | 带工具策略的 VS Code 智能体 | Sonnet 4.5 上得分最高的开放脚手架 |
| Devin（Cognition） | 专有 | 托管 VM + 规划器 | 首个「AI 软件工程师」产品类别 |
| Claude Code | 专有 | 权限模式 + routines | 第 10 课详细讲解智能体循环 |

### 为什么脚手架占主导

一次编码运行是一条长时程轨迹（第 1 课）。可靠性会在步骤之间复利。脚手架在三个地方换来分数：

1. **检索**：找到该读的文件是隐性瓶颈。SWE-agent 的 ACI、OpenHands 的文件索引、Aider 的仓库地图都在解决这个问题。
2. **验证器循环**：运行测试、读取堆栈跟踪并重试，在 SWE-bench 上能带来 10 分以上的提升。
3. **故障遏制**：出错时回滚的沙盒能防止损害累积。同一个模型，有无验证器循环，看起来就像两个不同的产品。

### 基准饱和与真实分布

OpenHands 的作者们和 Epoch AI 都指出，SWE-bench Verified 有一条简单任务长尾：500 个任务中有 161 个只需 1–2 行改动。高分部分是由这条长尾带动的。SWE-bench Pro 限定为 10 行以上改动，即使是前沿系统，得分也只在 23%–59% 区间。你的生产分布几乎肯定更接近 Pro 而非 Verified。

对选择智能体的含义是：用你自己的 bug 积压构建一个类似 Pro 的子集来测试。真正重要的分数，是在代表你实际交付内容的任务上的分数。

```figure
a5-scaffold-delta
```

## 使用它

`code/main.py` 在固定的迷你任务分布上比较两种玩具智能体脚手架：

1. 一种 **JSON 工具调用**脚手架，每轮执行一个动作。
2. 一种 **CodeAct** 脚手架，每个动作可以发出一小段 Python 代码。

两者都使用桩「模型」（确定性规则），使比较把脚手架与模型质量隔离开。输出显示 CodeAct 脚手架用更少的轮次解决更多任务，代价是每个动作的影响半径更大。

## 交付它

`outputs/skill-scaffold-audit.md` 帮助你在采用之前审计一个候选编码智能体脚手架：检索质量、验证器是否存在、沙盒隔离，以及基准与任务分布的匹配度。

## 练习

1. 运行 `code/main.py`。在同一个任务集上，每种脚手架各需要多少轮？各自每个动作的影响半径多大？

2. 阅读 OpenHands 论文（arXiv:2407.16741）。论文主张 CodeAct 在复杂任务上胜过 JSON 工具调用。找出论文承认的一种失效模式，并用一句话说明该模式在生产中何时会占主导。

3. 从你的 bug 积压中挑一个需要在两个文件里改动 10 行以上的任务。估计前沿模型在 (a) JSON 工具调用和 (b) CodeAct 下的端到端成功概率，并论证两者的差距。

4. SWE-bench Verified 有 161 个单文件、1–2 行的任务。构造一个排除这些任务的分数。排行榜会怎样洗牌？

5. 阅读 OpenAI 的「Introducing SWE-bench Verified」。解释用于去除含糊任务的具体方法，并指出这种筛选会漏掉的一个类别。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| SWE-bench | 「编码基准」 | 带标准答案补丁和测试套件的真实 GitHub issue |
| SWE-bench Verified | 「清洗过的子集」 | 500 个人工筛选任务，仍含简单任务长尾 |
| SWE-bench Pro | 「更难的子集」 | 10 行以上改动；前沿得分在 23%–59% |
| CodeAct | 「代码即动作」 | 智能体发出 Python；Jupyter 风格内核在沙盒中执行 |
| JSON 工具调用 | 「函数调用」 | 每个动作是结构化 JSON 载荷，执行前经过验证 |
| 脚手架 | 「智能体框架」 | 围绕基础模型的检索 + 规划器 + 执行器 + 验证器循环 |
| ACI（智能体-计算机接口） | 「SWE-agent 的格式」 | 为 LLM 使用习惯设计的命令集，而非为人类 shell 设计 |
| 验证器循环 | 「测试并重试」 | 运行测试、读取输出、修订补丁；非模型因素中最大的可靠性提升 |

## 延伸阅读

- [Jimenez 等 —— SWE-bench](https://www.swebench.com/) —— 原始基准与方法论。
- [OpenAI —— Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) —— 筛选子集是如何构建的。
- [Wang 等 —— OpenHands：一个面向 AI 软件开发者的开放平台](https://arxiv.org/abs/2407.16741) —— CodeAct 架构与事件流设计。
- [Epoch AI —— SWE-bench 排行榜](https://epoch.ai/benchmarks) —— 实时追踪的分数。
- [Anthropic —— 测量智能体自主性](https://www.anthropic.com/research/measuring-agent-autonomy) —— 长时程编码智能体可靠性的论述框架。
