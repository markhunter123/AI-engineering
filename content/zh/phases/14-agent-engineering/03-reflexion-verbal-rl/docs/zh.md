# Reflexion：语言化强化学习

> 基于梯度的强化学习要修复一种失败模式，需要上千次试验和 GPU 集群。Reflexion（Shinn 等，NeurIPS 2023）用自然语言做到同样的事：每次试验失败后，智能体写下一段反思，存入情景记忆，并让下一次试验以这段记忆为条件。Letta 的 sleep-time compute、Claude Code 的 CLAUDE.md 经验沉淀，以及 pro-workflow 的 learn-rule，背后的模式都是它。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 02 (ReWOO)
**Time:** ~60 minutes

## 学习目标

- 说出 Reflexion 的三个组件（Actor、Evaluator、Self-Reflector）以及情景记忆的作用。
- 实现一个 stdlib Reflexion 循环，包含二值评估器、反思缓冲区和全新的重试。
- 为给定任务在标量、启发式和自评估三类反馈来源之间做选择。
- 解释为什么语言化强化学习能抓住那些基于梯度的强化学习需要上千次试验才能修复的错误。

## 问题

一个智能体在某项任务上失败了。在标准强化学习中，你会再跑上千次试验、计算梯度、更新权重。这既昂贵又缓慢，而且大多数生产智能体并没有为每一次失败准备训练预算。

Reflexion（Shinn 等，arXiv:2303.11366）提出了另一个问题：如果智能体只是想清楚自己为什么失败，再带着这个想法重试呢？不更新权重。不算梯度。只是把自然语言存在两次试验之间。

结果是：在 ALFWorld 上它胜过 ReAct 和其他未微调的基线。在 HotpotQA 上它优于 ReAct。在代码生成（HumanEval/MBPP）上，它创下了当时的最佳成绩。而这一切没有走一步梯度。

## 概念

### 三个组件

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

外加一个数据结构：

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

一次试验运行 Actor。Evaluator 给它打分。如果分数低，Self-Reflector 生成一段反思（「我选错了工具，因为我把问题误读成在问 X，而它其实在问 Y」）。这段反思进入情景记忆。下一次试验重新开始，但会看到这段反思。

### 三类评估器

1. **标量（Scalar）**——外部二值信号。ALFWorld 成功或失败。HumanEval 的测试通过或不通过。最简单，信号最强。
2. **启发式（Heuristic）**——预定义的失败特征。「如果智能体连续两次产生同一个动作，标记为卡住。」「如果轨迹超过 50 步，标记为低效。」
3. **自评估（Self-evaluated）**——由 LLM 给自己的轨迹打分。在没有标准答案时需要它。信号较弱；适合与工具锚定的验证搭配使用（第 05 课——CRITIC）。

2026 年的默认做法是混合使用：有标量信号时用标量，没有时用自评估，启发式规则当作安全护栏。

### 为什么它能推广

与其说 Reflexion 是一种新算法，不如说它是一个被命名的模式。几乎所有生产环境中的「自愈」智能体都在跑它的某种变体：

- Letta 的 sleep-time compute（第 08 课）：由一个独立的智能体回顾过去的对话，并写入记忆块。
- Claude Code 的 `CLAUDE.md` /「保存记忆」模式：把反思记录为经验，在未来的会话前置。
- pro-workflow 的 `/learn-rule` 命令：把修正记录为显式规则。
- LangGraph 的反思节点：一个给输出打分、必要时路由到精炼环节的节点。

它们都源自同一个洞见：自然语言是一种足够丰富的媒介，可以在多次运行之间携带「我从失败中学到了什么」。

### 何时有效，何时无效

Reflexion 在以下情况有效：

- 存在清晰的失败信号（测试失败、工具报错、答案错误）。
- 任务类别可复现（同一类问题可以再问一次）。
- 反思有空间改进轨迹（动作预算充足）。

Reflexion 在以下情况无效：

- 智能体第一次尝试就成功。
- 失败是外部原因（网络断了、工具坏了）——对「当时网络断了」的反思对未来运行没有帮助。
- 反思变成了迷信——把一次偶发的不稳定运行当成规律存下来。

2026 年的陷阱：记忆腐化。反思不断累积，其中有些已经过时或错误；情景缓冲区越大，重跑越慢。缓解办法：定期压缩（第 06 课）、给反思设 TTL，或用独立的 sleep-time 清理智能体（Letta）。

```figure
react-trace
```

## 构建它

`code/main.py` 在一个玩具谜题上实现 Reflexion：生成一个和为目标的 3 元素列表。Actor 输出候选列表；Evaluator 检查总和；Self-Reflector 写下一句说明哪里出了问题。这段反思进入情景记忆，供下一次试验使用。

组件：

- `Actor`——一个在看到反思后会有所改进的脚本化策略。
- `Evaluator.binary()`——按目标和判定通过/失败。
- `SelfReflector`——生成一行失败诊断。
- `EpisodicMemory`——带 TTL 语义的有界列表。

运行它：

```
python3 code/main.py
```

轨迹展示三次试验。试验 1 失败，存入一段反思；试验 2 看到反思后有所改进但仍失败；试验 3 成功。与基线运行（没有反思）相比——基线会一直卡在试验 1 的答案上。

## 使用它

LangGraph 把反思作为一个节点模式提供。Claude Code 的 `/memory` 命令和 pro-workflow 的 `/learn-rule` 把情景缓冲区外化成 markdown 文件。Letta 的 sleep-time compute 在空闲时间运行 Self-Reflector，让主智能体保持对延迟敏感。OpenAI Agents SDK 没有直接提供 Reflexion；你需要用一个按分数拒绝轨迹的自定义 Guardrail，和一个跨运行存活的记忆 `Session` 来自己搭建。

## 交付它

`outputs/skill-reflexion-buffer.md` 创建并维护一个情景缓冲区，具备反思捕获、TTL 和去重能力。给定任务类别和一次失败，它会输出一段真正能帮助下一次试验的反思（而不是泛泛的「要更小心」）。

## 练习

1. 从二值评估器换成返回距离度量的标量评估器（离目标有多远）。它收敛得更快吗？
2. 给反思加上 10 次试验的 TTL。过了这个点之后，更早的反思是有害还是有益？
3. 实现启发式评估器：如果同一动作重复出现，就把该次试验标记为卡住。它与 Self-Reflector 如何相互作用？
4. 用一个无视反思的对抗性 Actor 来跑 Reflexion。最少需要什么样的反思 prompt 工程，才能迫使 Actor 注意到它们？
5. 阅读 Reflexion 论文中关于 AlfWorld 的第 4 节。在概念上复现 130% 的成功率提升：相对原生 ReAct 的关键差异是什么？

## 关键术语

| 术语 | 人们常怎么说 | 实际含义 |
|------|----------------|------------------------|
| Reflexion | 「自我修正」 | Shinn 等 2023——Actor、Evaluator、Self-Reflector 加上情景记忆 |
| Verbal reinforcement | 「无梯度学习」 | 把自然语言反思前置到下一次试验的 prompt 中 |
| Episodic memory | 「按任务的反思」 | 面向同一任务类别的、有界的既往反思缓冲区 |
| Scalar evaluator | 「二值成功信号」 | 来自标准答案的通过/失败或数值分数 |
| Heuristic evaluator | 「基于模式的检测器」 | 预定义的失败特征（如卡死循环、步数过多） |
| Self-evaluator | 「用自己的轨迹做 LLM 评审」 | 没有标准答案时的低信号兜底——与工具锚定的验证搭配 |
| Memory rot | 「过时的反思」 | 情景缓冲区塞满废弃条目；用压缩 / TTL 修复 |
| Sleep-time reflection | 「异步自我反思」 | 把 Self-Reflector 移出热路径运行，让主智能体保持快速 |

## 进一步阅读

- [Shinn 等，Reflexion: Language Agents with Verbal Reinforcement Learning（arXiv:2303.11366）](https://arxiv.org/abs/2303.11366)——经典论文
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute)——生产中的异步反思
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)——把情景缓冲区作为上下文的一部分来管理
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)——反思节点模式
