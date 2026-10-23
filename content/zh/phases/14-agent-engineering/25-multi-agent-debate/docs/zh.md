# 多智能体辩论与协作

> Du 等人（ICML 2024，「Society of Minds」）让 N 个模型实例独立给出答案，随后在 R 轮中相互迭代批评直至收敛。该方法提升了事实准确性、规则遵循与推理能力。在 token 成本上，稀疏拓扑优于全连接网状拓扑。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 05 (Self-Refine and CRITIC)
**Time:** ~60 minutes

## 学习目标

- 解释辩论协议：N 个提议者、R 轮、收敛到共享答案。
- 说明辩论为何能提升事实准确性、规则遵循与推理能力。
- 解释稀疏拓扑：并非每位辩手都需要看到其他所有辩手。
- 在脚本化 LLM 上用 stdlib 实现辩论，包含全连接与稀疏两种变体；测量 token 成本与准确率。

## 问题

Self-Refine（第 05 课）是一个模型批评自己——有陷入群体思维（groupthink）的风险。CRITIC（第 05 课）把批评建立在外部工具上——但这些工具并非总是可用。辩论引入了第三种模式：多个实例、交叉批评、通过分歧达成收敛。

## 概念

### Society of Minds（Du 等，ICML 2024）

- N 个模型实例独立地回答同一个问题。
- 在 R 轮中，每个模型阅读其他人的提议并加以批评。
- 模型根据批评更新自己的答案。
- R 轮结束后，返回收敛后的答案。

原始实验出于成本考虑采用 N=3、R=2。在困难问题（MMLU、GSM8K、国际象棋走法合法性、传记生成）上，智能体更多、轮数更多会带来更高准确率。

跨模型组合优于单模型辩论：ChatGPT + Bard 联手 > 两者各自单独表现。

### 稀疏拓扑

「Improving Multi-Agent Debate with Sparse Communication Topology」（arXiv:2406.11776，2024-2025）表明全连接辩论并不总是最优。稀疏拓扑（星形、环形、中心辐射式）能以更低的 token 成本达到相当准确率。每位辩手只看到同侪的一个子集。

含义：

- 全连接 N=5、R=3 = 5 × 3 = 15 次提议，每次阅读 4 位同侪 = 60 次批评操作。
- 星形 N=5、R=3（1 个中心 + 4 个辐条）= 15 次提议，辐条只读中心 = 12 次批评操作。

### 辩论何时有益

- **事实准确性。** N 个独立提议交叉核对，可减少幻觉。
- **规则遵循。** 国际象棋走法合法性——一个模型漏掉规则，其他模型能抓住。
- **开放式推理。** 多种视角的表述逐步逼近正确答案。

### 辩论何时有害

- **延迟敏感的 UX。** N × R 个串行轮次会带来你可能无法承受的延迟。
- **成本敏感的规模。** 每个问题消耗 N × R 份 token。
- **简单的事实查询。** 一次查询比五场辩论更便宜。

### 2026 年的实际落地

- **Anthropic orchestrator-workers**（第 12 课）——带综合步骤的辩论变体。
- **LangGraph supervisor**（第 13 课）——中心路由 + 专家智能体，可将辩论实现为一个节点。
- **OpenAI Agents SDK**（第 16 课）——智能体来回移交，进行迭代批评。
- **多智能体评测**——将辩论 + 评估器-优化器配对，以获得评测信号。

### 这个模式容易出错的地方

- **收敛崩塌（Convergence collapse）。** 所有智能体都收敛到第一个错误答案。用强制分歧轮次来缓解。
- **中心节点故障（Hub failure）。** 在星形拓扑中，一个有问题的中心会污染所有人。轮换中心，或使用多个中心。
- **提示同质化。** 所有智能体使用相同的 prompt，就会产出相同的答案。使用多样化的 prompt 和/或模型。

```figure
debate-converge
```

## 构建它

`code/main.py` 用 stdlib 实现辩论：

- `Debater` 类（脚本化 LLM，每位辩手的意见会漂移）。
- `FullMeshDebate` 与 `SparseDebate` 运行器。
- 三个问题：一个事实类、一个规则类、一个推理类。
- 指标：收敛答案、收敛所需轮数、批评操作总数。

运行它：

```
python3 code/main.py
```

输出：各协议的准确率与成本；稀疏拓扑在 2/3 的问题上与全连接持平，且成本更低。

## 使用它

- **Anthropic orchestrator-workers** 用于简单的 2-3 个 worker 的辩论。
- **LangGraph** 用于带 checkpointing 的有状态多轮辩论。
- **自定义实现** 用于研究，或需要专门正确性保证的场景。

## 交付它

`outputs/skill-debate.md` 为多智能体辩论搭建脚手架，支持配置拓扑、N、R 与收敛规则。

## 练习

1. 实现一条「强制分歧」规则：第 1 轮中每位辩手必须给出不同的提议。测量它对收敛速度的影响。
2. 加入置信度加权聚合：辩手返回 (answer, confidence)，聚合器按置信度加权。这有帮助吗？
3. 将一个「智能体」换成意见不同的另一个脚本化 LLM。异质性会提升准确率吗？
4. 测量你的 3 个问题上全连接与稀疏的 token 成本。绘制成本 vs 准确率。
5. 阅读 Society of Minds 论文。把你的玩具实现扩展到 N=5、R=3。什么会崩溃？什么会变好？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 辩论（Debate） | 「多智能体批评」 | N 个提议者、R 轮交叉批评、最终收敛 |
| 全连接网状（Full mesh） | 「人人读人人」 | 每轮每位辩手阅读所有同侪 |
| 稀疏拓扑（Sparse topology） | 「有限的同侪视野」 | 辩手只阅读部分同侪 |
| 中心辐射式（Hub-and-spoke） | 「星形拓扑」 | 一个中心辩手，N-1 个辐条只读中心 |
| 收敛（Convergence） | 「达成一致」 | 辩手收敛到共享答案 |
| Society of Minds | 「Du 等人的辩论论文」 | ICML 2024 的多智能体辩论方法 |

## 进一步阅读

- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) —— 多智能体辩论的经典论文
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776) —— 稀疏拓扑的研究结果
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) —— orchestrator-workers 作为辩论的一种变体
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) —— 单模型自我批评的对应方法
