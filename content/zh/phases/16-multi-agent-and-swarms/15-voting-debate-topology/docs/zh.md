# 投票、自一致性与辩论拓扑

> 最廉价的聚合方式：采样 N 个独立 Agent，做多数投票。Wang 等人 2022 年的 self-consistency 就是对同一个模型采样 N 次来做这件事。多 Agent 把它扩展为**异构**Agent，以摆脱单一文化——不同模型、不同 prompt、不同温度、不同上下文。超越多数投票之后，辩论拓扑变得重要：MultiAgentBench（arXiv:2503.01935，ACL 2025）评估了星型、链型、树型、图型协调，发现**图型最适合研究任务**，并在超过约 4 个 Agent 时出现「协调税（coordination tax）」。AgentVerse（ICLR 2024）记录了两种涌现模式——志愿行为与从众行为——从众既是特性（找到共识）也是风险（群体思维，第 24 课）。本课梳理拓扑空间，构建每种变体，并测量协调税。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 14 (Consensus and BFT)
**Time:** ~75 minutes

## 问题

辩论可以提升准确率（Du 等，arXiv:2305.14325），也可以降低它。辩论是否有帮助，取决于四个结构性选择：

1. 谁和谁说话（拓扑）。
2. 多少轮（Du 2023：轮数与 Agent 数量各自独立地起作用）。
3. Agent 是否异构（不同基础模型能打破单一文化）。
4. 是否存在对抗性声音（从最强论点出发 vs. 稻草人式驳斥）。

把「跑 5 个 Agent 再投票」硬套到任务上的团队，往往会比单 Agent 还差。这些失败不是随机的，而是与拓扑和异构性相关。本课就是那张拓扑地图。

## 概念

### 自一致性：单模型基线

Wang 等人 2022 年的工作（"Self-Consistency Improves Chain of Thought Reasoning"）在温度 > 0 下对同一模型采样 N 次，然后对推理路径的答案进行多数投票。在 GSM8K 上的结果是：相比单次贪心解码，N=40 次采样带来显著提升。自一致性是多 Agent 投票的单 Agent 前身。

局限：自一致性只用一个基础模型，错误在构造上就是相关的。如果模型存在系统性偏差，N 个样本会共享它。

### 多 Agent 投票：异构扩展

把 N 个样本换成 N 个*不同*的 Agent。不同基础模型（Claude、GPT、Llama）、不同 prompt、不同工具访问权限。好处：错误不相关。代价：不同 Agent 成本不同；协调它们会带来额外开销。

2026 年对异构辩论的典型叫法是 **A-HMAD**——Adversarial Heterogeneous Multi-Agent Debate。它并未被普遍采用，但论文用它来指「不同模型进行辩论，从而减少单一文化崩塌带来的相关错误」。

### 四种拓扑

```
star                chain               tree                graph

    ┌─A─┐           A─B─C─D         ┌──A──┐              A───B
    │   │                           │     │              │ × │
    B   C                           B     C              D───C
    │   │                          / \   / \
    D   E                         D   E F   G           (fully connected)
```

星型：一个中心，其他所有节点只与中心对话。相当于没有回传通道的监督者-工作者模式。
链型：线性，每个 Agent 看到前一个的输出。类似 pipeline。
树型：层次化，用于层次化 Agent 系统（第 06 课）。
图型：任意节点对任意节点。包括全连接团和任意 DAG。

### 协调税（MultiAgentBench）

MultiAgentBench（MARBLE，ACL 2025，arXiv:2503.01935）在包含研究、编码、规划的任务套件上对星型、链型、树型、图型做了基准测试。关键测量结果：

- **图型**拓扑在研究任务上胜出。信息任意流动；Agent 可以互相批评。
- **星型**在快速给出事实的任务上胜出。中心负责过滤与整合。
- **链型**在分步 pipeline（分阶段精炼）上胜出。
- **协调税**在图型拓扑超过约 4 个 Agent 时出现。实际耗时和 token 成本的增长快于质量。

4 个 Agent 的天花板是经验性的，而非根本性的。它反映了 2026 年 LLM 的上下文容量：每个 Agent 的上下文都被同侪的输出填满，而当所有人都能看到所有人后，再增加第 N+1 个 Agent 的边际价值会下降。

### 多 Agent 辩论策略（「Should we be going MAD?」）

arXiv:2311.17371 是 2023 年关于 MAD 策略的综述。其被他人反复验证的关键发现是：与自一致性*结构相似*的 MAD 变体（独立采样 + 聚合），在预算相同时往往不如自一致性。MAD 最能发挥作用的场合是 Agent 真正异构、且辩论具有对抗结构（有一个 Agent 专门反驳）时。

### AgentVerse 的涌现模式

AgentVerse（ICLR 2024，https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf）记录了两个即使没有显式设计也会从多 Agent 辩论中涌现的行为：

- **志愿（Volunteer）。** 一个 Agent 主动提出帮助（「我可以接手下一步」）。有用之处：它把子任务分配给最有能力的 Agent。
- **从众（Conformity）。** 一个 Agent 调整自己的立场去迎合批评者，即使批评者是错的。这是辩论场景中的谄媚（第 14 课）。

从众正是「辩论到达成一致」会奖励强词夺理者的原因。有界轮数加独立裁判可以缓解。

### 异构性：真正能撬动准确率的旋钮

2024-2026 年实践文献中的一个模式是：把 N 个 Agent 中的一个换成不同基础模型，带来的准确率提升比把 N 加 1 更大。直觉来自单一文化——每增加一个独立的错误来源，都比增加一个相关的样本更有价值。

在极限情况下，异构性胜过数量。在多数有明确客观答案的任务上，三个不同模型胜过同一个模型的五份副本。

### 评审团方法

Sibyl 框架（见于 Minsky-LLM 文献）把「评审团」形式化——一小组专业化 Agent，在每个阶段通过投票来精炼答案。与普通多数投票不同，评审团有角色分工：一个 Agent 交叉质询，一个提供上下文，一个评估合理性。评审团方法介于普通投票（便宜，易单一文化）与完整 MAD（昂贵，易从众）之间。

### 投票加辩论何时占优

- 问题有客观答案（事实、数学、代码行为）。投票收敛才有意义。
- Agent 能访问不同来源或工具（异构性可用）。
- 轮数有界（通常 2-3 轮），且有独立裁判或校验器。
- 预算允许 3-5 个 Agent。在图型拓扑上超过 5-7 个，协调税会占主导。

### 投票加辩论何时有害

- 问题是观点型的。Agent 会收敛到看起来最有底气的答案，而不是最正确的答案。
- 所有 Agent 共享同一基础模型。单一文化让共识失去意义。
- 轮数无界。从众每次都赢。
- 任务很简单。单个 Agent 做 N=5 的自一致性更便宜，准确率也相当。

```figure
sw-debate-topology
```

## 构建它

`code/main.py` 实现了：

- `run_star(agents, hub, question)` —— 中心轮询每个工作者并聚合。
- `run_chain(agents, question)` —— 顺序精炼。
- `run_tree(root, children, question)` —— 深度为 2 的层次化聚合。
- `run_graph(agents, question, rounds)` —— 全对全辩论，轮数有界。
- 一个脚本化的异构性旋钮：每个 Agent 有一个 `error_bias`，表示其系统性错误倾向。
- 一个测量工具，在 N=3、5、7 下运行每种拓扑，并报告（准确率、总 token 数、模拟实际耗时）。

运行：

```
python3 code/main.py
```

预期输出：一张 拓扑 × N →（准确率、tokens、延迟）的表。图型在 N=3-5 的研究型任务上胜出；星型在快速事实任务上胜出；图型在 N=7 时展现出协调税（延迟膨胀快于准确率提升）。

## 使用它

`outputs/skill-topology-picker.md` 是一个技能：读取任务描述后，推荐一种拓扑（星型/链型/树型/图型）、一个 N（Agent 数量）、一个异构性配置（使用哪些基础模型）以及轮数上限。

## 交付它

对于任何集成：

- 先用一个强基础模型从 **N=5 的自一致性**开始。这是廉价基线。
- 如果准确率重要，升级到 **N=3 的异构投票**。测量增量。
- 只有当任务具有结构（研究、多步）、且可以有界轮数时，才升级到**辩论拓扑**。
- 始终记录少数簇。当少数派持续正确时，你就得到了多样性信号。
- 在报告准确率的同时，也基准测试实际耗时和 token。「成本 10 倍换来更好准确率」是商业决策。

## 练习

1. 运行 `code/main.py`。画出图型拓扑的协调税曲线：准确率 vs N、tokens vs N。曲线在哪个 N 处出现拐点？
2. 实现 A-HMAD：三个有意设置不同偏差的 Agent。在第 14 课的单一文化攻击上，同偏差基线与 A-HMAD 相比如何？
3. 给图型拓扑增加一个「裁判」角色：它不投票，只给最终共识打分。这会改变涌现出的从众行为吗？
4. 阅读 AgentVerse 论文（ICLR 2024）。找出你的实现中最强烈表现出的涌现行为。你能通过修改 prompt 引出相反的行为吗？
5. 阅读 MultiAgentBench（arXiv:2503.01935）第 4 节（拓扑实验）。用你的测量工具，在论文中的一个任务上复现「图型在研究任务上胜出」的结果。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|----------------|------------------------|
| Self-consistency | "采样 N 次，投票" | Wang 2022。单模型、N 个温度>0 的样本，对推理路径做多数投票。 |
| Heterogeneity | "不同的模型" | 由不同基础模型或 prompt 族构成的集成。打破单一文化。 |
| MAD | "多 Agent 辩论" | Agent 在多轮中交换批评的统称。见 Du 2023。 |
| A-HMAD | "对抗性异构 MAD" | 强调不同模型 + 对抗结构的 MAD 变体。 |
| Topology | "谁和谁说话" | 星型、链型、树型、图型。决定信息流。 |
| Coordination tax | "收益递减" | 图型拓扑超过约 4 个 Agent 后，成本增长快于质量。 |
| Volunteer behavior | "未经请求的帮助" | AgentVerse 涌现模式：Agent 主动提出接手一步。 |
| Conformity behavior | "压力下的附和" | AgentVerse 涌现模式：Agent 向批评者靠拢。 |
| Jury | "小型专业评审团" | Sibyl 式的带角色集成（审查者、上下文、打分者）。 |

## 延伸阅读

- [Wang et al. — Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) —— 单模型基线
- [Du et al. — Improving Factuality and Reasoning via Multiagent Debate](https://arxiv.org/abs/2305.14325) —— Agent 数量和轮数各自独立地起作用
- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) —— 拓扑基准，表明图型最适合研究、链型最适合 pipeline
- [Should we be going MAD?](https://arxiv.org/abs/2311.17371) —— MAD 策略综述；发现同等预算下 MAD 常常不如自一致性
- [AgentVerse (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) —— 志愿与从众两种涌现模式
- [MARBLE repo](https://github.com/ulab-uiuc/MARBLE) —— 参考基准实现
