# MARL —— MADDPG、QMIX、MAPPO

> 多 Agent 协作的强化学习传统仍在 2026 年影响着 LLM Agent 系统。**MADDPG**（Lowe 等人，NeurIPS 2017，arXiv:1706.02275）提出了集中训练、分散执行（Centralized Training, Decentralized Execution，CTDE）：训练时每个 critic 能看到所有 Agent 的状态与动作；测试时只运行本地 actor。适用于合作、竞争和混合场景。**QMIX**（Rashid 等人，ICML 2018，arXiv:1803.11485）采用带单调混合网络的值分解；每个 Agent 的 Q 组合成联合 Q，使 `argmax` 能干净地分配——在星际争霸多 Agent 挑战赛（SMAC）上占主导。**MAPPO**（Yu 等人，NeurIPS 2022，arXiv:2103.01955）是带集中价值函数的 PPO；在 particle-world、SMAC、Google Research Football、Hanabi 上以极少的调参「出奇有效」。这些算法支撑着必须以分散方式行动的 Agent 团队的策略训练。MAPPO 是 **2026 年合作式 MARL 的默认基线**。本课在一个小型网格世界玩具环境中逐一实现它们，在接触 LLM Agent 训练之前把三个思想刻进肌肉记忆。

**Type:** Learn
**Languages:** Python (stdlib, small NumPy-free implementations)
**Prerequisites:** Phase 09 (Reinforcement Learning), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~90 minutes

## 问题

LLM Agent 系统越来越多地为 Agent 间协作训练策略：何时让位、何时行动、调用哪个同伴。告诉你如何训练这类策略的文献是多 Agent 强化学习（Multi-Agent Reinforcement Learning，MARL），它早于 LLM 浪潮，拥有一小组占主导地位的算法。

没有模式词汇就去读 MARL 论文很痛苦。集中训练分散执行（CTDE）、值分解、集中 critic 不是空话——它们是针对具体问题的具体答案：

- 独立 RL（每个 Agent 单独学习）从每个 Agent 的视角看是非平稳的。不好。
- 集中 RL（一个 Agent 控制所有）无法扩展，且违反执行约束。
- CTDE 兼取两者之长：用全局信息训练，用本地策略部署。

## 概念

### 论文使用的三个环境

- **Particle World（多 Agent 粒子环境）。** 简单的 2D 物理，包含合作/竞争任务。MADDPG 的原始试验台。
- **星际争霸多 Agent 挑战赛（SMAC）。** 合作式微操，部分可观测。QMIX 的试验台。离散动作，连续状态。
- **Google Research Football、Hanabi、MPE。** MAPPO 的基线环境。

不同环境有不同的动作/观测类型。算法据此选择。

### MADDPG（2017）—— CTDE 模式

每个 Agent `i` 有一个 actor `mu_i(o_i)`，把自己的观测映射为动作。每个 Agent 还有一个 critic `Q_i(x, a_1, ..., a_n)`，训练时能看到所有观测和所有动作。actor 依据 critic 的评估，通过策略梯度更新。

```
actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimate
```

为什么用 CTDE：训练时我们知道每个人的动作；用这些信息来降低每个 critic 的方差。部署时，每个 Agent 只看到 `o_i` 并调用 `mu_i(o_i)`。

失效模式：critic 随 Agent 数量 N 增长（输入包含所有动作）。在不做近似的情况下，超过约 10 个 Agent 就无法扩展。

### QMIX（2018）—— 值分解

仅限合作式。全局奖励是每个 Agent Q 值的单调函数之和：

```math
Q_{\text{tot}}(\tau, a) = f\big(Q_1(\tau_1, a_1), \ldots, Q_n(\tau_n, a_n)\big), \quad \frac{\partial f}{\partial Q_i} \ge 0
```

单调性保证 `argmax_a Q_tot` 可以由每个 Agent 独立选择 `argmax_{a_i} Q_i` 来计算。这**恰恰就是**你需要的分散执行性质。训练时，一个混合网络从每个 Agent 的 Q 产生 `Q_tot`。

QMIX 为何在 SMAC 上胜出：合作式星际争霸微操具有同质 Agent、局部观测、全局奖励——与值分解完美契合。

失效模式：单调性约束有局限；有些任务的奖励结构并非单调可分解（一个 Agent 为团队牺牲）。后续扩展（QTRAN、QPLEX）放宽了这一约束。

### MAPPO（2022）—— 被忽视的默认选择

Multi-Agent PPO：带集中价值函数的 PPO。每个 Agent 有自己的策略；所有 Agent 共享（或各自拥有）能看到完整状态的价值函数。Yu 等人 2022 年在五个基准上将 MAPPO 与 MADDPG、QMIX 及其扩展进行比较，发现：

- MAPPO 在 particle-world、SMAC、Google Research Football、Hanabi、MPE 上达到或超过 off-policy MARL 方法。
- 需要极少的超参数调优。
- 训练稳定；跨随机种子可复现。

在这篇论文之前，社区低估了 on-policy MARL。2026 年，MAPPO 是合作式 MARL 的默认基线；任何新方法都必须超过它。

### LLM Agent 工程师为何应关注

三个直接用途：

1. **路由训练。** 一个元 Agent 选择哪个子 Agent 处理任务。这是一个 MARL 问题：N 个分散的子 Agent 加一个集中路由器。MAPPO 适用。
2. **角色涌现。** 在生成式 Agent 仿真中，训练 Agent 随时间承担互补角色，本质上是一个 MARL 问题。QMIX 式的值分解在构造上强制互补性。
3. **多 Agent 工具使用。** 当 Agent 共享工具并争夺预算时，用 CTDE 训练它们可以产出遵守资源约束、可部署的本地策略。

实际注意事项：2026 年，大多数生产 LLM Agent 系统用 prompt 表达策略，而不是训练策略。MARL 适用于以下条件齐备时：(a) 大量交互数据，(b) 清晰的奖励信号，(c) 愿意投入训练基础设施。

### 超越 RL 的 CTDE 设计模式

即使不训练，CTDE 也是一种有用的架构模式：

- 在*设计*时，假设整个团队可见。
- 在*运行*时，强制分散执行：每个 Agent 只看到 `o_i`。

这一模式迫使你把每个 Agent 的状态显式化，并提前考虑部分可观测性。许多生产多 Agent 系统默默假设处处共享状态——CTDE 纪律能防止这一点。

### 非平稳性问题

当多个 Agent 同时学习时，每个 Agent 的环境（包含其他 Agent 的策略）是非平稳的。经典的单 Agent RL 证明失效。本课中的 MARL 算法都在应对这一点：

- MADDPG：全局 critic 看到所有动作，因此其价值估计是平稳的。
- QMIX：值分解把学习移到联合 Q 空间，在那里最优性有明确定义。
- MAPPO：集中价值函数削弱了他者策略变化带来的方差。

在 LLM Agent 系统中，非平稳性表现为「我的 Agent 上个月还好用，现在上游那个 Agent 一改，我的就开始出问题」。用 CTDE 训练 MARL 是有原理依据的修复；提示层面的修复更快但不够持久。

### 本课不涉及的内容

训练真实网络是第 09 阶段的话题。本课构建脚本化策略版本，在不做梯度更新的情况下演示 CTDE、值分解和集中价值模式。目标是在你上手完整 MARL 库（PyMARL、MARLlib、RLlib multi-agent）之前把这些模式内化。

```figure
sw-ctde
```

## 构建它

`code/main.py` 实现了三个模式演示，都在一个极小的 2 Agent 合作网格世界上：

- 环境：4x4 网格上的 2 个 Agent，一个奖励 pellet。奖励 = 1，只要任一 Agent 到达 pellet；任务结束。
- `IndependentAgents`——每个 Agent 把其他 Agent 视为环境。基线。
- `MADDPGStyle`——集中 critic 计算联合价值；actor 策略据此更新。脚本化的策略改进。
- `QMIXStyle`——带单调混合器的值分解。
- `MAPPOStyle`——集中价值函数；策略依据共享基线更新。

四者运行相同的回合，并报告平均到达目标步数。CTDE 变体比独立基线收敛到更短的路径。

运行：

```
python3 code/main.py
```

预期输出：独立 Agent 平均约需 6 步；CTDE 变体收敛到约 3.5 步（4x4 网格的最优值为 3）。尽管策略是脚本化的，模式差异仍然显现。

## 使用它

`outputs/skill-marl-picker.md` 是一个技能，为给定的多 Agent 任务选择 MARL 算法：合作还是竞争、同质还是异构、动作空间类型、规模、奖励信号。

## 交付它

生产中 MARL 很少见。当你确实要用时：

- **从 MAPPO 开始。** 2022 年的论文确立了它作为基线；先复现它能省下数周追逐更花哨方法的功夫。
- **记录每个 Agent 的观测与动作流。** 没有逐 Agent 的追踪，调试 MARL 毫无希望。
- **把训练代码与执行代码分开。** CTDE 是一种纪律；让执行路径真正只看到 `o_i`。
- **奖励塑形警告。** MARL 对奖励设计极其敏感。塑形中一个协作 bug，Agent 就会学会利用它。运行对抗性测试。
- **对于 LLM Agent**，先考虑提示层面的策略。只有当交互数据 + 奖励信号 + 基础设施都具备时，才投入 MARL 训练。

## 练习

1. 运行 `code/main.py`。测量独立 Agent 与 MAPPO 风格 Agent 之间的到达目标步数差距。在 6x6 网格上，这个差距会变大还是变小？
2. 实现一个竞争变体：两个 Agent、一个 pellet，只有先到者获得奖励。哪种模式能干净地处理竞争？历史上是 MADDPG。
3. 阅读 MADDPG（arXiv:1706.02275）第 3 节。用你自己的话，以伪代码符号化地实现确切的 critic 更新规则。
4. 阅读 MAPPO（arXiv:2103.01955）。作者为何论证集中价值 + PPO 在其基准上优于 off-policy MARL？列出三条最强的论据。
5. 把 CTDE 作为设计模式应用到一个假想的 LLM Agent 系统（例如研究 Agent + 摘要器 + 编码器）。设计时有哪些联合信息在运行时不可用？

## 关键术语

| 术语 | 常见说法 | 真实含义 |
|------|----------------|------------------------|
| MARL | 「多 Agent 强化学习」 | 面向多 Agent 系统的强化学习。 |
| CTDE | 「集中训练、分散执行」 | 用全局信息训练；用本地策略部署。 |
| MADDPG | 「多 Agent DDPG」 | CTDE，每个 Agent 的 critic 看到所有观测 + 动作。 |
| QMIX | 「值分解」 | 对每个 Agent Q 的单调混合。合作式。 |
| MAPPO | 「多 Agent PPO」 | 带集中价值函数的 PPO。2026 默认基线。 |
| 值分解 | 「个体 Q 之和」 | 联合 Q 表示为每个 Agent Q 的单调函数。 |
| 非平稳性 | 「移动的目标」 | 每个 Agent 的环境随他人学习而变化。MARL 的核心问题。 |
| On-policy / off-policy | 「从当前 / 回放中学习」 | PPO 是 on-policy（MAPPO）；DDPG 和 Q-learning 是 off-policy。 |
| SMAC | 「星际争霸多 Agent 挑战赛」 | 合作式微操基准；QMIX 的主场。 |

## 延伸阅读

- [Lowe 等人 — 面向混合合作竞争环境的多 Agent actor-critic](https://arxiv.org/abs/1706.02275) — MADDPG；NeurIPS 2017
- [Rashid 等人 — QMIX：用于深度多 Agent 强化学习的单调值函数分解](https://arxiv.org/abs/1803.11485) — QMIX；ICML 2018
- [Yu 等人 — PPO 在合作式多 Agent 游戏中的惊人有效性](https://arxiv.org/abs/2103.01955) — MAPPO；NeurIPS 2022
- [BAIR 关于 MAPPO 的博客文章](https://bair.berkeley.edu/blog/2021/07/14/mappo/) — 对 MAPPO 结果的可读性阐释
- [SMAC 仓库](https://github.com/oxwhirl/smac) — 星际争霸多 Agent 挑战赛
