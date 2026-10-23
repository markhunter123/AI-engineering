# Jamba —— 混合 SSM-Transformer

> 状态空间模型（SSM）和 Transformer 想要的东西不同。Transformer 通过注意力以二次成本换取质量。SSM 通过递归换取线性时间推理和恒定内存，但质量落后。AI21 的 Jamba（2024 年 3 月）和 Jamba 1.5（2024 年 8 月）把它们放进同一个模型：每 7 个 Mamba 层配 1 个 Transformer 层，每隔一个块加 MoE，上下文窗口达到 256k，且能装进单块 80GB GPU。Mamba-3（ICLR 2026）用复值状态空间和 MIMO 投影收紧了 SSM 一侧。本课从头到尾通读这两种架构，并解释为什么混合配方在纯 SSM 和纯 Transformer 的长上下文尝试都失败后，仍然挺过了三年的规模扩展。

**Type:** Learn
**Languages:** Python (stdlib, layer-mix calculator)
**Prerequisites:** Phase 10 · 14 (open-model architectures), Phase 10 · 17 (native sparse attention)
**Time:** ~60 minutes

## 学习目标

- 解释 Jamba 块中的三种原语——Transformer 层、Mamba 层、MoE——以及 1:7、隔层交替的配方。
- 在高层次上描述 SSM 的递归形式，并说明它为什么能实现恒定内存推理。
- 计算 Jamba 模型在 256k 上下文下的 KV cache 占用，并与纯 Transformer 模型所需的内存对比。
- 说出 Mamba-3 的三项创新（指数-梯形离散化、复值状态更新、MIMO）以及各自针对的问题。

## 问题

注意力在序列长度上是二次的。状态空间模型是线性的。这个差异会复利：在 256k token 时，Transformer 的注意力图每个头有 65B 个条目；而 SSM 的递归状态无论序列多长都是固定大小。

纯 SSM 模型（Mamba、Mamba-2）在较小规模上能匹配 Transformer 的困惑度，但在状态跟踪任务上落后，并在某些类别的上下文内检索上失败。直觉是：SSM 把历史压缩进固定状态，历史一长，信息就会泄漏。注意力能精确记住一切，但要付出二次成本。

显而易见的修复：两者都用。把 Transformer 层放在需要精确回忆的地方，其余地方用 SSM 层。调整比例。Jamba 是第一个在生产级别上以规模交付这一混合配方的模型（52B 总量、12B 激活、256k 上下文、单块 80GB GPU）。Jamba 1.5 把家族扩展到 398B 总量 / 94B 激活。Mamba-3（ICLR 2026）是目前最好的纯 SSM 基线，混合模型可以围绕它重建。

本课通读这三篇论文，并给出「选择正确比例」的心智模型。

## 概念

### 一页讲清 SSM

状态空间模型通过固定大小的状态 `h` 处理序列 `x_1, ..., x_N`：

```math
\begin{aligned}
h_t &= A h_{t-1} + B x_t \\
y_t &= C h_t
\end{aligned}
```

每一步状态通过线性动力学 `A` 演化，接收输入 `B x_t`，并输出 `C h_t`。`A, B, C` 都可以学习。注意这个关键性质：计算 `y_t` 只需要 `h_{t-1}` 和 `x_t`，不需要任何更早的 `x`。内存恒定。推理每个 token 是 $O(1)$。

建模质量的关键在于 `A` 的结构。S4（Gu 2021）使用高度结构化的矩阵，训练时可以高效地按长卷积求值。Mamba（Gu, Dao 2023）把固定的 `A, B, C` 换成数据相关的版本（即「选择性」部分）。Mamba-2（2024）进一步简化了结构。Mamba-3（2026）在特定位置重新加入复杂度。

关键性质：对于解码器 LLM，SSM 层可以即插即用地替代注意力层，用固定大小的每层状态取代不断增长的 KV cache。

### Jamba 块

Jamba 块按两个数字交错排列层：

- `l`：注意力与 Mamba 的比例。Jamba 使用 `l = 8`，即每 7 个 Mamba 层配 1 个 Transformer 层（7 个 Mamba + 1 个 Attention = 每组 8 层）。
- `e`：MoE 频率。Jamba 使用 `e = 2`，即每隔一层应用 MoE。

块内的层序列：

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

每个 Jamba 块是 8 层。深度为 4 个块（共 32 层）时，你得到 28 个 Mamba 层和 4 个 Attention 层。其中 16 层使用 MoE。

### 为什么是 1:7

AI21 做了消融实验：什么样的注意力与 Mamba 比例，能在他们的长上下文评测中同时取得最佳的每参数困惑度和上下文内回忆？

- 注意力太多（1:1）：质量上升，但内存和速度下降。
- 注意力太少（1:15）：内存很好，但上下文内检索失败。
- 甜点：1:7 或 1:8。

直觉是：Transformer 层负责精确回忆和状态跟踪，Mamba 层负责廉价的大宗处理。

### 位置编码

Mamba 层本身就具备位置感知（通过递归）。早期基于 Mamba 的混合模型中，注意力层不使用 RoPE——位置信息由 SSM 层提供。Jamba 1.5 为注意力层加入 RoPE 以提升长上下文泛化，这是基于经验性长上下文评测的事后改进。

### 内存预算

对于 Jamba-1 形态（32 层：28 个 Mamba + 4 个 Attention，hidden 4096，32 个注意力头）：

- KV cache（仅注意力层）：256k BF16 下为 `2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`。只有 4 个注意力层贡献。
- SSM 状态：每个 token 前缀 `28 * hidden * state_size`，但这是每层的固定大小，不随序列长度扩展。典型 Mamba 状态是每个特征 16，hidden 4096：总计 `28 * 4096 * 16 * 2 = 3.7 MB`。

对比 32 层、相同 hidden、32 头全 MHA 的纯 Transformer：256k BF16 下 `2 * 32 * 32 * 128 * 256k * 2 = 128 GB`。KV cache 缩减 8 倍。即便对比大多数 2024 年模型使用的 GQA(8) 基线（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`），Jamba 的 1:7 混合模型以 16 GB 仍然小 2 倍。

这就是 AI21 所说的「单块 80GB GPU 上的 256k 上下文」。全 MHA 纯 Transformer 的 KV cache 根本放不下；即便是 GQA 基线，也没给权重和激活留出空间；Jamba 可以。

### Mamba-3：2026 年的纯 SSM 基线

Mamba-3（ICLR 2026，arXiv:2603.15569）在纯 SSM 一侧引入三项创新：

1. **指数-梯形离散化。** 用更具表达力的递归取代 Mamba-2 中的欧拉法离散化。类卷积运算施加在核心递归内部的状态-输入上，而不是作为 `x_t` 上的外层卷积。

2. **复值状态更新。** 此前的 Mamba 把状态矩阵从复数（S4）简化为实对角（Mamba），再简化为缩放单位阵（Mamba-2）。Mamba-3 重新加入复数值——等价于对状态施加数据相关的旋转位置编码。这恢复了此前的实值简化所牺牲的状态跟踪能力。

3. **多输入多输出（MIMO）投影。** 用矩阵值投影取代逐特征的标量投影。提升建模能力和推理时的硬件利用率，且不增加解码延迟。

在 1.5B 参数下，Mamba-3 的平均下游准确率比 Gated DeltaNet 高 0.6 个百分点；MIMO 变体再增加 1.2 个点，总提升 1.8 个点。在相同的状态大小下，Mamba-3 以一半的状态追平 Mamba-2。

Mamba-3 尚未在规模化生产混合模型中使用——但它是下一代 Jamba 级模型 SSM 一侧的显然候选。

### 什么时候该用混合架构

混合架构胜出的场景：

- 上下文足够长，纯 Transformer 的 KV cache 开始吃力（64k+）。
- 任务混合了短程结构（适合 SSM）与长程回忆（需要 Transformer）。
- 你需要在单 GPU 内存预算内部署，而仅 Transformer 的 KV cache 就放不下。

混合架构失利的场景：

- 上下文很短（16k 以下）。SSM 的开销被浪费；纯 Transformer 就够了。
- 任务需要处处到处的注意力（深度推理、多文档交叉引用）。混合模型中注意力层的稀疏性会拖后腿。
- 你在扩展到万亿参数的前沿模型。纯 Transformer + MLA + MoE（DeepSeek-V3 风格）目前正在能力竞赛中获胜。

### 竞争格局

| 模型 | 家族 | 规模 | 独特主张 |
|-------|--------|------|-------------|
| Mamba-2 | 纯 SSM | 3B | 线性时间、恒定内存 |
| Jamba | 混合 | 52B/12B | 80GB 上 256k |
| Jamba 1.5 Large | 混合 | 398B/94B | 企业级长上下文 |
| Mamba-3 | 纯 SSM | 1.5B（论文） | 状态跟踪能力恢复 |
| DeepSeek-V3 | 纯 Transformer + MoE | 671B/37B | 前沿能力 |

2026 年的格局：纯 Transformer MoE 主导前沿，但混合架构占据 256k 以上的上下文利基。Mamba-3 在状态跟踪上的胜利，可能推动下一代把混合比例调得更低（更多 SSM，更少注意力）。

```figure
swiglu-ffn
```

## 使用它

`code/main.py` 是一个面向混合架构的内存计算器。给定一个 SSM-Transformer 比例和 hidden 大小 / 层数配置，它计算：

- 目标上下文下的 KV cache。
- SSM 状态内存。
- 一系列模型形态在上下文 N 下的总内存。

计算器支持：

- 纯 Transformer 基线（KV cache 随 N 增长）。
- Jamba 风格 1:7 混合。
- 纯 SSM（完全没有 KV cache）。

对于已发布的形态，数字直接来自 Jamba-1 和 Jamba-1.5 论文；对假想变体则做外推。

真实部署的集成考量：

- 大多数生产推理服务器（vLLM、SGLang）支持 Jamba 和 Mamba。请核对具体版本。
- 在 256k 上下文下，Jamba 的内存优势体现在并发请求吞吐量上。同样的 VRAM 能装下更多 Jamba 序列而非 Transformer 序列。
- Mamba-3 作为独立模型尚未在生产中发布——只有 1.5B 的研究预览。

## 交付它

本课产出 `outputs/skill-hybrid-picker.md`。给定工作负载规格（上下文长度概貌、任务组合、内存预算），它在纯 Transformer、Jamba 风格混合和纯 SSM 之间给出推荐，并明确说明内存与质量上的权衡。

## 练习

1. 运行 `code/main.py`，计算 32 层纯 Transformer（hidden 4096，32 头）和相同形态的 Jamba-1 混合在 256k 上下文下的 KV cache。验证 AI21 论文声称的约 8 倍内存缩减。

2. 修改计算器，对 1:3 混合（4 Mamba : 1 Attention）和 1:15 混合（14 Mamba : 1 Attention）建模。绘制 KV cache 随比例变化的曲线。在什么比例下 KV cache 等于 SSM 状态内存？

3. 阅读 Jamba 论文（arXiv:2403.19887）第 3 节。解释为什么 AI21 使用 Mamba-1 而不是更快的 Mamba-2。提示：混合消融那一节记录了这个决定。

4. 计算 Jamba 1.5 Large（398B 总量，94B 激活）中每隔一层加 MoE 带来的参数开销。把激活比例与 DeepSeek-V3（37B/671B）对比，并解释为什么 Jamba 的架构把激活比例推得更高。

5. 阅读 Mamba-3 论文（arXiv:2603.15569）第 3 节。用三句话解释为什么复值状态更新等价于数据相关的旋转位置编码。把答案与第 7 阶段 · 第 04 课的 RoPE 推导联系起来。

## 关键术语

| 术语 | 常见的说法 | 实际含义 |
|------|----------------|------------------------|
| 状态空间模型（SSM） | 「固定状态的递归」 | 一个带可学习递归 `h_t = A h_{t-1} + B x_t` 的层；每个 token 内存恒定 |
| 选择性 SSM | 「Mamba 的招数」 | 数据相关的 A、B、C 参数，在线性时间下提供类门控的选择性 |
| 注意力与 Mamba 比例 | 「有多少注意力层」 | 在 Jamba 中，`l = 8` 表示每 7 个 Mamba 层配 1 个注意力层 |
| Jamba 块 | 「8 层分组」 | 一个注意力层 + 七个 Mamba 层 + 在交替位置应用 MoE |
| SSM 状态 | 「隐藏缓冲区」 | 固定大小的每层状态，在 Mamba 层中取代 KV cache |
| 256k 上下文 | 「Jamba 的旗舰数字」 | Jamba-1 在单块 80GB GPU 上能容纳的序列长度；纯 Transformer 在该规模下做不到 |
| Mamba-3 | 「2026 纯 SSM」 | 目前最好的纯 SSM 架构，带复值状态 + MIMO；混合模型重建的基线 |
| MIMO | 「多输入多输出」 | Mamba-3 的创新，使用矩阵值投影而非逐特征标量 |
| 指数-梯形离散化 | 「Mamba-3 的递归」 | 更具表达力的递归，涵盖 Mamba-2 的欧拉法离散化 |
| 混合架构 | 「注意力与 SSM 混合」 | 任何交错安置 Transformer 与 SSM 层的模型；Jamba 是生产原型 |

## 延伸阅读

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887)——Jamba 原始论文、比例消融、256k 上下文主张
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570)——扩展后的家族，398B/94B 与 12B/52B 公开版本
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752)——Jamba 所基于的选择性 SSM 论文
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060)——简化结构化状态空间的后继
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569)——复值状态、MIMO，2026 年纯 SSM 前沿
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396)——S4 论文，LLM 领域 SSM 谱系的起点
