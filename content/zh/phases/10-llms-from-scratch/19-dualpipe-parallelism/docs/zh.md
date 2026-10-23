# DualPipe 并行

> DeepSeek-V3 在 2,048 块 H800 GPU 上训练，MoE 专家散布在多个节点。跨节点专家 all-to-all 通信成本是每 1 GPU 小时计算对应 1 GPU 小时通信。GPU 有一半时间闲着。DualPipe（DeepSeek，2024 年 12 月）是一个双向流水线，把前向与反向计算同它们触发的 all-to-all 通信重叠。气泡下降，吞吐量上升；而且既然专家并行（EP）本就把专家分散到各 rank 上，保留两份模型参数副本（即名称中的「dual」）代价很低。本课是 Learn 类型的讲解：DualPipe 实际做了什么，以及为什么 Sea AI Lab 的 DualPipeV 改进以略微更紧的气泡为代价，去掉了 2× 参数成本。

**Type:** Learn
**Languages:** Python (stdlib, schedule simulator)
**Prerequisites:** Phase 10 · 05 (distributed training, FSDP, DeepSpeed), Phase 10 · 14 (open-model architectures and MoE)
**Time:** ~60 minutes

## 学习目标

- 说出 DualPipe 前向-反向 chunk 的四个组成部分，以及为什么每个都有独立的重叠窗口。
- 解释规模化的流水线气泡问题，以及「无气泡」在实践中与营销宣传中的差别。
- 为 8 个 PP rank 和 16 个 micro-batch 手工推演一遍 DualPipe 调度，确认前向与反向流互相填补空闲槽位。
- 陈述 DualPipeV（Sea AI Lab，2025）的权衡：在专家并行不活跃时，以略大的气泡为代价去掉 2× 参数复制。

## 问题

在 2k 块 H800 GPU 上训练 671B MoE 模型会遇到三个叠加的瓶颈：

1. **内存压力。** 每块 GPU 持有模型的一个切片。序列 8k、61 层、128 头上的激活内存极大。
2. **流水线气泡。** 传统流水线并行（GPipe、1F1B）在等待本阶段输入或梯度时让 GPU 闲着。8 个阶段时，即使用 1F1B 调度，约 12% 的 GPU 时间也可能是气泡。
3. **跨节点 all-to-all。** 带专家并行的 MoE 把专家散布到各节点。每次前向传播都触发一次 all-to-all 把 token 派发给专家，再触发一次 combine。在 2k GPU 上，这很容易变成 1:1 的计算通信比。

每个问题都有单独的解法：内存用梯度 checkpointing，流水线气泡用 Zero Bubble（Sea AI Lab，2023），all-to-all 用专家并行通信 kernel。DualPipe 做的是让它们协同工作。该调度在单个前向-反向 chunk 内重叠计算与通信，同时从流水线两端注入 micro-batch，并利用由此产生的调度把 all-to-all 藏在计算窗口内。

报告结果：流水线气泡接近消除，DeepSeek-V3 的 14.8T token 训练运行中 GPU 利用率超过 95%。

## 概念

### 流水线并行速览

把 N 层模型切分到 P 个设备上。设备 `i` 持有层 `i * N/P .. (i+1) * N/P - 1`。一个 micro-batch 从设备 0 到 P-1 前向流动，再从 P-1 到 0 反向流动。每个设备只有在前一个设备送来输出时才能开始自己的前向阶段，只有在下游设备送来上游梯度时才能开始反向。

GPipe（Huang 等，2019）一次调度一个 micro-batch，浪费了大部分 GPU 时间。1F1B（Narayanan 等，2021）为多个 micro-batch 交错前向与反向传播。Zero Bubble（Qi 等，2023）把反向传播拆成两部分——对输入的梯度（B）和对权重的梯度（W）——并调度它们填补气泡。Zero Bubble 之后，流水线几乎紧凑。

DualPipe 是下一步。它在上面加了两个想法：

### 想法 1：chunk 分解

每个前向 chunk 拆成四个组件：

- **注意力。** Q/K/V 投影、注意力、输出投影。
- **All-to-all dispatch。** 把 token 送到专家的跨节点通信。
- **MLP。** MoE 专家计算。
- **All-to-all combine。** 把专家输出带回的跨节点通信。

反向 chunk 增加每个组件的梯度版本。DualPipe 调度它们，使 all-to-all dispatch 与下一个 chunk 的注意力计算并行发生，all-to-all combine 与再下一个 chunk 的 MLP 计算并行发生。

### 想法 2：双向调度

多数流水线调度从阶段 0 注入 micro-batch，向阶段 P-1 流动。DualPipe 从两端注入。阶段 0 看到从这里出发的前向 micro-batch；阶段 P-1 也看到从这里出发的前向 micro-batch。两股流在中间相遇。

要做到这一点，设备 `i` 必须同时持有流水线前段层 `i` 和后段层 `P - 1 - i`。这就是 DualPipe 的「dual」部分：每个设备保留它需要服务的模型层的两份副本（每个方向一份）。在 DeepSeek-V3 的规模下，这是 2× 参数复制成本。之所以可负担，是因为专家并行已经把 MoE 专家摊得足够薄，把非专家层复制两份只是小菜一碟。

关键的是，一个方向的前向流与另一方向的反向流，恰好在单向调度中会出现气泡的地方重叠。气泡消失。

### 手工推演的调度

考虑 P = 4 个 rank，8 个 micro-batch，分成 4 个前向 / 4 个反向。时间从左到右；行是设备 rank。

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

解读「F4/F5R」记号：rank 1 在同一时间槽内既运行 micro-batch 4 的前向（在流水线中从左到右），又运行 micro-batch 5 的前向（从右到左）。这就是「双向」在操作上的含义。

在 rank 2 交叉流重叠得更早，在 rank 0 和 P-1 重叠得最晚。在调度的稳定中段，每个 rank 都让 X 方向的前向与 Y 方向的反向重叠。计算保持忙碌。前向传播的 all-to-all dispatch 藏在反向计算里。all-to-all combine 藏在前向计算里。气泡被挤出去。

### 气泡核算

标准 1F1B 流水线气泡（每 rank 浪费的时间）：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 改进降低了它，但没有到零。DualPipe 在稳定阶段，如果 micro-batch 数能被流水线深度的两倍整除，则气泡为零。稳定阶段之外（warmup 和 cooldown）有一点气泡，但不随 micro-batch 数量增长——这是论文强调的关键性质。

用营销术语说：「无气泡」。用技术术语说：气泡不随 micro-batch 数量增长。Sea AI Lab 的后续分析（DualPipeV / Cut-in-half）表明，只有在专家并行不是瓶颈时才是完全零气泡；在 EP 驱动的 all-to-all 下，总存在某种调度折衷。

### DualPipeV —— 改进版

Sea AI Lab（2025）观察到，当 EP 通信重叠不是重点时，2× 参数复制是浪费。他们的 DualPipeV 调度把双向注入折叠成「V 形」调度，在单份参数副本上运行。气泡比 DualPipe 略大，但内存节省可观。DeepSeek 在开源 DualPipe 实现中采用 DualPipeV 作为 EP 关闭模式。

权衡：

| 特性 | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| 每设备参数副本 | 2 | 1 | 1 | 1 |
| 气泡与 micro-batch 的关系 | 恒定 | 小幅增长 | 增长 | 增长 |
| 计算-通信重叠 | 完全 | 部分 | 极少 | 部分 |
| 适用场景 | EP 密集的 MoE | 稠密或 EP 较轻 | 基线 | 任意流水线 |

### 对 14.8T token 运行意味着什么

DeepSeek-V3 的预训练在 2,048 块 H800 GPU 上消耗 14.8T token，约 2.8M GPU 小时。用朴素 1F1B，他们本会把其中 12-15% 损失在流水线气泡上——340-420K GPU 小时，足以训练一个完整的 70B 模型。DualPipe 找回了大部分。没有内部日志很难直接量化贡献，但论文声称训练全过程平均 GPU 利用率超过 95%。

对较小的运行（1k GPU 以下），DualPipe 是过度设计——流水线气泡相对总成本更小，稠密模型训练也很少触及 all-to-all 瓶颈。对数千 GPU 规模的前沿 MoE 训练，它实际上是必需的。

### 它在栈中的位置

- 与 **FSDP**（第 10 阶段 · 05）互补。FSDP 把模型参数分片到各 rank；DualPipe 把计算调度到各 rank。二者结合。
- 兼容 **ZeRO-3** 梯度分片。双副本复制的记账需要与 ZeRO 的分片梯度协同。
- 需要**针对特定集群拓扑调优的自定义 all-to-all kernel**。DeepSeek 的开源 kernel 是参考实现。

```figure
expert-capacity
```

## 使用它

`code/main.py` 是一个流水线调度模拟器。它接受 `(P, n_micro_batches, schedule)` 并打印 1F1B、Zero Bubble、DualPipe 和 DualPipeV 各自的稳定阶段利用率。它是教学工具——数字与论文的定性主张一致，并不代表生产实测加速。

模拟器的价值：用不同的 P 和 micro-batch 数运行，观察 1F1B 的气泡比例如何增长，而 DualPipe 不会。

真实训练运行的集成考虑：

- 选择能整除 micro-batch 数的流水线并行深度。
- 确保专家并行 mesh 支持双向 all-to-all。DeepSeek 的 kernel 是参考。
- 第一次做时预计要在调度本身上烧一周调试时间。记账很繁琐。
- 监控每 rank 的 GPU 利用率，而不只是聚合值。DualPipe 的收益来自收紧落后节点。

## 交付它

本课产出 `outputs/skill-dualpipe-planner.md`。给定训练集群规格（GPU 数量、拓扑、互连、模型形状），它推荐流水线并行策略、使用的调度算法，以及目标规模下的预期气泡比例。

## 练习

1. 在 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)` 上运行 `code/main.py`。计算 GPU 利用率差异，并将其表示为每百万训练 token 找回的 GPU 小时。

2. 手工画出 `(P=4, micro_batches=8, schedule=dualpipe)` 的调度表。在每个时间槽标出 micro-batch ID 和方向。找出第一个没有气泡的时间槽。

3. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）的图 5。找出 DualPipe 前向 chunk 内 all-to-all dispatch 的重叠窗口。解释计算调度如何隐藏它。

4. 计算 DualPipe 对 P=8 流水线阶段的 70B 稠密模型、以及 P=16 流水线阶段的 671B MoE 模型的 2× 参数开销。说明为什么 MoE 情形的开销比例更小（大多数参数是专家，分片在大型 EP 组上）。

5. 比较 DualPipe 与 Chimera（2021 年的竞争性双向调度器）。以论文第 3.4 节为参考，指出 DualPipe 新增而 Chimera 没有的两个具体性质。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 流水线气泡 | 「每 rank 的空闲时间」 | 因流水线阶段等待输入或梯度而浪费的 GPU 周期 |
| 1F1B | 「默认流水线调度」 | 一个前向 / 一个反向交错调度；DualPipe 击败的基线 |
| Zero Bubble | 「Sea AI Lab 2023」 | 把反向拆成 B（输入梯度）和 W（权重梯度）；几乎完全收紧流水线 |
| DualPipe | 「DeepSeek-V3 调度」 | 双向流水线 + 计算-通信重叠；气泡不随 micro-batch 数增长 |
| DualPipeV | 「Cut-in-half」 | V 形改进，去掉 2× 参数复制，代价是气泡略大 |
| Chunk | 「流水线工作单元」 | 一个 micro-batch 通过一个流水线阶段的一次前向或反向 pass |
| All-to-all dispatch | 「把 token 发给专家」 | 把 token 路由到其分配的 MoE 专家的跨节点通信 |
| All-to-all combine | 「把专家输出带回来」 | MLP 之后收集专家输出的跨节点通信 |
| 专家并行（EP） | 「专家分布在各 GPU 上」 | 把 MoE 专家分片到各 rank，使不同 GPU 持有不同专家 |
| 流水线并行（PP） | 「层分布在各 GPU 上」 | 把模型层分片到各 rank；DualPipe 调度的维度 |
| 气泡比例 | 「浪费的 GPU 时间」 | (bubble_time / total_time)；DualPipe 推向零的比例 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) —— DualPipe 的原始参考
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) —— 开源参考实现，包括 DualPipeV（Cut-in-half）模式
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) —— Zero Bubble 前身
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) —— 影响 DeepSeek EP 关闭模式的 DualPipeV 分析
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) —— DualPipe 对比的 1F1B 调度
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) —— 原始流水线并行论文与气泡问题
