# 训练资源核算：FLOPs、显存与挂钟时间

> 一次训练运行本质上是一个披着代码外衣的采购决策：在你要 GPU 之前，得先知道这次训练要多少 FLOPs、每块卡要装多少显存、集群要跑多久。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10 · 04 (Pre-Training mini-GPT), Phase 10 · 05 (Scaling Distributed), Phase 3 · 14 (Experiment Tracking and Hyperparameter Sweeps)
**Time:** ~65 minutes

## Learning Objectives
- 从前后向计算的算术推导 6ND 训练 FLOPs 规则。
- 为混合精度 Adam 训练建立逐项显存预算，并解释为何惯例总计是每参数 16 字节。
- 根据 FLOPs、硬件峰值、MFU 与 GPU 数量计算挂钟时间，并解读真实 MFU 范围。
- 应用 Chinchilla 最优点，把算力预算换算成模型大小与 token 数量，并解释现代过训练实践。
- 比较 ZeRO-1/2/3 与梯度检查点，指出各自削减的显存区域。
- 使用提供的计算器回答真实的采购问题：集群规模、训练时长、GPU-小时估算成本。

## 问题

你正在参加排期会议。有人问：“我们能不能用 64 张 H100 在一个月里训完一个 7B 模型、100B token？”这个问题不是靠直觉，也不是去翻别人的评测表格。它可以拆成三个量：

1. **FLOPs。** 这次训练需要多少浮点运算？
2. **显存。** 模型状态与激活值能否装进每块 GPU，哪些并行或检查点杠杆能改变它？
3. **挂钟时间。** 在真实的 Model FLOPs Utilization（MFU）下，集群要跑多少小时或天？

只要这三个量有一个算错，项目就会延期。FLOPs 算低了会少租 GPU；显存算低了会在第 0 步就 OOM；挂钟时间太乐观会错过交付日期。这节课做一个每位参会工程师都应该能复现的口袋计算器。

## 概念

### 6ND FLOPs 规则

对于一个有 $N$ 个参数、在 $D$ 个 token 上训练的稠密 Transformer，总训练 FLOPs 近似为：

```math
F = 6 \, N \, D
```

系数 6 来自每个 token 的一次前向-后向传递。

**前向传播。** 一个 token 会在矩阵乘法或嵌入查找中接触每个参数一次。每次乘加运算计 2 FLOPs，因此前向成本为每个 token $2N$ FLOPs。

**反向传播。** 反向传播要算两样东西：流过每个操作的梯度，以及关于每个权重的梯度。对于典型稠密层，这大约是前向计算量的两倍，即每个 token $4N$ FLOPs。

```math
F_{\text{forward}} = 2ND, \qquad F_{\text{backward}} = 4ND, \qquad F_{\text{total}} = 6ND
```

因此反向传播的 FLOPs 是前向传播的两倍。这条经验法则就是 PaLM 训练报告和后续大量容量规划中使用的 $6ND$ 估算。

### 逐项显存预算

使用 Adam 的混合精度训练为每个参数保存以下内容：

| 组件 | 精度 | 每参数字节 |
|-----------|-----------|---------------------|
| 模型权重 | fp16/bf16 | 2 |
| 梯度 | fp16/bf16 | 2 |
| Adam 一阶矩 ($m$) | fp32 | 4 |
| Adam 二阶矩 ($v$) | fp32 | 4 |
| 主权重（混合精度副本） | fp32 | 4 |
| **合计** | | **16** |

```math
M_{\text{model\_state}} = 16 \, N
```

激活值还会带来另一项，它取决于批量大小 $B$、序列长度 $S$、隐藏维度 $H$、层数 $L$ 和精度字节 $b$：

```math
M_{\text{activations}} \propto B \cdot S \cdot H \cdot L \cdot b
```

对于 Transformer 块，在不使用梯度检查点的情况下，每层的激活值通常在 $12 \, BSH$ 到 $24 \, BSH$ 字节之间。梯度检查点用额外的正向重计算换取显存：只保存每层 Transformer 块的输入，反向时重新计算其余部分，从而把保存的激活值占用大致降低 $L$ 倍。

### ZeRO 与梯度检查点：每个杠杆削减哪块显存

**ZeRO-1** 把优化器状态（$m$ 和 $v$）切分到数据并行 rank 上。每张 GPU 只存 $8N / \text{world\_size}$ 字节的优化器状态，而不是 $8N$。

**ZeRO-2**  additionally 切分梯度。每张 GPU 存 $(8N + 2N) / \text{world\_size}$ 字节的优化器加梯度。

**ZeRO-3**  additionally 切分模型权重。每张 GPU 存 $(8N + 2N + 2N) / \text{world\_size}$ 字节，即每 rank 只保留权重、梯度和优化器状态的一个切片。

**梯度检查点** 不切分状态；它通过在反向传播时重计算激活值来减少激活值显存。代价是多跑一次前向，总 FLOPs 大约增加 33%。

### MFU 与挂钟时间

硬件峰值吞吐并不等于实际吞吐。Model FLOPs Utilization 描述这个差距：

```math
\text{MFU} = \frac{\text{实际达成 TFLOP/s}}{\text{硬件峰值 TFLOP/s}}
```

NVIDIA H100 SXM5 在稠密 FP16/BF16 张量核心模式下大约提供 989 TFLOP/s。大型 Transformer 训练通常观察到 0.3–0.5 的 MFU，因为数据加载、通信、内核启动和流水线气泡都会吃掉峰值。

给定 $G$ 张峰值 $P$、MFU 为 $\eta$ 的 GPU，挂钟时间（秒）为：

```math
T = \frac{6ND}{P \cdot \eta \cdot G}
```

如果你知道每 GPU 每小时的集群租赁价格，总成本为：

```math
\text{cost} = T \cdot G \cdot \frac{\text{price}}{3600}
```

### Chinchilla 最优点

Hoffmann 等人（2021）的 Chinchilla 论文指出，对于给定算力预算 $C = 6ND$，当 token 数约为参数数的 20 倍时损失最小：

```math
D \approx 20 \, N
```

代入算力预算：

```math
C = 6N(20N) = 120N^2
```

给定 $C$ 解出 $N$ 和 $D$：

```math
N = \sqrt{\frac{C}{120}}, \qquad D = 20 \, \sqrt{\frac{C}{120}}
```

对于 7B 参数模型，Chinchilla 最优数据集大小约为 $20 \times 7\,\text{B} = 140\,\text{B}$ token。

后续研究和生产模型会故意**过训练**：使用 Chinchilla 最优值 2 倍到 5 倍的 token 数，以提升推理质量和下游任务表现，尽管额外训练在“损失/算力”意义上是次优的。选择取决于模型是否会在高流量下长期服务——此时推理质量会主导生命周期成本。

## 动手实现

这节课提供一个命令行训练预算计算器。它接收模型大小、token 数、GPU 型号、GPU 数量、MFU、精度与并行级别，然后打印：

- 总训练 FLOPs，
- 每参数显存分项表，
- 估算挂钟时间与 GPU-小时成本，
- 激活值显存估算。

运行方式：

```console
python3 main.py
```

计算器还会断言本节课使用的三条合理性检查：7B/100B token 的 FLOPs 估算、70B 参数的 Adam 模型状态占用，以及 Chinchilla 公式的内部一致性。

## 验证

`code/tests/test_main.py` 锁定计算器的账目：

- 7B 参数、100B token 训练约产生 $4.2 \times 10^{21}$ FLOPs。
- 70B 参数混合精度 Adam 运行在不计激活值前约占 1.1 TB 模型状态显存。
- Chinchilla 最优点满足 $C = 6ND$（当 $D = 20N$ 时）。
- 在固定 MFU 下，挂钟时间与 GPU 数量成反比。
- ZeRO-1/2/3 各自按宣传削减对应显存组件。
- GPU 峰值 FLOP 查询返回 H100、A100、RTX 4090 的预期值。

## 为什么在职工程师需要这个

- **采购从算术开始。** “我们能不能一个月训完？”要靠 $6ND / (P \cdot \eta \cdot G)$ 回答，而不是靠希望。把公式带进会议室，讨论就会围绕 MFU 假设展开，而不是拍脑袋。
- **OOM 是预算 bug，不是意外。** 如果显存分项装不进 GPU 显存，你要在提交任何作业前就决定是用 ZeRO 阶段、梯度检查点，还是加卡。
- **MFU 才是真实容量。** 纸面集群是峰值 TFLOP/s 乘 GPU 数；你实际拿到的集群还要乘上 MFU。0.3 与 0.5 的挂钟时间相差 67%。
- **Chinchilla 是起点，不是定律。** 7B/140B 是计算最优，但高流量推理模型可能训得更久。计算器让你并排比较计算最优方案与过训练方案。
- **推理团队也需要同样技能。** KV cache 显存与 batch 吞吐就是优化器状态显存与 MFU 在推理侧的对应物。建立分项预算的习惯可以直接迁移过去。

## 提示

- 每参数 16 字节的规则已经假设了混合精度加 Adam 主权重。如果你用 8-bit 优化器或 ZeRO 切分，就把对应项按比例下调。
- 激活值是最可能出乎意料的项；序列长度和批量大小都是线性进入，因此上下文翻倍就是显存翻倍，除非梯度检查点吃掉它。
- 不知道 MFU 时，可以把 0.35 作为调优良好的稠密 Transformer 训练集群的保守初猜。
- 比较 Chinchilla 最优与过训练时，可以固定挂钟时间，看哪种模型大小和 token 数能塞进预算。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|----------------|
| 6ND 规则 | “训练 FLOPs” | $N$ 个参数、$D$ 个 token：前向 $2ND$ 加反向 $4ND$。 |
| 模型状态 | “权重 + 梯度 + 优化器” | 混合精度 Adam 的每参数 16 字节：fp16 权重、fp16 梯度、fp32 $m$、fp32 $v$、fp32 主权重。 |
| MFU | “GPU 忙不忙” | 实际吞吐与硬件峰值吞吐之比；大型 Transformer 训练常见 0.3–0.5。 |
| ZeRO | “切分优化器” | 一组数据并行策略：ZeRO-1 切优化器状态，ZeRO-2 再加梯度，ZeRO-3 再加权重。 |
| 梯度检查点 | “重算激活值” | 用约 33% 额外计算换取激活值显存大幅下降，只保存每层输入。 |
| Chinchilla 最优点 | “每个参数 20 个 token” | Hoffmann 等人（2021）导出的计算最优数据量 $D \approx 20N$。 |
| 过训练 | “比 Chinchilla 训得更久” | 故意使用超过计算最优值的 token 数以提升推理质量。 |
