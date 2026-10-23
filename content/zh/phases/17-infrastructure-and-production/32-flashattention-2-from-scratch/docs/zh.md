# 从零实现 FlashAttention-2：分块在线 Softmax

> 标准 attention 需要先把 N×N 的分数矩阵写到 HBM 才能做归一化。FlashAttention-2 把 softmax 变成一个流式、tile 局部的归约，结果与朴素公式逐位一致。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 12 (KV Cache & Flash Attention), Phase 17 · 30 (GPU Architecture and the Execution Model)
**Time:** ~75 minutes

## 学习目标
- 解释为什么 softmax 的行级归一化会迫使朴素 attention 内核在 HBM 中物化一个 N×N 矩阵。
- 推导在线 softmax 重缩放恒等式，并说明当新 tile 提高行最大值时，它如何让正在运行的输出累加器保持正确。
- 用 NumPy 实现一个 FlashAttention-2 前向传播的 CPU 级模拟，包含 Q 外层循环、tile 内因果掩码，以及运行 max/sum 重缩放。
- 在多种形状与因果掩码下，与稳定的朴素参考实现做逐位一致性验证，float64 下最大绝对差小于 1e-6。
- 用解析方法比较朴素与分块 attention 的 HBM 流量，并解读序列长度从 128 到 4096 时 O(N²) 与 O(N) 的差距。
- 描述 FlashAttention-2 相比 FlashAttention-1 的改进，以及它们对 vLLM、SGLang、FlashInfer 等生产推理后端的意义。

## 问题背景

一个 transformer attention 头的计算是：

```math
S = QK^T / \sqrt{d}, \qquad P = \operatorname{softmax}(S), \qquad O = PV
```

朴素实现简单、正确，但对长序列很慢，原因很结构：**softmax 必须先看到整行才能归一化。** 这意味着内核必须先把整个 N×N 分数矩阵算出来、存下来，再按行做 max 与指数和，接着物化 N×N 的概率矩阵，最后才能乘以 V。

在 H100 上，HBM 带宽约 3.35 TB/s，而 SRAM 带宽要高一个数量级。每一次 HBM 往返都很贵。当 N=4K、d=128 时，仅分数矩阵就有 1600 万元素；FP16 下要先读 32 MB、再写 32 MB，下一个内核才能启动。此时的瓶颈不是 FLOPs，而是**内存带宽**，墙钟时间主要由搬字节决定。

FlashAttention-1（Dao et al. 2022）沿序列维度分块，并在 SRAM 内维护运行 softmax 统计量，解决了这个问题。FlashAttention-2（Dao 2023）保留了同样的 IO-Aware 算法，但重新组织了循环顺序与工作划分：

- 前向循环改为外层遍历 Q tile、内层遍历 K/V tile，减少非 matmul FLOPs；
- 沿序列维度挖掘并行度，而不仅依赖 batch 与 head；
- 为 warp 分配固定的 Q、K、V 子 tile，减少同步。

最终结果与标准 attention 数值一致，但 HBM 流量从 O(N²) 降到 O(N)。

## 核心概念

### 为什么 softmax 强迫使用整行 buffer

向量 x 的 softmax 是：

```math
\operatorname{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
```

要算分母，必须对整行元素做指数并求和。朴素内核因此先算完整的 S=QK^T、存下来、读回来做 softmax、再存 P=softmax(S)，最后再读 P 做 PV。这相当于为一个 N×N 张量做三次 HBM 往返。在现代加速器上，这些 HBM 往返比 matmul 本身更昂贵。

### 在线 softmax：重缩放恒等式

FlashAttention 用 tile 上的在线归约替代全局 softmax。对单个 attention 行，令：

- $m_{\text{old}}$ 为目前已见过的所有 key tile 的运行最大值；
- $\ell_{\text{old}}$ 为相对于该旧最大值的运行指数和；
- $O_{\text{old}}$ 为相对于该旧最大值的运行加权值之和。

当新的 key tile 到达时，它有自己的 tile 最大值 $m_{\text{tile}}$ 与 tile 和 $\ell_{\text{tile}}$。新的行统计量为：

```math
m_{\text{new}} = \max(m_{\text{old}}, m_{\text{tile}})
```

```math
\ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \ell_{\text{old}} + e^{m_{\text{tile}} - m_{\text{new}}} \ell_{\text{tile}}
```

输出累加器也乘以同样的因子：

```math
O_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} O_{\text{old}} + e^{S_{\text{tile}} - m_{\text{new}}} V_{\text{tile}}
```

为什么这与一次性公式等价？因为给每个指数同时减去一个常数 $c$，分子和分母会乘以同一个常数：

```math
e^{x - m_{\text{new}}} = e^{x - m_{\text{old}}} \cdot e^{m_{\text{old}} - m_{\text{new}}}
```

因此，把所有先前累加的项乘以 $e^{m_{\text{old}} - m_{\text{new}}}$，就能让整个部分结果与新最大值对齐。最后一个 key tile 处理完后，最终输出是 $O = O_{\text{final}} / \ell_{\text{final}}$。在浮点重 association 范围内，计算是精确的。

这也是数值稳定性的来源。指数总是相对于目前见过的最大值来取，而不会相对于某个任意全局常数。即使原始 logits 整体偏移了几百，运行 max 的减法也能让指数保持在安全区间。

### 从 FlashAttention-1 到 FlashAttention-2

两个版本都避免物化完整 N×N 矩阵，差别在于调度与并行：

| 方面 | FlashAttention-1 | FlashAttention-2 |
|------|------------------|------------------|
| 前向循环顺序 | K/V 外层，Q 内层 | Q 外层，K/V 内层 |
| 非 matmul FLOPs | 每次输出需要更多重缩放/归约 | Q tile 驻留，减少非 matmul 工作 |
| 序列并行 | 受限 | 行块分配到不同 warp/SM |
| warp 划分 | 不够规则 | 每个 warp 固定处理 Q tile，K/V 切片一致 |
| 因果处理 | 单独因果掩码逻辑 | 在 tile 循环内融合三角掩码 |

Q 外层顺序很重要：一个 Q tile 驻留在 SRAM 中，内核可以流式遍历所有 K 和 V tile。而在 K 外层顺序中，每处理一个 K block 都要重新加载 Q，浪费内存带宽并增加非 matmul 工作。FlashAttention-2 还减少了处理同一个 Q tile 的各 warp 之间不必要的同步。

### 反向传播：重算，而不是存储

训练需要反向传播过 attention。对 Q 的梯度涉及 $dO \cdot P^T$ 这类项，其中 P 是 N×N softmax 矩阵。如果在反向时保存 P，就会抵消 FlashAttention 省内存的全部意义。FlashAttention 改为在反向时利用保存的 Q、K、V 以及前向输出的每行 max 与 sum 重算 score tile。重算很便宜，因为它在 SRAM 中 tile-by-tile 完成，HBM 流量仍是 O(N)。

对于因果 attention，FlashAttention-2 的反向尤其快，因为三角掩码在同一个 tile 循环内完成。不需要单独的稀疏掩码张量，只有下三角 tile 会被计算。

## 动手实现

本节课提供了一个纯 NumPy 的 FlashAttention-2 模拟。它不是 GPU 内核，但实现了完全相同的 tile 循环代数，让你在没有 CUDA 脚手架的情况下看到数学如何工作。

运行方式：

```console
python3 main.py
```

文件包含：

- `naive_attention(Q, K, V, causal=False)` —— 稳定参考实现，做 softmax 前先减去行 max。
- `flash_attention_forward(Q, K, V, Br, Bc, causal=False)` —— 分块在线 softmax 前向传播，Q 外层循环，因果掩码在 tile 内完成。
- `memory_work(N, d)` —— 解析 HBM 流量模型，统计完整 N×N 矩阵 touches 与总字节数。
- `print_traffic_table(seq_lengths)` —— 打印 N=128 到 4096 的 O(N²) 与 O(N) 对比表。

演示覆盖四个方面：

1. `flash_attention_forward` 与 `naive_attention` 在 full/causal 掩码下的数值一致性，float64 下最大绝对差接近机器精度。
2. HBM 流量表：朴素 attention 的 N×N touches 随 N 二次增长，而 FlashAttention 保持为零。
3. 极端值稳定性测试：logits 整体偏移 +500。朴素 float32 的 `exp`/`sum`/`divide` 会溢出成 NaN；在线 softmax 仍有限且与稳定参考一致。
4. 因果 attention 不变量：第 `i` 行只 attend 到 key `0..i`。

本节课的 Shipped artifact 是 `flash_attention_forward`：你可以把它 import 到任何 NumPy 项目里，作为逐位精确、用于实验和教学的 attention 实现。

## 验证

`code/tests/test_main.py` 锁定以下实现：

- 多种非因果形状下，分块与朴素 attention 逐位一致；
- 因果掩码下，分块与朴素 attention 逐位一致；
- 改变 `Br`、`Bc` 的 tile 大小不改变结果；
- 因果依赖：第 `i` 行仅依赖 key `0..i`；
- +500 logit 偏移下的极端值稳定性；
- 不做 max 偏移的朴素 softmax 在 float32 下溢出；
- 内存工作模型：朴素 N×N touches 二次增长，FlashAttention HBM 字节线性增长；
- `flash_attention_forward` 可 import，`main()` 干净退出。

## 对工程师为什么重要

- **attention 后端选择不再是黑魔法。** 当 vLLM 或 SGLang 提供 `flash_attention_2`、`flash_attention_3` 或 `flashinfer` 时，你明白背后的取舍：分块在线 softmax 消除了 N×N HBM 写入，后续版本在此基础上增加 warp 级并行与 Hopper 专用异步。
- **因果 attention 是 FlashAttention-2 最大的收益场景。** 解码器 prefill 带因果掩码时，过去需要物化并清零一个三角矩阵；把掩码融合进 tile 循环彻底去掉了这部分工作。
- **反向传播才是内存故事的核心。** 重算 score tile 让激活内存保持 O(N)，这是长上下文训练可行的原因。
- **知道何时停止手写。** PyTorch `scaled_dot_product_attention` 会根据硬件与掩码自动分派到 FlashAttention、memory-efficient attention 或数学回退。生产 serving 优先用框架分派；自定义掩码、稀疏模式或研究内核时，你刚写的循环代数就是起点。
- **数值稳定性不是事后补丁。** 运行 max 减法与 `logsumexp` 的安全来源相同，也是 transformer 不需要逐 head 限制 logits 的原因。

## 提示

- 如果分块输出与朴素参考的差大于 1e-6，检查两者是否使用相同的 `scale` 和 `dtype`；本课验证建议使用 float64。
- `Br` 和 `Bc` 改变的是 SRAM 工作集，不改变数学结果。真实内核中选择能整除序列长度的值可避免边界 bug；这里的代码显式处理了边缘 tile。
- tile 循环内的因果掩码只需要遮盖 key 位置大于 query 位置的元素，不要单独构建 N×N 掩码。
- 重缩放因子 $e^{m_{\text{old}} - m_{\text{new}}}$ 在运行最大值不变时恰好为 1，因此常见情况很廉价。
- 在线 softmax 稳定的原因是每个指数都相对于目前见过的最大值计算，而非固定全局常数。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|---------|---------|
| Online softmax | "流式 softmax" | 跨 tile 增量更新 max 与指数和，使 softmax 不需要一次性拿到整行。 |
| Running max | "m" | 某个 attention 行目前已见最大分数，用于重缩放旧的部分结果。 |
| Running sum | "l" | 相对于当前运行最大值的未归一化指数和。 |
| Rescaling identity | "FA 技巧" | $e^{x - m_{\text{new}}} = e^{x - m_{\text{old}}} e^{m_{\text{old}} - m_{\text{new}}}$；新 tile 提高最大值时保持部分输出一致。 |
| Q-outer loop | "FA2 循环顺序" | 前向传播让 Q tile 驻留并流式遍历 K/V tile，减少非 matmul 工作。 |
| Recomputation | "不存 N×N 激活" | 反向传播从 Q/K/V 重算 score tile，而不是保存 softmax 矩阵。 |
| Causal mask fusion | "tile 内因果" | 在 tile 循环内应用三角掩码，避免物化单独稀疏掩码。 |
| Memory-bound | "带宽受限" | HBM 带宽而非 FLOPs 决定运行时间；不分块的 attention 正处此状态。 |
