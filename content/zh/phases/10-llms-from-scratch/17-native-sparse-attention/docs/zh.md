# 原生稀疏注意力（DeepSeek NSA）

> 在 64k token 时，注意力占解码延迟的 70-80%。每个开放模型实验室都有修复计划。DeepSeek 的 NSA（ACL 2025 最佳论文）是留下来并成为主流的那一个：三条并行注意力分支——压缩的粗粒度 token、选择性保留的细粒度 token，以及用于局部上下文的滑动窗口——通过一个学习到的门控组合。它硬件对齐（对 kernel 友好）、原生可训练（在预训练中生效，而不是推理时才外挂），在 64k 解码上比 FlashAttention 更快，同时匹配或超过全注意力的质量。本课端到端构建三条分支，并展示为什么这种稀疏性端到端可微。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 12 (KV cache, flash-attention), Phase 7 · 15 (attention variants), Phase 10 · 16 (differential attention)
**Time:** ~60 minutes

## 学习目标

- 说出 NSA 的三条注意力分支以及各自捕获什么。
- 解释为什么 NSA 是「原生可训练」的，而此前的稀疏注意力方法只能用于推理。
- 以压缩块大小和选择 top-k 为函数，计算 NSA 在 64k 上下文下相对全注意力的注意力计算节省。
- 在短合成序列上用 stdlib Python 实现三分支组合，并验证门控权重的行为。

## 问题

全注意力在序列长度 N 时每层需要 `O(N^2)` 时间和 `O(N)` KV cache。在 64k token 时，计算和内存带宽数字是灾难性的。NSA 论文的理论估计：64k 时注意力占解码总延迟的 70-80%。所有下游指标——TTFT、tokens/sec、每百万 token 成本——都被注意力成本主导。

稀疏注意力是显而易见的答案。此前尝试分两类。固定模式稀疏（滑动窗口、strided、块局部）丢弃信息，在长程召回任务上失败。推理时稀疏（KV cache 剪枝、H2O、StreamingLLM）应用于在稠密注意力上预训练的模型，只能恢复潜力加速的一小部分，因为模型从未被要求让信息经由稀疏模式路由。

Native Sparse Attention（Yuan 等，DeepSeek + PKU + UW，ACL 2025 最佳论文，arXiv:2502.11089）两者兼顾：模型在预训练期间学习到的稀疏模式，以 kernel 对齐的算法实现，在推理时真正兑现计算节省。两年后，NSA 或其直系后代将成为每个前沿长上下文模型的默认注意力。

## 概念

### 三条并行分支

对每个查询，NSA 对 KV cache 的三种不同视图各运行一次注意力：

1. **压缩分支。** token 按大小 `l`（通常 32 或 64）分块。每块通过一个小型学习到的 MLP 压缩成一个摘要 token。查询在这些压缩 token 上做注意力，得到整条序列的粗粒度视图。

2. **选择分支。** 利用压缩分支的注意力分数，找出与当前查询最相关的前 k 个块。读取这些块的细粒度（未压缩）token，查询在它们全部上做注意力。把压缩分支的注意力视为选择的路由信号。

3. **滑动窗口分支。** 查询对最近的 `W` 个 token（通常 512）做注意力，获取局部上下文。这条分支捕获结构密集的短程模式（语法、局部共指），另外两条可能漏掉。

三条分支的输出通过一个学习到的逐位置门控组合：

```math
\text{out} = g_{\text{cmp}} \cdot \text{out}_{\text{cmp}} + g_{\text{sel}} \cdot \text{out}_{\text{sel}} + g_{\text{win}} \cdot \text{out}_{\text{win}}
```

`g_cmp, g_sel, g_win` 来自基于查询的小型 MLP 的门控权重。它们不必和为 1——可以独立加权各分支。

### 为什么它是「原生可训练」的

选择步骤（top-k 块）是离散的。离散操作会中断梯度流。此前的稀疏注意力工作要么跳过选择的反向传播（限制训练），要么使用连续松弛，在推理时无法带来真正的稀疏性。

NSA 绕开了这一点：压缩分支注意力本身就是对整条序列的可微粗粒度注意力。top-k 操作只是复用压缩分支的注意力分数，来决定加载哪些细粒度块。梯度通过压缩分支分数流动（这些分数既影响压缩输出，也影响选择逻辑），被选块对最终输出的贡献也可微。不可微的 `top_k` 操作在前向计算图上是 no-op——它只控制从内存加载哪些块。

这就是 NSA 能端到端用于预训练的原因。模型学会联合地在三条分支之间路由信息，产生的稀疏模式在推理时真正兑现所承诺的加速。

### 硬件对齐的 kernel

NSA 的 kernel 针对现代 GPU 内存层级设计。kernel 按 GQA 组加载查询（外层循环），为每组获取对应的稀疏 KV 块（内层循环），并在 SRAM 上运行注意力。因为每个查询组看到相同的被选块（选择按查询组而非查询头进行），KV 加载在组内摊销。算术强度保持高位。

论文报告 Triton kernel 在 64k 解码上比 FlashAttention 快 9 倍，且加速比随序列长度增长。前向和反向 kernel 都提供了。

### 计算预算

设 `N` 为序列长度，`l` 为压缩块大小，`k` 为 top-k 选择数量，`w` 为滑动窗口，`b` 为被选块大小（通常等于 `l`）。

- 压缩分支：每个查询 `O(N/l)` 个键，总计 `O(N * N / l)`。
- 选择分支：每个查询 `O(k * b)` 个键，总计 `O(N * k * b)`。
- 滑动分支：每个查询 `O(w)` 个键，总计 `O(N * w)`。

总计：`O(N * (N/l + k*b + w))`。

取 `N = 64k, l = 64, k = 16, b = 64, w = 512`：每查询成本为 `1000 + 1024 + 512 = 2536 keys`。全注意力是 `64000 keys`。计算减少 25 倍。

取 `N = 128k, l = 64, k = 16, b = 64, w = 512`：每查询成本为 `2000 + 1024 + 512 = 3536 keys`。全注意力是 `128000 keys`。减少 36 倍。收益随序列长度增长，这正是重点。

### 对比如何

| 方法 | 可微 | 真实推理加速 | 长程召回 |
|--------|---------------|----------------------|-------------------|
| 仅滑动窗口 | 是 | 是 | 失败 |
| Strided / 块稀疏 | 是 | 是 | 部分 |
| KV 剪枝（H2O, StreamingLLM） | N/A（推理时） | 是 | 部分 |
| MoBA（Moonshot） | 部分 | 是 | 好 |
| NSA | 是（原生） | 是（64k 时 9x） | 匹配全注意力 |

MoBA（Moonshot，arXiv:2502.13189）同期发表，采用类似的「三胜于一」思路，把 MoE 原理应用于注意力块。NSA 和 MoBA 是 2026 年长上下文预训练需要了解的两种架构。

```figure
sliding-window-attention
```

## 构建它

`code/main.py` 在短合成序列上实现三条分支，并展示：

- 压缩 MLP（为教学清晰使用简单的均值池化基线；真正的 NSA 使用学习到的 MLP）。
- 由压缩分支分数驱动的 top-k 块选择。
- 最后 `w` 个 token 上的滑动窗口注意力。
- 门控组合。
- 与全注意力对比的计算计数打印。

### 第 1 步：把 token 压缩成块

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### 第 2 步：压缩分支注意力

对查询与压缩键运行 softmax 注意力。压缩分支的分数兼作 top-k 选择的信号。

### 第 3 步：top-k 块选择

选出分数最高的 `k` 个压缩块的索引。从这些块加载原始未压缩 token，并在其上运行注意力。

### 第 4 步：滑动窗口注意力

取最后 `w` 个 token，对它们运行标准注意力。

### 第 5 步：门控 + 组合

基于查询的小型 MLP 产生三个门控权重。最终输出是三条分支输出的加权和。

### 第 6 步：计算计数

打印每条分支每查询注意到的键数及总数。与 `N`（全注意力）比较。在 1024 token 的合成数据上，取 `l = 32, k = 4, w = 128`，NSA 每查询看到 `32 + 128 + 128 = 288` 个键，而全注意力是 1024 个——少 3.5 倍。

## 使用它

NSA 已在 DeepSeek 自家的长上下文预训练 pipeline 中出货。截至 2026 年 4 月公开推理栈中的集成状态：

- **DeepSeek 内部**：原生支持，公开权重使用 NSA 或其继任者 DSA（Deepseek Sparse Attention）。
- **vLLM**：针对 DeepSeek-V3.x 权重的实验性 NSA 支持正在开发中。
- **SGLang**：已发布 NSA 基准，生产路径跟随 vLLM。
- **llama.cpp / CPU**：不支持；在 CPU 吞吐量下，kernel 分解的开销不划算。

何时该用 NSA：

- 面向 64k 以上上下文、有严肃计算预算的预训练或继续训练。
- 推理 DeepSeek 自己的长上下文 checkpoint。这些权重是 NSA 原生的。

何时不该用：

- 服务一个现有的稠密注意力预训练模型。不继续训练就无法改装 NSA。
- 上下文低于 16k。三分支开销超过节省。
- batch-1 交互式聊天。延迟敏感的解码会有收益，但仅在长上下文时。

## 交付它

本课产出 `outputs/skill-nsa-integrator.md`。给定长上下文预训练运行规格，它生成一份 NSA 集成计划：压缩块大小、top-k、滑动窗口、门控 MLP 宽度、kernel 选择，以及能证明架构改动合理的具体长上下文评估。

## 练习

1. 在 1024 token 的合成数据上运行 `code/main.py`。在三个预设上扫描 `(l, k, w)` 并打印计算计数。找出在 needle-in-haystack 测试中相对全注意力保持 95% 召回的前提下，每查询键数最低的预设。

2. 把均值池化压缩器换成小型学习 MLP（2 层，隐藏 32）。在信号为块平均值的合成任务上训练它。在留出数据上测量相对均值池化基线的困惑度差距。

3. 实现门控 MLP。它以查询为输入，输出三个标量。展示门控行为合理：随机查询上接近均匀加权，查询命中靠后的块时选择分支权重大。

4. 计算 128k 上下文下、启用 NSA 的 70B 模型的 KV cache 内存预算。KV 头 8 个，头维度 128，BF16。与全注意力和 MLA 比较（第 10 阶段 · 14 给出了 MLA 的数字）。找出 NSA 细粒度分支 KV cache 等于全注意力的序列长度。

5. 阅读 NSA 论文（arXiv:2502.11089）第 4 节，用三句话解释为什么复用压缩分支的注意力分数做 top-k 选择，而不是单独计算路由分数。把答案与梯度流联系起来。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 压缩分支 | 「粗粒度视图」 | 对块平均键的注意力，以每查询 $O(N/l)$ 个键提供全局上下文 |
| 选择分支 | 「top-k 块」 | 对压缩分支分数最高的 `k` 个块做细粒度注意力 |
| 滑动窗口 | 「局部上下文」 | 对最后 `W` 个 token 的注意力，捕获短程模式 |
| 原生可训练性 | 「带着稀疏性预训练」 | 稀疏模式在预训练中学到，而不是推理时外挂 |
| 压缩块大小 l | 「粗粒度视图的分组大小」 | 多少 token 合并成一个摘要；通常 32-64 |
| Top-k | 「保留的块数」 | 读取其未压缩 token 的压缩块数量；通常 16 |
| 滑动窗口 W | 「局部注意力半径」 | 通常 512；更短损害局部连贯性，更长浪费计算 |
| 分支门控 | 「如何混合三者」 | 逐位置 MLP 输出，为三条分支的贡献加权 |
| 硬件对齐 | 「对 kernel 友好的稀疏性」 | 选择稀疏模式时让实际 GPU kernel 能实现理论加速 |
| DSA | 「NSA 的继任者」 | Deepseek Sparse Attention，DeepSeek 谱系中继 NSA 之后的架构 |

## 延伸阅读

- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) —— 论文本身
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) —— NSA 面向的架构家族
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) —— 同期工作，对块做 MoE 式注意力
- [Beltagy et al. — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) —— 滑动窗口的起源
- [Xiao et al. — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) —— NSA 所改进的推理时稀疏基线
- [Dao et al. — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) —— NSA kernel 在 64k 时击败的全注意力基线
