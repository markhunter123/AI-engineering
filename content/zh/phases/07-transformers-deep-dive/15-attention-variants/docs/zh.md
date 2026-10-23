# 注意力变体——滑动窗口、稀疏、差分

> 全注意力是一个圆：每个 token 都看见每个 token，内存为此买单。四种变体弯曲这个圆的形状，收回一半成本。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head), Phase 7 · 12 (KV Cache / Flash Attention)
**Time:** ~60 minutes

## 问题

全注意力在序列长度上的内存是 $O(N^2)$，计算也是 $O(N^2)$。对于 128K 上下文的 Llama 3 70B，这意味着每层 160 亿个注意力条目，再乘以 80 层。Flash Attention（第 12 课）隐藏了 $O(N^2)$ 的激活内存，但不改变算术成本——每个 token 仍然要关注其他所有 token。

三类变体改变了注意力矩阵本身的拓扑：

1. **滑动窗口注意力（SWA）。** 每个 token 只关注固定窗口内的邻居，而不是完整前缀。内存和计算降到 $O(N \cdot W)$，其中 $W$ 是窗口大小。Gemma 2/3、Mistral 7B 的前几层、Phi-3-Long 使用这一方案。
2. **稀疏 / 块注意力。** 只有选定的 `(i, j)` 对被打分，其余强制为零权重。Longformer、BigBird、OpenAI 稀疏 Transformer。
3. **差分注意力。** 用独立的 Q/K 投影计算两张注意力图，再相减。消除把权重渗到最初几个 token 的「注意力汇」。微软的 DIFF Transformer（2024）。

这些变体可以共存。2026 年的前沿模型常常混用：大多数层是 SWA-1024，每第五层是全局全注意力，还有少数差分头负责清理检索。Gemma 3 的 5:1 SWA 对全局比例是目前教科书式的默认。

## 概念

### 滑动窗口注意力（SWA）

位置 $i$ 的每个 query 只关注 $[i - W, i]$（因果 SWA）或 $[i - W/2, i + W/2]$（双向）内的位置。窗口之外的 token 在分数矩阵中为 $-\infty$。

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

对于 $N = 8192$、$W = 1024$，分数矩阵在期望上约有 1024 × 8192 个非零条目——缩减为原来的 1/8。

**KV cache 随 SWA 缩小。** 每层只需保留 K 和 V 的最后 $W$ 个 token。对于 Gemma-3 级别的配置（1024 窗口、128K 上下文），KV cache 缩减 128 倍。

**质量代价。** 纯 SWA 的 Transformer 在长距离检索上表现不佳。解决办法：把 SWA 层与全注意力层交错。Gemma 3 使用 5:1 的 SWA:global。Mistral 7B 使用因果 SWA 堆叠，信息通过重叠窗口「向前流动」——每层把有效感受野扩展 $W$，经过 `L` 层后模型能回看 `L × W` 个 token。

### 稀疏 / 块注意力

提前选定一个 `N × N` 的稀疏模式。三种经典形状：

- **局部 + 步进（OpenAI 稀疏 Transformer）。** 关注最后 $W$ 个 token，加上之前每隔 `stride` 个的 token。以 $O(N \cdot \sqrt{N})$ 的计算同时捕捉局部和长程信息。
- **Longformer / BigBird。** 局部窗口 + 少量全局 token（例如 `[CLS]`），它们关注所有人、也被所有人关注 + 随机稀疏连接。在同等质量下实测上下文翻倍。
- **原生稀疏注意力（Native Sparse Attention，DeepSeek，2025）。** 学习哪些 `(Q, K)` 块重要；在内核层面跳过零块。兼容 FlashAttention。

稀疏注意力是内核工程的故事。数学很简单（掩蔽分数矩阵）；收益来自永远不把零条目加载进 SRAM。FlashAttention-3 和 2026 年的 FlexAttention API 让自定义稀疏模式在 PyTorch 中成为一等公民。

### 差分注意力（DIFF Transformer，2024）

常规注意力有「注意力汇」问题：softmax 强制每行求和为 1，因此不想关注任何特定内容的 token 会把权重倾倒到第一个 token（或前几个）上。这偷走了本应分配给真实内容的容量。

差分注意力通过计算**两张**注意力图并相减来修复：

```math
\begin{aligned}
A_1 &= \mathop{\mathrm{softmax}}(Q_1 K_1^T / \sqrt{d}) \\
A_2 &= \mathop{\mathrm{softmax}}(Q_2 K_2^T / \sqrt{d}) \\
\text{DiffAttn} &= (A_1 - \lambda \cdot A_2) V
\end{aligned}
```

其中 $\lambda$ 是可学习的标量（通常 0.5–0.8）。A1 捕捉真实内容权重；A2 捕捉汇。相减抵消汇，把权重重新分配给相关 token。

报告结果（微软 2024）：困惑度降低 5–10%，相同训练长度下有效上下文延长 1.5–2 倍，大海捞针检索更锐利。

### 变体对比

| 变体 | 计算 | KV cache | 相对全注意力的质量 | 生产使用 |
|---------|---------|----------|-----------------|----------------|
| 全注意力 | $O(N^2)$ | 每层 $O(N)$ | 基线 | 所有模型的默认层 |
| SWA（窗口 1024） | $O(N \cdot W)$ | 每层 $O(W)$ | -0.1 ppl，配合全局层表现好 | Gemma 2/3、Phi-3-Long |
| 局部 + 步进稀疏 | $O(N \cdot \sqrt{N})$ | 混合 | 与 SWA 相近 | OpenAI 稀疏 Transformer、Longformer |
| BigBird（局部 + 全局 + 随机） | 约 $O(N)$ | 混合 | 2 倍上下文下匹配全注意力 | 早期长上下文 BERT |
| 原生稀疏（DeepSeek-V3.2） | $O(N \cdot \text{active fraction})$ | $O(N)$ | 差距在 0.05 ppl 内 | DeepSeek-V3.2，2025 |
| 差分 | $O(2 \cdot N^2)$ | $O(2N)$ | ppl 降低 5–10% | DIFF Transformer、2026 年初的模型 |

```figure
gqa-kv-sharing
```

## 构建它

见 `code/main.py`。我们实现一个因果掩码对比器，在玩具序列上并排展示全注意力、SWA、局部+步进和差分注意力。

### 第 1 步：完整因果掩码（基线）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

第 07 课的基线。下三角；对角线以上为零权重。

### 第 2 步：滑动窗口因果掩码

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

一个参数——`window`。当 `window >= n` 时，恢复完整因果注意力。当 `window = 1` 时，每个 token 只关注自身。

### 第 3 步：局部 + 步进稀疏掩码

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

密集局部窗口，加上每隔 `stride` 个、一直回溯到序列开头的 token。感受野随层数按 log 步长增长。

### 第 4 步：差分注意力

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次注意力计算，用可学习的混合系数相减。代码中我们对比单注意力与差分注意力的注意力汇热图，观察汇的坍缩。

### 第 5 步：KV cache 大小

在 $N = 131072$ 时打印每种变体每层的缓存大小。SWA 和稀疏变体减少 10–100 倍。差分翻倍。有意识地支付你的内存账单。

## 使用它

2026 年的生产模式：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 的 FlexAttention 接受一个掩码函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

这会编译成自定义 Triton 内核。对常见模式，速度在 FlashAttention-3 的 10% 以内，而掩码函数是 Python 可调用对象。

**如何选择：**

- **纯全注意力**——上下文到约 16K 的每一层，或检索质量至关重要时。
- **SWA + 全局混合**——长上下文（>32K）、训练和推理受内存限制时。32K 以上的 2026 年默认。
- **稀疏块注意力**——自定义内核、自定义模式。留给专门的工作负载（检索、音频）。
- **差分注意力**——任何注意力汇污染会造成伤害的工作负载（长上下文 RAG、大海捞针）。

## 交付它

见 `outputs/skill-attention-variant-picker.md`。该技能根据目标上下文长度、检索需求和训练/推理算力画像，为新模型选择注意力拓扑。

## 练习

1. **简单。** 运行 `code/main.py`。验证 `window=4` 的 SWA 会把每行最后 4 个 token 之外的全部置零。验证 `window=n` 逐位重现完整因果注意力。
2. **中等。** 在第 07 课项目的基础上实现 `window=1024` 的因果 SWA。在 tinyshakespeare 上训练 1,000 步。验证损失相对全注意力退化多少？峰值内存下降多少？
3. **困难。** 在项目模型中实现 Gemma-3 风格的 5:1 层混合（5 层 SWA，1 层全局）。在相同参数下比较损失、内存和生成质量，对照纯 SWA 和纯全局基线。
4. **困难。** 实现每个头带可学习 $\lambda$ 的差分注意力。在合成检索任务上训练（一根针，2,000 个干扰项）。在相同参数下测量检索准确率，对照单注意力基线。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 滑动窗口注意力（SWA） | 「局部注意力」 | 每个 query 只关注最近 $W$ 个 token；KV cache 缩小到 $O(W)$。 |
| 有效感受野 | 「模型能看到多远」 | 窗口为 $W$ 的 $L$ 层 SWA 堆叠中，最多 $L \times W$ 个 token。 |
| Longformer / BigBird | 「局部 + 全局 + 随机」 | 带少量始终参与注意力的全局 token 的稀疏模式；早期长上下文方法。 |
| 原生稀疏注意力 | 「DeepSeek 的内核技巧」 | 学习块级稀疏性；在内核层面跳过零块，同时保持质量。 |
| 差分注意力 | 「两张图，相减」 | DIFF Transformer：从第一张注意力图减去可学习 $\lambda$ 倍的第二张，以抵消注意力汇。 |
| 注意力汇 | 「权重渗向 token 0」 | Softmax 归一化强制每行求和为 1；无信息的 query 会把权重倒在位置 0。 |
| FlexAttention | 「掩码即 Python」 | PyTorch 2.5+ API，把任意掩码函数编译成 FlashAttention 形状的内核。 |
| 层类型混合 | 「5:1 SWA 对全局」 | 在堆叠中交错稀疏层与全注意力层，以更低内存保持质量。 |

## 延伸阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) —— 滑动窗口 + 全局 token 的经典论文。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) —— 局部 + 全局 + 随机。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) —— OpenAI 的局部+步进模式。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) —— 1:1 的 SWA:global 混合。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) —— window=1024 的 5:1 混合，如今已是教科书式默认。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) —— DIFF Transformer 论文。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) —— DeepSeek-V3.2 的学习型稀疏注意力。
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) —— 「使用它」中掩码即可调用对象模式的 API 参考。
