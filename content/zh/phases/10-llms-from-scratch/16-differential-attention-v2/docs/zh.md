# 差分注意力（V2）

> Softmax 注意力会把少量概率摊到每个不匹配的 token 上。在 100k token 的上下文中，这些噪声累积起来会淹没信号。Differential Transformer（Ye 等，ICLR 2025）用两个 softmax 之差来计算注意力，减去共享的噪声底，从而修复这一点。DIFF V2（微软，2026 年 1 月）是生产栈的重写：解码延迟与基线 Transformer 持平，无需自定义 kernel，兼容 FlashAttention。本课完整讲解 V1 到 V2，并给出一个可在 stdlib Python 中运行的差分运算玩具实现。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 02 (self-attention), Phase 7 · 15 (attention variants), Phase 10 · 14 (architecture walkthrough)
**Time:** ~60 minutes

## 学习目标

- 准确说明 softmax 注意力为何存在噪声底，以及它为何随上下文长度增长。
- 推导差分注意力公式，解释为何相减能抵消共享的噪声成分而保留信号。
- 梳理 V1 到 V2 的差异：什么变快了、什么变简单了、什么更稳定了，以及每项改动为何是生产预训练所必需的。
- 用纯 Python 从零实现差分注意力，并在合成的信号加噪声查询上实证验证噪声抵消特性。

## 问题

标准 softmax 注意力有一个数学性质，在规模上会变成运维噩梦。对查询 `q`，注意力权重是 `softmax(qK^T / sqrt(d))`。Softmax 永远不会产生精确的零——每个不匹配的 token 都会分到一些正质量。这些残余质量就是噪声，并且随上下文长度增长。在 128k token 时，即使每个不匹配 token 只分到 0.001% 的概率，127,999 个加起来也贡献了总量约 12%。模型不得不学会绕过这个随上下文增长的噪声底。

经验上，这表现为注意力头之间的干扰：长上下文 RAG 中幻觉的引用、100k token 检索任务上的「中间迷失」（lost-in-the-middle）失败，以及超过 32k 后 needle-in-haystack 基准上细微的准确率下降。Differential Transformer 论文（arXiv:2410.05258，ICLR 2025）测量了差距：DIFF Transformer 比同尺寸基线困惑度更低、长上下文准确率更高、幻觉更少。

DIFF V1 有三个问题，使它无法进入前沿预训练流水线。它的 value cache 在每次解码步要加载两次；它需要自定义 CUDA kernel，破坏了 FlashAttention 兼容性；它的逐头 RMSNorm 在 70B 以上规模破坏长期训练的稳定性。DIFF V2（微软 unilm 博客，2026 年 1 月 20 日）修复了全部三点。本课讲解两个版本，构建差分算子，并在玩具查询上基准测试噪声抵消。

## 概念

### softmax 的噪声底

对查询 `q` 和键 `K = [k_1, ..., k_N]`，注意力权重为：

```math
w_i = \frac{\exp(q \cdot k_i / \sqrt{d})}{\sum_j \exp(q \cdot k_j / \sqrt{d})}
```

没有哪个 `w_i` 会为零。如果 `k_i` 与 `q` 完全无关，得分 `q . k_i` 也不是 0——它在零附近以方差 `||q||^2 / d` 波动。经过 softmax 归一化后，每个无关 token 仍向加权和贡献 `O(1/N)`。无关 token 的总贡献是 `O((N-1)/N) = O(1)`——不是小量。

模型想要的是类似硬 top-k 的东西：匹配 token 上权重高，其他地方接近零。Softmax 太光滑，无法直接做到。

### 差分思想

把每个头的 Q 和 K 投影分成两份：$Q = (Q_1, Q_2)$，$K = (K_1, K_2)$。计算两个注意力图：

```math
\begin{aligned}
A_1 &= \mathop{\mathrm{softmax}}(Q_1 K_1^T / \sqrt{d}) \\
A_2 &= \mathop{\mathrm{softmax}}(Q_2 K_2^T / \sqrt{d})
\end{aligned}
```

输出：

```math
\mathop{\mathrm{DiffAttn}} = (A_1 - \lambda \cdot A_2) V
```

相减会抵消两个图共享的任何噪声分布。如果两个图在 127k 个无关 token 上都有大致均匀的权重（随机初始化时必然如此），它们会相互抵消。信号——少数真正相关 token 上的尖峰权重——只有在两个图中以相同强度出现时才会被抵消，而模型训练后不会这样。

`lambda` 是每个头可学习的标量，参数化为 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以为负。`lambda_init` 默认为一个小的正数，比如 0.8。

### 为什么这类似于耳机降噪

想象两个有噪声的麦克风录同一个声音。两者都拾取说话人加上相关的背景噪声。用一个减去另一个，共享噪声就消失了。声音得以保留，因为两个信号在相位或幅度上的差异足以避免完全抵消。逐头的 `lambda` 学的正是这个平衡。

### V1 vs V2：差异

V1 保持参数量与基线 Transformer 相等。为了每个头得到两个查询，它把头维度减半。这牺牲了头的表达能力，更痛苦的是，每个头的 value cache 也减半。解码时每步必须加载 value cache 两次（每个 softmax 分支一次）。结果：尽管参数量持平，解码却比基线慢。

V2 把查询头数量翻倍，KV 头保持不变（参数从 up-projection 借来）。头维度与基线相同。相减之后，额外维度被投影回与基线 Transformer 的 $O_W$ 投影匹配。三件事同时发生：

1. 解码速度与基线持平（KV cache 只加载一次）。
2. FlashAttention 无需改动即可运行（无自定义 kernel）。
3. 解码时的算术强度提高（每字节 HBM 加载对应更多计算）。

V2 还去掉了 V1 用来稳定相减的逐头 RMSNorm。在 70B 级预训练规模下，那个 RMSNorm 会破坏训练后期稳定性。V2 用一个更简单的初始化方案替代，无需额外模块就能保持训练稳定。

### 何时使用它

| 负载 | 收益 |
|----------|---------|
| 长上下文 RAG（64k+） | 更干净的注意力图，更少幻觉引用 |
| Needle-in-haystack 基准 | 超过 32k 后准确率显著提升 |
| 多文档问答 | 跨文档干扰更少 |
| 8k 代码补全 | 边际收益，不值得改动架构 |
| 短聊天（< 4k） | 与基线基本无法区分 |

收益随上下文长度增长。在 4k token 时噪声底足够小，标准注意力就够用。到 128k 时它就在伤害你。

### 与其他 2026 年选项如何叠加

| 特性 | 与 DIFF V2 兼容？ |
|---------|------------------------|
| GQA | 是（V2 增加 Q 头，不增加 KV 头） |
| MLA（DeepSeek） | 原则上可以，尚无结合两者的公开论文 |
| MoE | 是（注意力与 MLP 块独立） |
| RoPE | 是（不变） |
| YaRN / 长上下文扩展 | 是（正是 DIFF 帮助最大的地方） |
| FlashAttention | V2 中是（V1 中否） |
| 投机解码 | 是（注意力改动对投机解码循环不可见） |

```figure
differential-attention
```

## 构建它

`code/main.py` 用纯 Python 实现差分注意力。一个具有已知信号加噪声结构的玩具查询，让你可以直接测量噪声抵消比。

### 第 1 步：标准 softmax 注意力

Stdlib 矩阵运算：列表的列表、手写矩阵乘法、带最大值数值稳定减法的 softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 第 2 步：把 Q、K 分成两半

V1 风格：头维度减半。V2 风格：保持头维度，把头数量翻倍。玩具实现为教学清晰采用 V1——数学完全相同，只是记账不同。

### 第 3 步：两个 softmax 分支 + 相减

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：输出权重可以为负。这没问题——value cache 仍能处理带符号的贡献。后续的 V 投影会吸收符号。

### 第 4 步：噪声抵消测量

构造长度为 1024 的合成序列。把信号 token 放在已知位置，其余用噪声填充。计算 (a) 标准 softmax 注意力在信号位置上的权重，以及 (b) 差分注意力权重。测量各自的信噪比。DIFF 注意力可靠地产生更高的信噪比，幅度为 3x-10x，取决于两个分支被训练得有多不同。

### 第 5 步：V1 vs V2 参数核算

给定配置（hidden=4096, heads=32, d_head=128），打印：

- 基线 Transformer：Q、K、V 各为 `hidden * hidden`，MLP 为 4 * hidden。
- DIFF V1：Q、K 各为 `hidden * hidden`，V 为 `hidden * hidden`（不变），头维度内部减半。增加逐头 `lambda` 参数 $(O(\text{heads} \cdot d_{\text{head}}))$。
- DIFF V2：Q 为 `2 * hidden * hidden`，K 为 `hidden * hidden`，V 为 `hidden * hidden`。额外维度在 $O_W$ 之前投影回原尺寸。增加同样的 `lambda` 参数。

玩具实现测量 V2 的额外参数成本（每个注意力块大约多 `hidden * hidden`）并打印出来。

## 使用它

截至 2026 年 4 月，DIFF V2 尚未在所有生产推理服务器中出货，但 vLLM 和 SGLang 的集成正在进行。与此同时，这种模式出现在：

- 微软内部的长上下文生产模型。
- 若干面向 256k 以上上下文的开源模型训练中的研究复现。
- 把 DIFF 注意力与滑动窗口注意力在交替层上结合的混合架构。

2026 年何时该用它：

- 从零训练一个面向 64k 以上有效上下文的新模型。从一开始就加入差分注意力；之后再重训代价高昂。
- 微调一个长上下文模型，而「中间迷失」失败主导了你的评估。Q 投影上的 LoRA 可以近似 DIFF 结构。

何时不该用：

- 你在服务一个长上下文性能稳定的预训练稠密模型。重训成本很少能在现有权重上收回。
- 你的上下文始终低于 16k。噪声底可以忽略。

## 交付它

本课产出 `outputs/skill-diff-attention-integrator.md`。给定模型架构、目标上下文长度、幻觉画像和训练预算，它生成一份集成计划，说明如何在新的预训练或 LoRA 微调中加入差分注意力。

## 练习

1. 运行 `code/main.py`。验证差分注意力在合成查询上报告的信噪比高于标准 softmax 注意力。改变噪声幅度，展示标准注意力变得不可用的交叉点。

2. 计算 7B 级模型（hidden=4096, heads=32, d_head=128, 32 层）从基线到 DIFF V1、以及从基线到 DIFF V2 的参数数量差。说明哪些组件增加了参数，哪些保持不变。

3. 阅读 DIFF V1 论文（arXiv:2410.05258）第 3 节和 DIFF V2 的 Hugging Face 博客第 2 节。用两句话解释为什么 V1 的逐头 RMSNorm 是必要的，以及为什么 V2 能去掉它而不导致训练发散。

4. 实现一个消融：用 `lambda = 0`（纯第一个 softmax）和 `lambda = 1`（完全相减）计算差分注意力。在合成查询上，测量信号随扫描的变化。找出使信噪比最大的 `lambda`。

5. 把玩具扩展到 GQA + DIFF V2。选择 8 个 KV 头和 32 个 Q 头。证明 KV cache 大小与相同 (8, 32) 配置的基线 GQA 模型一致。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 差分注意力 | 「两个 softmax 相减」 | 把 Q、K 分成两半，计算两个 softmax 图，从第一个中减去第二个（按 $\lambda$ 缩放），再乘以 V |
| 噪声底 | 「softmax 的非零尾部」 | softmax 给每个无关 token 的 $O(1/N)$ 权重，在长上下文中累加为 $O(1)$ |
| lambda | 「相减的缩放系数」 | 逐头可学习标量，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`，可为负 |
| DIFF V1 | 「ICLR 2025 的版本」 | 原始 Differential Transformer；头维度减半以保持参数量，需要自定义 kernel，解码更慢 |
| DIFF V2 | 「2026 年 1 月的修复」 | Q 头翻倍而 KV 头不变；解码速度与基线持平，兼容 FlashAttention |
| 逐头 RMSNorm | 「V1 的稳定器」 | V1 在差分后施加的额外归一化；V2 去掉它以防止训练后期不稳定 |
| 信噪比 | 「有多少注意力被浪费」 | 真实信号位置上的权重与无关位置平均权重之比 |
| 中间迷失 | 「长上下文失败模式」 | 文档位于长上下文中间时检索准确率下降的经验现象——DIFF 注意力能减轻它 |
| 算术强度 | 「每字节加载对应的 FLOPs」 | V2 通过让每次 KV 加载服务更多查询来提高解码时的该比值；对内存受限的解码很重要 |

## 延伸阅读

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) —— 原始论文，包含噪声抵消理论和长上下文消融
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) —— 生产栈重写，解码与基线持平，兼容 FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) —— 分析相减为何能恢复预训练注意力结构的理论
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) —— 参数共享变体
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) —— DIFF 从中相减的基线 Transformer
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) —— DIFF 注意力针对的长上下文基准
