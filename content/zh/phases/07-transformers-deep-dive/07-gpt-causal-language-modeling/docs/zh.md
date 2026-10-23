# GPT——因果语言建模

> BERT 看两侧，GPT 只看过去。三角掩码是现代 AI 中最有分量的单行代码。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT)
**Time:** ~75 minutes

## 问题

语言模型只回答一个问题：给定前 `t-1` 个 token，token `t` 上的概率分布是什么？用这个信号训练——也就是下一 token 预测——就能得到一个可以逐 token 生成任意文本的模型。

要在一整条序列上端到端地并行训练，就必须让每个位置的预测只依赖更早的位置。否则模型只要偷看答案就能轻松作弊。

因果掩码负责这件事。它就是一张上三角矩阵，元素为 `-inf`，在 softmax 之前加到注意力分数上。经过 softmax，这些位置变成 0。每个位置只能关注自身和更早的位置。而由于你对整条序列一次性施加掩码，一次前向传播就能得到 N 个并行的下一 token 预测。

GPT-1（2018）、GPT-2（2019）、GPT-3（2020）、GPT-4（2023）、GPT-5（2025）、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi——它们都是仅解码器（decoder-only）的因果 Transformer，核心循环相同。把它们区分开的是数据质量、规模与架构改进，以及后训练（SFT、RLHF、DPO 及其后继方法）。

## 概念

![因果掩码产生三角注意力矩阵](../assets/causal-attention.svg)

### 掩码

给定长度为 $N$ 的序列，构造一个 $N \times N$ 的矩阵：

```math
M_{ij} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}
```

在 softmax 之前把 `M` 加到原始注意力分数上。$\exp(-\infty) = 0$，所以被掩码的位置权重为零。注意力矩阵的每一行都只是一段对先前位置的概率分布。

实现成本：一次 `torch.tril()` 调用。计算时间：纳秒级。对整个领域的影响：一切。

### 三角形从何而来

掩码通常被当作补丁，硬加在注意力上。换一个方向做推导，它就不再神秘：注意力是前缀平均的第三次精化，而三角形就是那次平均的循环边界，写成矩阵形式。

**第 1 步——前缀平均。** 对序列最朴素的一种因果汇总：位置 $i$ 取位置 $0 \ldots i$ 的均值。写成循环就是 `out[i] = X[:i+1].mean(0)`。同样的计算可以用一次矩阵乘法完成。取一个由 1 组成的下三角矩阵，将每一行除以其计数，再相乘：

```python
import numpy as np

A = np.tril(np.ones((n, n)))
A = A / A.sum(axis=1, keepdims=True)
out = A @ X
```

$A$ 的第 $i$ 行是 $[1/(i+1), \ldots, 1/(i+1), 0, \ldots, 0]$。对角线上方的零就是因果性。未来并没有被掩掉，未来从来就不在求和里。

**第 2 步——可学习权重。** 均匀平均把每个过去的 token 视为同等重要。用可学习的分数矩阵 `S` 替换全 1 矩阵。此时行不再天然求和为 1，于是改用 softmax 对每行归一化，而不是除以计数。softmax 永远不会输出精确的零，这会破坏因果性——除非未来的分数以 `-inf` 输入，因为 `exp(-inf) = 0`：

```python
def softmax(x, axis):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

S = S + np.triu(np.full((n, n), -np.inf), k=1)
A = softmax(S, axis=1)
out = A @ X
```

同样的三角形，同样的行随机矩阵，同样一次矩阵乘法。`-inf` 掩码不是什么新机制。它就是第 1 步里的零元素，翻译到了 softmax 的输入域。

**第 3 步——内容相关权重。** 在第 2 步中，`S` 在训练后是固定的：位置 7 对位置 3 的权重始终一样，无论 token 内容是什么。让分数依赖于 token 本身：`S = Q @ K.T / sqrt(d_k)`。其余一切不变。掩码、softmax、矩阵乘法——完全相同。

三个阶段，一个不变量：一个下三角行随机矩阵乘以序列。均匀平均、学习的静态权重、内容相关权重。掩码从来不是加到注意力上的新机制，它从平均里保留了下来。

```figure
mask-derivation
```

### 并行训练，串行推理

训练：对整个 `(N, d_model)` 序列做一次前向传播，计算 N 个交叉熵损失（每个位置一个），求和，反向传播。沿序列方向完全并行。这就是 GPT 训练可扩展的原因——一个 batch 里 100 万 token，一块 GPU 一次前向就能处理完。

推理：逐 token 生成。输入 `[t1, t2, t3]`，得到 `t4`。输入 `[t1, t2, t3, t4]`，得到 `t5`。输入 `[t1, t2, t3, t4, t5]`，得到 `t6`。KV cache（第 12 课）保存 `t1…tn` 的隐藏状态，避免每一步重算。但推理时的串行深度等于输出长度。这就是自回归的代价，也是解码成为每个 LLM 延迟瓶颈的原因。

### 损失——错一位

给定 token `[t1, t2, t3, t4]`：

- 输入：`[t1, t2, t3]`
- 目标：`[t2, t3, t4]`

对每个位置 `i`，计算 `-log P(target_i | inputs[:i+1])`，再求和。这就是整条序列的交叉熵。

你听说过的每一个 Transformer 语言模型都用这个损失训练。预训练、微调、SFT——损失相同，数据不同。

### 解码策略

训练之后，采样策略的重要性超出很多人的想象。

| 方法 | 作用 | 适用场景 |
|--------|--------------|-------------|
| 贪心 | 每一步取 argmax | 确定性任务、代码补全 |
| 温度 | logit 除以 T 后采样 | 创意任务，T 越高越多样 |
| Top-k | 只从前 k 个 token 中采样 | 砍掉低概率长尾 |
| Top-p（核采样） | 从累积概率 ≥ p 的最小集合中采样 | 2020 年后的默认方案；能适应分布形状 |
| Min-p | 保留 `p > min_p * max_p` 的 token | 2024 年起流行；比 top-p 更能拒绝长尾 |
| 投机解码 | 草稿模型提议 N 个 token，大模型验证 | 同等质量下延迟降低 2–3 倍 |

2026 年，min-p 加 temperature 0.7 是开放权重模型的合理默认设置。投机解码则是任何生产推理栈的标配。

### 「GPT 配方」为何奏效

1. **仅解码器。** 没有编码器开销。每层只做一遍注意力加 FFN。
2. **扩展。** 124M → 1.5B → 175B → 数万亿参数。Chinchilla 缩放定律（第 13 课）告诉你如何分配算力。
3. **上下文学习。** 大约在 6B–13B 参数时涌现。模型无需微调即可遵循少样本示例。
4. **RLHF。** 基于人类偏好的后训练，把原始预训练文本模型变成聊天助手。
5. **Pre-norm + RoPE + SwiGLU。** 大规模训练更稳定。

核心架构自 GPT-2 以来变化不大。真正有趣的事都发生在数据、规模和后训练上。

```figure
causal-mask
```

## 构建它

### 第 1 步：因果掩码

见 `code/main.py`。一行代码：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

在 softmax 之前把它加到注意力分数上。机制就这么简单。

### 第 2 步：一个两层、类 GPT 的模型

堆叠两个解码器块（掩码自注意力 + FFN，无交叉注意力）。加上 token embedding、位置编码和一个反嵌入（unembedding，与 token embedding 矩阵共享参数——GPT-2 以来的标准技巧）。

### 第 3 步：端到端的下一 token 预测

在一个 20 token 的玩具词表上，在每个位置产生 logit。针对错一位的目标计算交叉熵损失。不做梯度——这是一次前向传播的健全性检查。

### 第 4 步：采样

实现贪心、温度、top-k、top-p、min-p 采样。在固定 prompt 上分别运行并比较输出。一个采样函数只要 10 行。

## 使用它

PyTorch 2026 惯用写法：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

底层上，`generate()` 执行前向传播，取出最后位置的 logit，采样下一个 token，追加，然后重复。每个生产级 LLM 推理栈（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都以同样的循环为基础并做了大量优化——批量 prefill、连续批处理、KV cache 分页、投机解码。

**GPT 与 BERT，各用一句话：** GPT 预测 $P(x_t \mid x_{<t})$。BERT 预测 $P(x_{\text{masked}} \mid x_{\text{unmasked}})$。损失函数决定了模型能否生成。

## 交付它

见 `outputs/skill-sampling-tuner.md`。该 skill 为新的生成任务选择采样参数，并在需要确定性解码时给出提示。

## 练习

1. **简单。** 运行 `code/main.py`，验证 softmax 之后的因果注意力矩阵是下三角矩阵。抽查：第 3 行的权重应只出现在第 0–3 列。
2. **中等。** 实现宽度为 4 的束搜索。在 10 个短 prompt 上比较 beam-4 与贪心的困惑度。束搜索总是赢吗？（提示：翻译任务通常是，开放式对话则不然。）
3. **困难。** 实现投机解码：用 2 层的小模型做草稿模型，6 层模型做验证模型。在 100 条长度为 64 的补全上测量实际耗时加速比。确认输出与验证模型的贪心输出一致。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 因果掩码 | 「那个三角形」 | 加到注意力分数上的上三角 `-inf` 矩阵，使位置 `i` 只能看到位置 `≤ i`。 |
| 下一 token 预测 | 「那个损失」 | 在每个位置上，模型分布与真实下一 token 之间的交叉熵。 |
| 自回归 | 「一个一个生成」 | 把输出作为输入喂回；并行只存在于训练阶段，生成阶段没有。 |
| Logit | 「softmax 之前的分数」 | 语言模型头在 softmax 之前的原始输出；采样就在这些值上进行。 |
| 温度 | 「创意旋钮」 | logit 除以 $T$；$T \to 0$ 等价于贪心，$T \to \infty$ 等价于均匀分布。 |
| Top-p | 「核采样」 | 把分布截断到累积和 $\ge p$ 的最小集合；从剩余部分采样。 |
| Min-p | 「比 top-p 更好」 | 保留 `p ≥ min_p × max_p` 的 token；截断阈值随分布的尖锐程度自适应。 |
| 投机解码 | 「草稿 + 验证」 | 廉价模型提议 N 个 token；大模型并行验证。 |
| 教师强制 | 「训练技巧」 | 训练时喂入真实的上一 token，而非模型自己的预测。每个 seq2seq 语言模型的标准做法。 |

## 延伸阅读

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) —— GPT-1。
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) —— GPT-2。
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) —— GPT-3 与上下文学习。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) —— 投机解码论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) —— 权威的因果语言模型参考代码。
