# Flamingo 与面向少样本 VLM 的门控交叉注意力

> DeepMind 的 Flamingo（2022）比任何人都更早做到了两件事。它表明单个模型可以处理任意交错的图像、视频和文本序列。它还表明 VLM 可以在上下文中学习——给出包含三个（图像，标题）示例的少样本 prompt，模型就能为一张新图像生成标题，无需任何梯度步骤。机制是：门控交叉注意力层，插入冻结 LLM 现有层之间，带一个从零开始的可学习 tanh 门，因此 LLM 的文本能力在初始化时得以保留。本节课走一遍 Flamingo 的 Perceiver resampler 和门控交叉注意力架构——Gemini 交错输入与 Idefics2 视觉 token 的祖先。

**Type:** Learn
**Languages:** Python (stdlib, gated cross-attention + Perceiver resampler demo)
**Prerequisites:** Phase 12 · 03 (BLIP-2 Q-Former)
**Time:** ~120 minutes

## 学习目标

- 解释门控交叉注意力如何通过 tanh(gate) = 0 在初始化时保留冻结 LLM 的文本能力。
- 走一遍 Perceiver resampler：N 个图像 patch → 通过交叉注意力得到 K 个固定「latent」query。
- 描述 Flamingo 如何处理交错图像-文本序列，并用尊重图像位置的因果掩码。
- 复现一个少样本多模态 prompt 结构（3 个图像-标题示例，然后是查询图像）。

## 问题

BLIP-2 把 32 个视觉 token 喂进冻结 LLM 的输入层。每个 prompt 一张图能行。但如果你想喂*许多*图像并与文本交错，比如「这是图像 A，给它写标题；这是图像 B，给它写标题；现在这是图像 C，给它写标题」呢？LLM 的自注意力需要在一个流中处理图像 token 和文本 token，而且哪些位置能关注哪些图像会变得很棘手。

Flamingo 的回答：完全不动 LLM 的输入流。在现有 LLM 块之间插入额外的交叉注意力层。文本 token 照常流经 LLM 的因果自注意力。每隔几个 LLM 块，文本 token 还会通过新的门控层交叉关注图像特征。门（初始化为零）意味着在第 0 步新层是空操作——模型的行为与预训练 LLM 完全一致。随着训练推进，门打开，视觉信息开始流入。

Flamingo 回答的第二个问题：如何处理每个 prompt 中数量可变的图像（0 张、1 张或多张）？Perceiver resampler——一个小型交叉注意力模块，接收任意数量的 patch，产出固定数量的视觉 latent token。无论 prompt 中有多少张图像，LLM 交叉注意力层看到的形状都相同。

## 概念

### 冻结的 LLM

Flamingo 从冻结的 Chinchilla 70B LLM 开始。全部 70B 权重不动。现有的文本自注意力和 FFN 正常运作。

### Perceiver 重采样器

对 prompt 中的每张图像，ViT 产出 $N$ 个 patch token。Perceiver resampler 有 $K$ 个固定的可学习 latent（Flamingo 用 $K=64$）。每个 resampler 块包含两个子步骤：

1. 交叉注意力：$K$ 个 latent 关注 $N$ 个 patch token（Q 来自 latent，K/V 来自 patch）。
2. 在 latent 内部做自注意力 + FFN。

经过 6 个 resampler 块后，输出是 $K=64$ 个维度为 1024 的视觉 token，无论 ViT 产出多少 patch。224x224 图像（196 个 patch）和 480x480 图像（900 个 patch）都输出 64 个 resampler token。

对于视频，resampler 在时间维度上应用：每一帧的 patch 产出 64 个 latent，再加一个时间位置编码，让模型区分 $t=0$ 与 $t=N$。整段视频变成 $T \cdot 64$ 个视觉 token。

### 门控交叉注意力

在冻结 LLM 的每 $M$ 层之间（Flamingo 取 $M=4$），插入一个新的门控交叉注意力块：

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是可学习标量，初始化为零。
- `tanh(0) = 0`，所以初始化时门控分支贡献为零。
- 随着 `alpha` 远离零，交叉注意力的贡献平滑增长。
- 残差连接意味着即使门完全打开，也不会覆盖 LLM 的文本表示；它只是在上面叠加视觉信息。

这是 Flamingo 最重要的设计选择：视觉条件化是可加的、门控的，且在初始化时为零。第 0 步的 Flamingo 在纯文本输入上就是一个完美的 Chinchilla 70B。

### 面向交错输入的掩码交叉注意力

在「<image A> caption A <image B> caption B <image C> ?」这样的 prompt 中，每个文本 token 只应看到序列中位于它之前的图像。交叉注意力掩码强制：位置 `t` 的文本 token 只关注图像索引 `i < i_t` 的 resampler token，其中 `i_t` 是位置 `t` 之前最近的那张图像。「只看最后一张前置图像」和「看所有前置图像」都是有效选择；Flamingo 选了前者。

### 上下文少样本学习

Flamingo 的 prompt 长这样：

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

模型看到补全模式，输出「bird」（或 image3 实际展示的东西）。没有梯度步骤。冻结 LLM 的上下文学习能力通过门控交叉注意力延续下来——这是论文的点睛之笔，也是它重要的原因。

### 训练数据

Flamingo 在三个数据集上训练：

1. MultiModal MassiveWeb（M3W）：4300 万个带交错图像和文本的网页，按阅读顺序重建。
2. Image-Text Pairs（ALIGN + LTIP）：44 亿对。
3. Video-Text Pairs（VTP）：2700 万段短视频。

OBELICS（2023）是交错网页语料的开放复现，Idefics、Idefics2 和大多数开放的「类 Flamingo」模型都在它上面训练。

### OpenFlamingo 与 Otter

OpenFlamingo（2023）是开放复现。架构相同（Perceiver resampler + 冻结 LLaMA 或 MPT 上的门控交叉注意力）。checkpoint 有 3B、4B、9B。由于基座 LLM 更小、数据更少，质量落后于 Flamingo。

Otter（2023）在 OpenFlamingo 基础上用 MIMIC-IT（一个多模态指令数据集）做指令微调，表明门控交叉注意力同样适用于指令遵循。

### 后代们

- Idefics / Idefics2 / Idefics3：Hugging Face 的门控交叉注意力谱系，逐步简化（Idefics2 放弃 resampler，改用带自适应池化的直接 patch token）。
- Flamingo 到 Chameleon 的过渡：到 2024 年，许多团队转向早期融合（第 12.11 课）；当必须冻结骨干时，Flamingo 式门控交叉注意力仍在生产中使用。
- Gemini 的交错输入：概念上继承了 Flamingo 交错格式的灵活性，尽管具体机制是专有的。

### 与 BLIP-2 的比较

| | BLIP-2 | Flamingo |
|---|---|---|
| 视觉桥梁 | 输入处一次 Q-Former | 每隔 M 层一次门控交叉注意力 |
| 视觉 token | 每图 32 个 | 每图每交叉注意力层 64 个 |
| 冻结 LLM | 是 | 是 |
| 上下文少样本 | 弱 | 强——论文的核心 |
| 交错输入 | 无原生支持 | 有，这是设计目标 |
| 训练数据 | 1.3 亿对 | 13 亿对 + 4300 万交错网页 |
| 参数量 | 训练 1.88 亿 | 训练约 100 亿（交叉注意力层） |
| 计算 | 8 张 A100 上几天 | 数千张 TPUv4 上几周 |

预算有限、单图 VQA 选 BLIP-2。交错、少样本或多图推理选 Flamingo/Idefics2。

```figure
cross-attention-fusion
```

## 使用它

`code/main.py` 演示：

1. 在 36 个假 patch token、8 个可学习 latent 上的 Perceiver resampler（纯 Python 交叉注意力）。
2. 一个门控交叉注意力步骤：`alpha = 0` → 输出等于输入（LLM 不变），然后 `alpha = 2.0` → 混入视觉贡献。
3. 一个交错掩码构建器，为「(图像 1) (文本 1) (图像 2) (文本 2)」序列生成二维注意力掩码。

## 交付它

本节课产出 `outputs/skill-gated-bridge-diagnostic.md`。给定一个开放 VLM 的配置（有无 resampler、交叉注意力频率、门控方案），它识别出 Flamingo 谱系的元素并解释冻结策略。用于调试微调后文本性能下降的原因（答案：门开得太宽、太快）。

## 练习

1. 计算 Flamingo-9B 的视觉参数量：9B LLM + 1.4B 门控交叉注意力层 + 64M resampler。训练的参数占总数的多少比例？

2. 用 PyTorch 实现门控残差 `y = tanh(alpha) * cross + x`。通过实验证明 `alpha=0` 时，初始化阶段 `y==x` 精确成立。

3. 阅读 OpenFlamingo 第 3.2 节（arXiv:2308.01390）关于如何处理一个 batch 中每个 prompt 图像数量不同的情况。描述填充策略。

4. 为什么 Flamingo 的交叉注意力掩码让文本 token *只关注最近一张*前置图像，而不是所有前置图像？阅读 Flamingo 论文第 2.4 节并解释这一权衡。

5. 上下文少样本：为一个新的 Flamingo 变体构造 prompt，包含 4 个「图像 → 主要物体的颜色」示例。描述当示例数量从 0 变到 8 时，预期的准确率模式。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Perceiver resampler | 「固定 latent 交叉注意力」 | 从可变数量的输入 patch 产出 K 个固定 token 的模块 |
| Gated cross-attention | 「tanh 门控桥梁」 | 残差层 `y = tanh(alpha)*cross + x`，alpha 可学习，初始为 0 |
| Interleaved input | 「混合序列」 | 图像和文本按阅读顺序自由混合的 prompt 格式 |
| Frozen LLM | 「LLM 无梯度」 | 文本 LLM 的权重不更新；只有 resampler + 交叉注意力层训练 |
| Few-shot | 「上下文示例」 | 在 prompt 中给出几对（图像，答案）；模型无需微调即可泛化 |
| OBELICS | 「交错网页语料」 | 1.41 亿个按阅读顺序排列图像与文本的网页开放数据集 |
| Chinchilla | 「70B 冻结基座」 | Flamingo 的冻结文本 LLM，来自 DeepMind 的 Chinchilla 论文 |
| Gate schedule | 「alpha 如何变化」 | 训练期间交叉注意力门打开的速率 |
| Cross-attn frequency | 「每隔 M 层」 | 门控交叉注意力块的插入频率；Flamingo 用 $M=4$ |
| OpenFlamingo | 「开放复现」 | MosaicML/LAION 的 3–9B 开放 checkpoint；架构与 Flamingo 一致 |

## 延伸阅读

- [Alayrac 等——Flamingo（arXiv:2204.14198）](https://arxiv.org/abs/2204.14198)——原始论文。
- [Awadalla 等——OpenFlamingo（arXiv:2308.01390）](https://arxiv.org/abs/2308.01390)——开放复现。
- [Laurençon 等——OBELICS（arXiv:2306.16527）](https://arxiv.org/abs/2306.16527)——交错网页语料。
- [Jaegle 等——Perceiver IO（arXiv:2107.14795）](https://arxiv.org/abs/2107.14795)——通用的 Perceiver 架构。
- [Li 等——Otter（arXiv:2305.03726）](https://arxiv.org/abs/2305.03726)——指令微调的 Flamingo 后代。
- [Laurençon 等——Idefics2（arXiv:2405.02246）](https://arxiv.org/abs/2405.02246)——Flamingo 路线的现代简化。
