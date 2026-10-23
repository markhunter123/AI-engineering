# 视觉自回归建模（VAR）：下一尺度预测

> 扩散模型在时间上迭代采样（去噪步）。VAR 在尺度上迭代采样——先预测一个 1x1 token，然后 2x2，然后 4x4，一直到最终分辨率，每个尺度都以之前的尺度为条件。2024 年的论文表明，VAR 在图像生成上符合 GPT 式的扩展律，并在同等算力预算下击败 DiT。本节课构建它的核心机制。

**Type:** Build
**Languages:** Python (with PyTorch)
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time:** ~90 minutes

## 问题

自回归生成统治了语言建模，因为它可预测地扩展：更多算力、更多参数、更低的困惑度、更好的输出。2024 年之前，图像生成有过两次主要的 AR 尝试：PixelRNN/PixelCNN（逐像素）和 DALL-E 1 / Parti / MuseGAN（在 VQ-VAE 码上逐 token）。

两者都受困于生成顺序问题。像素和 token 排列在二维网格上，但 AR 模型必须按一维光栅顺序访问它们。早期角落的像素完全不知道图像最终会变成什么样。生成质量的扩展性不如文本上的 GPT，在同等算力下也从未达到扩散模型的质量。

VAR 通过改变「被生成的对象」来解决生成顺序问题。VAR 不逐个空间位置预测图像 token，而是以递增的分辨率预测整张图像。第 1 步：预测一个 1x1 token（整张图像的「摘要」）。第 2 步：预测 2x2 的 token 网格（更粗的特征）。第 3 步：预测 4x4 网格。第 K 步：预测最终的 (H/8)x(W/8) 网格。

每个尺度都关注所有之前的尺度（按「尺度顺序」因果），并在自身尺度内并行。顺序问题消失了：第 k 个尺度的整张图像在一次 Transformer 前向中产生。

## 概念

### VQ-VAE 多尺度 Tokenizer

VAR 需要一个**多尺度离散 tokenizer**。对于图像 $x$，它产生一系列分辨率递增的 token 网格：

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

每个 $z_k$ 使用相同的码本（典型大小 4096-16384）。各尺度的 tokenization 不是独立的——训练时要求各尺度残差之和能重建 $f$：

```math
f \approx \mathop{\mathrm{upsample}}(\mathop{\mathrm{embed}}(z_1), \text{target\_size}) + \cdots + \mathop{\mathrm{upsample}}(\mathop{\mathrm{embed}}(z_K), \text{target\_size})
```

这是**残差 VQ** 的一个变体。尺度 $k$ 捕获尺度 $1 \ldots k-1$ 遗漏的内容。解码器取所有尺度 embedding 之和并生成图像。

多尺度 VQ tokenizer 训练一次（像 VQGAN 那样），然后冻结。所有生成工作都由其上的自回归模型完成。

### 下一尺度预测

生成模型是一个 Transformer，它看到所有之前尺度的 token，并预测下一尺度的 token。

输入序列结构：
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

位置编码同时编码尺度索引和尺度内的空间位置。注意力按尺度顺序因果：尺度 $k$、位置 $(i, j)$ 的 token 可以关注尺度 $1 \ldots k$ 的所有 token，以及尺度 $k$ 内部按所用尺度内顺序排在前面的 token（VAR 使用固定的位置注意力，尺度内没有因果性——一个尺度内的所有位置并行预测）。

训练损失：在每个尺度 $k$，在给定所有先前尺度 token 的条件下预测 token $z_k$。对离散 VQ 码做交叉熵损失。结构与 GPT 相同，只是「序列」现在按尺度结构化。

### 生成

推理时：
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

对于 $K = 10$ 个尺度，生成是 10 次 Transformer 前向。每次前向并行产生整个尺度——尺度内没有逐 token 自回归。对于 256x256 图像，大约 10 次前向，而 DiT 需要 28-50 次。

### 为什么下一尺度优于下一 token

三个结构性优势：
1. **由粗到细符合自然图像统计。** 人类视觉感知和图像数据集都表现出与尺度相关的规律：低频结构稳定且可预测；高频细节以低频内容为条件。下一尺度预测利用了这一点。
2. **尺度内并行生成。** 与 GPT 式 token AR 不同，VAR 一步就能产生一个尺度上的所有 token。有效生成长度从线性变成对数级。
3. **没有生成顺序偏置。** 尺度 k 的 token 能看到整个尺度 k-1；不存在「左侧」或「上方」的偏置，不会强迫早期 token 在后期上下文可用之前就做出承诺。

### 扩展律

Tian et al. 证明 VAR 在 ImageNet 上的 FID 遵循幂律扩展曲线——正如 GPT 对困惑度那样。参数或算力翻倍，误差可靠地减半。这是第一个像语言模型一样干净地展现这类扩展行为的图像生成模型。结果是，VAR 的规模预测可以从算力推算，而不是按架构做经验猜测。

### 与扩散的关系

VAR 和扩散共享同样的数据压缩叙事：两者都把生成问题拆成一串更简单的子问题。

- 扩散：逐步加噪，学会撤销一步。
- VAR：逐步加分辨率，学会预测下一尺度。

它们是穿过这个问题的不同轴。两者都得到可处理的条件分布。经验上，VAR 推理更快（前向次数更少，尺度内全并行），在类别条件 ImageNet 上追平或击败 DiT。文本条件的 VAR（VARclip、HART）是活跃的研究方向。

```figure
gx-var-next-scale
```

## 构建它

在 `code/main.py` 中你将：
1. 在合成的「图像」数据（二维高斯环）上构建一个微型**多尺度 VQ tokenizer**。
2. 训练一个 **VAR 式 Transformer** 做下一尺度 token 预测。
3. 调用 Transformer 4 次（4 个尺度）并解码来采样。
4. 验证按尺度排序的训练使生成在尺度内并行。

这是一个玩具实现。重点在于看到尺度结构的注意力掩码和尺度内并行生成真正跑起来。

## 交付它

本节课产出 `outputs/skill-var-tokenizer-designer.md`——一个设计多尺度 tokenizer 的 skill：尺度数量、尺度比例、码本大小、残差共享、解码器架构。

## 练习

1. **尺度数量消融。** 用 4、6、8、10 个尺度训练 VAR。衡量重建质量与自回归前向次数的关系。更多尺度 = 更细的残差 = 更好质量，但更多前向。

2. **码本大小。** 用 512、4096、16384 的码本训练 tokenizer。更大的码本重建更好，但预测更难。找到拐点。

3. **尺度内并行检查。** 对训练好的 VAR，显式测量注意力模式。在尺度 k 内，模型是否关注跨尺度位置而不关注尺度内位置？验证掩码实现。

4. **VAR vs DiT 扩展。** 在同一 ImageNet 类别条件任务上，用匹配的参数预算（例如 33M、130M、458M）训练 VAR 和 DiT。绘制 FID 与算力的关系。VAR 应在每个规模上领先 DiT——在小规模复现论文结果。

5. **文本条件。** 扩展 VAR，通过 adaLN 把文本 embedding（CLIP pooled）作为额外条件输入。这就是 HART 配方。文本对齐采样下 FID 能改善多少？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| VAR | 「Visual AutoRegressive」 | 在 VQ token 网格金字塔上做下一尺度预测的图像生成 |
| Next-scale prediction | 「先粗后细地预测」 | 模型在递增的分辨率尺度上预测 token，以所有先前尺度为条件 |
| Multi-scale VQ tokenizer | 「残差 VQ」 | 产生 K 个分辨率递增 token 网格的 VQ-VAE，解码器对各尺度求和 |
| Scale k | 「金字塔第 k 层」 | K 个分辨率层级之一，从 k=1 的 1x1 到 k=K 的 (H/p)x(W/p) |
| Parallel-within-scale | 「每个尺度一次前向」 | 尺度 k 的所有 token 在一次 Transformer 前向中预测，而非自回归 |
| Causal-across-scales | 「尺度顺序注意力」 | 尺度 $k$ 的 token 可以关注尺度 $1 \ldots k$ 的全部，但不能关注尺度 $k+1 \ldots K$ |
| Residual VQ | 「加性 tokenization」 | 每个尺度的 token 编码低尺度留下的残差；解码器对所有尺度 embedding 求和 |
| VAR scaling law | 「图像 GPT 扩展律」 | FID 随算力遵循可预测的幂律，如同语言模型的困惑度 |
| HART | 「混合 VAR + 文本」 | 文本条件的 VAR 变体，把 MaskGIT 式迭代解码与 VAR 的尺度结构结合 |
| Scale position embedding | 「(scale, row, col) 三元组」 | 位置编码同时携带尺度索引与尺度内的空间坐标 |

## 延伸阅读

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — VAR 论文，权威参考
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT，扩散对比基线
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN，VAR 多尺度 tokenizer 所扩展的 tokenizer 家族
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE，离散图像 tokenization 的基础
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — 文本条件 VAR
