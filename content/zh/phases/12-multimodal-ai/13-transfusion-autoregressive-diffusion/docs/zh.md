# Transfusion：一个 Transformer 中的自回归文本 + 扩散图像

> Chameleon 和 Emu3 把所有筹码押在离散 token 上。它们能跑通，但量化瓶颈清晰可见——图像质量在连续空间扩散模型之下停滞不前。Transfusion（Meta，Zhou 等，2024 年 8 月）押的是相反的方向：让图像保持连续，彻底去掉 VQ-VAE，用一个 Transformer 搭配两种损失来训练。文本 token 走下一 token 预测（NTP），图像 patch 走 flow matching / 扩散损失。两个目标优化的是同一组权重。支撑 Stable Diffusion 3 的架构（MMDiT）与它是近亲。本课解读 Transfusion 的核心论点，搭建一个玩具级双损失训练器，并追踪让同一个 Transformer 同时胜任两项工作的注意力掩码。

**Type:** Build
**Languages:** Python (stdlib, two-loss trainer on MNIST-scale toy)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 8 (Generative AI)
**Time:** ~180 minutes

## 学习目标

- 在一个主干网络上接出两种损失（文本 token 上的 NTP、图像 patch 上的扩散 MSE）。
- 解释为什么「图像 patch 之间双向注意力 + 文本 token 上因果注意力」是正确的掩码选择。
- 从算力、质量、代码复杂度三方面比较 Transfusion 式（连续图像、扩散损失）与 Chameleon 式（离散图像、NTP）。
- 说出 MMDiT 的贡献：每个 block 使用模态专属权重，在残差流上做联合注意力。

## 问题

离散图像 token 与连续图像 token 的争论比 LLM 还要早。连续表示（原始像素、VAE latent）保留细节。离散 token（VQ 索引）能接入 Transformer 的原生词表，但在量化那一步丢失细节。

Chameleon / Emu3 选择了离散：一个损失、一套架构，但图像保真度被 tokenizer 的质量封顶。

扩散模型选择了连续：图像质量极为出色，但它是与 LLM 分离的另一个模型，噪声调度工程复杂，而且无法与文本生成干净地整合。

Transfusion 提出的问题是：能不能两者兼得？图像保持连续，仍然只训练一个模型，把两种损失缝进同一个梯度步。

## 概念

### 双损失架构

一个纯解码器 Transformer 处理这样一条序列：

- 文本 token（离散，来自 BPE 词表）。
- 图像 patch（连续，16x16 像素块经线性嵌入投影到隐藏维度——与 ViT 编码器的输入相同）。
- `<image>` 和 `</image>` 标签，标记连续 patch 所在的位置。

前向传播只跑一次。损失函数按 token 类型在两种 head 中二选一：

- 对文本 token：在词表 logits head 上做标准交叉熵。
- 对图像 patch：在连续 patch 上做扩散损失——预测加到每个 patch 上的噪声。

梯度流经共享的 Transformer 主体。两种损失同时改进共享权重。

### 注意力掩码：文本因果 + 图像双向

文本 token 必须是因果的——不能让一个文本 token 注意到未来的文本，否则 teacher forcing 就会失效。但图像 patch 表示的是同一个快照，它们应当在同一个图像块内部彼此双向注意。

掩码的定义：

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

在训练和推理时都以块三角（block-triangular）掩码实现。

### Transformer 内部的扩散损失

扩散损失是标准的：给图像 patch 加噪，让模型预测噪声（等价地，也可以预测干净 patch）。Transfusion 的版本使用 flow matching——预测从噪声到干净数据的速度场。

训练时：

1. 对每个图像 patch $x_0$，采样一个随机时间步 $t$。
2. 采样噪声 $\epsilon$，计算 $x_t = (1-t) \cdot x_0 + t \cdot \epsilon$（flow matching 的线性插值）。
3. Transformer 预测 $v_\theta(x_t, t)$；$\text{loss} = \mathop{\mathrm{MSE}}(v_\theta(x_t, t), \epsilon - x_0)$。
4. 与同一条序列上的文本 NTP 损失一起反向传播。

推理时，生成过程是：

- 文本 token：标准的自回归采样。
- 图像 patch：以先前的文本 token 为条件，跑扩散采样循环（通常 10-30 步）。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser 等，2024 年 3 月）在 Transfusion 前后脚发布了 MMDiT（Multimodal Diffusion Transformer）。这两个架构是同胞兄弟。

MMDiT 的关键差异：

- 每个 block 有模态专属权重。每个 Transformer block 为文本 token 和图像 patch 分别设置 Q、K、V 和 MLP 权重。注意力是联合的（跨模态），其余部分都是模态专属的。
- Rectified flow 训练。一种特定的 flow matching 变体，采样过程已知，数学比 DDPM 更简单。
- 规模。MMDiT 是 SD3 的主干（2B 和 8B 参数两个版本）。Transfusion 论文扩展到 7B。

两者在同一个核心思想上殊途同归：一个 Transformer 在文本上跑 NTP，在连续图像表示上跑扩散。

### 为什么这优于 Chameleon 式

连续扩散与离散 NTP 在图像生成上的质量差距是可以测量的。Transfusion 论文报告：

- 在 7B 参数下，FID 比同尺寸的 Chameleon 式模型好 3-5 个点。
- 不需要训练 tokenizer——图像编码器更简单（线性投影到隐藏维度，与 ViT 的输入层相同）。
- 推理时可以并行去噪图像 patch，而不像自回归图像 token 那样必须顺序生成。

缺点：Transfusion 是双损失模型，训练动态更棘手。损失权重需要调。NTP 与扩散的调度不匹配时，可能让某个 head 占据主导。

### 下游是什么

Janus-Pro（第 12.15 课）改进了 Transfusion 的思路：把理解用的视觉编码器和生成用的视觉编码器解耦——一个用 SigLIP，另一个用 VQ——同时共享 Transformer 主体。Show-o（第 12.14 课）把扩散换成离散扩散（掩码预测）。在 Transfusion 之后，统一生成这一家族迅速分叉。

2026 年能输出图像的生产级 VLM——Gemini 3 Pro、GPT-5、Claude Opus 4.7 的图像生成路径——几乎可以肯定使用了这个家族的某种后代。细节是专有的。

```figure
cfg-guidance-scale
```

## 使用它

`code/main.py` 在一个微型 MNIST 式问题上构建玩具版 Transfusion：

- 文本描述是描述数字（0-9）的短整数序列。
- 图像是 4x4 的字节网格。
- 一对共享权重的线性投影充当 Transformer 的替身；文本上是 NTP 损失，噪声 patch 上是 MSE 损失。
- 训练循环交替优化两种损失，注意力掩码是显式构造的。
- 生成在一次前向传播中同时产出一条文本描述和一张 4x4 图像。

这个 Transformer 是玩具级的。真正有价值的产物是双损失的管线、注意力掩码的构造和推理循环。

## 交付它

本课产出 `outputs/skill-two-loss-trainer-designer.md`。给定一个新的多模态训练任务（文本 + 图像、文本 + 音频、文本 + 视频），它会设计双损失调度（损失权重、掩码形状、共享块与模态专属块），并指出实现风险。

## 练习

1. 一个 Transfusion 式模型训练时 70% 是文本 token、30% 是图像 patch。图像扩散损失的量级约为文本 NTP 损失的 10 倍。怎样的损失权重能让两者平衡？

2. 为序列 `[T, T, <image>, P, P, P, P, </image>, T]` 实现块三角掩码。把每个条目标成 0 或 1。

3. MMDiT 有模态专属的 QKV 权重。相比 Transfusion 全共享的 Transformer，这增加了多少参数开销？在 7B 参数下值得吗？

4. 生成：给定一个文本 prompt，模型先跑 50 个 token 的 NTP，然后碰到 `<image>`，再在 256 个 patch 上跑 20 步去噪。总共多少次前向传播？

5. 读 SD3 论文第 3 节。描述 rectified flow，并解释为什么它的推理步数比 DDPM 少就能收敛。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 双损失训练 | 「NTP + 扩散」 | 一个 Transformer 在同一个梯度步中同时优化文本 token 的交叉熵和连续图像 patch 的 MSE |
| Flow matching | 「Rectified flow」 | 预测从噪声到干净数据的速度场的扩散变体；数学比 DDPM 简单 |
| MMDiT | 「Multimodal DiT」 | Stable Diffusion 3 的架构：联合注意力，模态专属的 MLP 和 norm |
| 块三角掩码 | 「文本因果 + 图像双向」 | 在文本上因果、在图像区域内双向的注意力掩码 |
| 连续图像表示 | 「No VQ」 | 图像 patch 是实值向量，而不是整数码本索引 |
| 速度预测 | 「v-parameterization」 | 网络输出的是噪声与数据之间的速度场，而不是噪声本身 |

## 延伸阅读

- [Zhou 等 — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser 等 — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao 等 — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie 等 — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
