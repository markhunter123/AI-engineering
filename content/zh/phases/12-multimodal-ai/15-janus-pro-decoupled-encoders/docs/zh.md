# Janus-Pro：面向统一多模态模型的解耦编码器

> 统一多模态模型存在一个绕不开的矛盾。理解需要语义特征——SigLIP 或 DINOv2 输出的向量富含概念级信息。生成需要利于重建的编码——VQ token 能重新组合成清晰的像素。这两个目标在单个编码器里无法兼容。Janus（DeepSeek，2024 年 10 月）和 Janus-Pro（DeepSeek，2025 年 1 月）主张的解决办法是停止强行统一：把两个编码器解耦。任务之间共享 Transformer 主体，但理解走 SigLIP、生成走 VQ tokenizer。在 7B 规模下，Janus-Pro 的 GenEval 成绩超过 DALL-E 3，MMMU 成绩与 LLaVA 持平。本课解读为什么两个编码器能成，而一个不行。

**Type:** Build
**Languages:** Python (stdlib, dual-encoder routing + shared-body signal)
**Prerequisites:** Phase 12 · 13 (Transfusion), Phase 12 · 14 (Show-o)
**Time:** ~120 minutes

## 学习目标

- 解释为什么单一共享编码器必然牺牲理解或生成质量中的一项。
- 描述 Janus-Pro 的路由方式：输入端用 SigLIP 特征做理解，输入和输出两端都用 VQ token 做生成。
- 梳理让 Janus-Pro 成功、而 Janus 未能成功的数据配比扩展过程。
- 比较解耦架构（Janus-Pro）、耦合连续架构（Transfusion）和耦合离散架构（Show-o）。

## 问题

统一模型在理解和生成之间共享一个 Transformer 主体。此前的尝试（Chameleon、Show-o、Transfusion）在两个方向上都使用同一个视觉 tokenizer。这个 tokenizer 是一种折中：

- 为重建（生成）优化：VQ-VAE 能捕捉细粒度的像素细节，但产出的 token 语义连贯性弱。
- 为语义（理解）优化：SigLIP 嵌入把「猫」的图像聚在「猫」的 token 附近，但无法很好地重建。

Show-o 和 Transfusion 为此在某个方向上付出了可见的质量代价。Janus-Pro 问的是：既然任务需求不同，为什么非要共用一个 tokenizer？

## 概念

### 解耦的视觉编码

Janus-Pro 的架构把两个编码器分开：

- 理解路径。输入图像 → SigLIP-SO400m → 2 层 MLP → Transformer 主体。
- 生成路径。输入图像（如果要以现有图像为条件）→ VQ tokenizer → token ID → Transformer 主体。
- 输出生成。Transformer 预测的图像 token → VQ 解码器 → 像素。

Transformer 主体是共享的。主体上下游的一切都按任务区分。

输入靠 prompt 格式来区分：`<understand>` 标签走 SigLIP，`<generate>` 标签走 VQ。或者由任务隐式决定路由。

### 为什么这样可行

理解损失拿到的是 SigLIP 特征，CLIP 风格的预训练已将它调教得适合语义相似度。模型的感知类基准优于 Show-o / Transfusion，因为输入特征更贴合任务。

生成损失拿到的是 VQ token，tokenizer 已将它调教得适合重建。图像质量优于 Show-o，因为 VQ 编码能干净地还原成像素。

共享的 Transformer 主体见到两种输入分布（SigLIP 和 VQ），并学会同时处理两者。其论点是：数据足够多、参数足够大，主体就能吸收这种切换。

### 数据扩展——Janus 与 Janus-Pro

Janus（初代，arXiv 2410.13848）提出了解耦，但规模很小（1.3B 参数，数据有限）。Janus-Pro（arXiv 2501.17811）做了扩展：

- 7B 参数（对比 1.3B）。
- 第一阶段（对齐）的图像-文本对从 72M 增加到 90M。
- 第二阶段（统一）从 26M 增加到 72M。
- 为第三阶段新增 20 万条图像生成指令样本。

结果是：Janus-Pro-7B 在 MMMU 上与 LLaVA 持平（60.3 对约 58），在 GenEval 上超过 DALL-E 3（0.80 对 0.67）。一个开放模型，在统一模型光谱的两端都具备竞争力。

### JanusFlow——rectified flow 变体

JanusFlow（arXiv 2411.07975）把 VQ 生成路径换成 rectified-flow 生成路径（连续）。分工变成 SigLIP 负责理解 + rectified flow 负责生成。质量上限进一步提高。架构仍是解耦编码器 + 共享主体。

### 共享主体的职责

Transformer 主体处理的是一条统一序列，但输入来自两种分布。它的职责是：

- 理解：接收 SigLIP 特征 + 文本 token → 自回归地输出文本。
- 生成：接收文本 token +（可选的图像 VQ token）→ 自回归地输出图像 VQ token。

主体在每个 block 上都没有模态专属权重。它就是你会在 Qwen 或 Llama 内部看到的那种文本式 Transformer，外加两个输入适配器。

有意思的是，这意味着 Janus-Pro 的主体可以从预训练 LLM 初始化。Janus-Pro 确实从 DeepSeek-MoE-7B 初始化。这个选择很重要：LLM 带来的推理能力，是纯从零训练的统一模型难以达到的。

### 与 InternVL-U 的比较

InternVL-U（第 12.10 课）是 2026 年的后续工作。它结合了：

- 原生多模态预训练（InternVL3 主干）。
- 解耦编码器路由（SigLIP 进，VQ + 扩散 head 出）。
- 统一的理解 + 生成 + 编辑。

InternVL-U 把 Janus-Pro 的架构选择纳入了一个更大的框架。解耦编码器的思路如今已是规模化统一模型的默认做法。

### 局限

解耦编码器增加了架构复杂度。要训练两个 tokenizer、维护两条输入路径、面对两组失效模式。对于不需要生成的产品，Janus-Pro 属于过度设计——选 LLaVA 家族的理解模型即可。

对于不需要理解的产品，Janus-Pro 又大材小用——选 Stable Diffusion 3 / Flux 模型即可。

对于两者都需要的产品，Janus-Pro 目前是开放架构的参考实现。

```figure
l5-janus-decouple
```

## 使用它

`code/main.py` 模拟 Janus-Pro 的路由：

- 两个模拟编码器：类 SigLIP（产出 256 维语义向量）和类 VQ（产出整数编码）。
- 一个 prompt 路由器，根据任务标签选择编码器。
- 一个共享主体（替身），无论 token 来自哪个编码器都能处理。
- 一个从第一阶段（对齐）到第三阶段（指令微调）的加权采样调度切换。

打印 3 个样例的路由路径：图像问答、T2I、图像编辑。

## 交付它

本课产出 `outputs/skill-decoupled-encoder-picker.md`。给定一个希望以接近前沿的质量实现统一生成 + 理解的产品，它在 Janus-Pro、JanusFlow 和 InternVL-U 之间做选择，并给出具体的数据规模建议。

## 练习

1. Janus-Pro-7B 在 GenEval 上超过 DALL-E 3。解释为什么一个 7B 开放模型能在生成上比肩前沿专有模型，却在理解上做不到。

2. 实现一个路由函数：给定 prompt 文本，分类为 `understand` 或 `generate`。对于「先描述再画出来」这类模糊 prompt 你如何处理？

3. JanusFlow 用 rectified flow 替换了 VQ 路径。此时 Transformer 主体输出什么，损失函数有什么变化？

4. 提出 Janus-Pro 架构再加一个解耦编码器就能处理的第四项任务。例如：图像分割（DINO 风格）、深度估计（MiDaS 风格）。

5. 读 Janus-Pro 第 4.2 节关于数据扩展的内容。相比 Janus，哪个数据阶段对 T2I 质量的提升贡献最大？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 解耦编码 | 「两个视觉编码器」 | 每个方向使用独立的 tokenizer 或编码器：理解用语义编码，生成用重建编码 |
| 共享主体 | 「一个 Transformer」 | 单个 Transformer 处理任一编码器的输出；没有模态专属权重 |
| SigLIP 用于理解 | 「语义特征」 | CLIP 家族的视觉塔，提供丰富的概念特征，但重建能力差 |
| VQ 用于生成 | 「重建编码」 | 向量量化 token，能干净地解码回像素 |
| JanusFlow | 「rectified-flow 变体」 | 把 Janus-Pro 的生成 head 换成连续 flow matching 而非 VQ |
| 路由标签 | 「任务标签」 | 选择输入编码器的 prompt 标记（`<understand>` / `<generate>`） |

## 延伸阅读

- [Wu 等 — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen 等 — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma 等 — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong 等 — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
