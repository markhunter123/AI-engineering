# Emu3：用下一 token 预测做图像与视频生成

> BAAI 的 Emu3（Wang 等，2024 年 9 月）本应终结扩散与自回归之争。一个 Llama 风格的仅解码器 Transformer，只用下一 token 预测目标训练，覆盖文本 + VQ 图像 token + 3D VQ 视频 token 的统一词表，在图像生成上击败 SDXL，在感知上击败 LLaVA-1.6。没有 CLIP 损失，没有扩散调度。推理时使用无分类器引导（classifier-free guidance）来提升质量，但核心训练目标是带教师强制（teacher forcing）的下一 token 预测。发表在 Nature 上。本课解读 Emu3 的论点——为什么一个更好的 tokenizer 加上规模就够了——并与扩散方案对比。

**Type:** Learn
**Languages:** Python (stdlib, 3D video tokenizer math + autoregressive sampler skeleton)
**Prerequisites:** Phase 12 · 11 (Chameleon)
**Time:** ~120 minutes

## 学习目标

- 解释为什么 Emu3 的单损失下一 token 目标能奏效，尽管长期存在「图像质量必须用扩散」的假设。
- 描述 3D 视频 tokenizer：时空 VQ 码本长什么样，为什么 patch 要跨时间。
- 在（训练算力、推理成本、质量上限）上比较 Emu3 与 Stable Diffusion XL。
- 说出同一个 Emu3 模型扮演的三种角色：Emu3-Gen（图像生成）、Emu3-Chat（感知）、Emu3-Stage2（视频生成）。

## 问题

到 2024 年，传统观点认为：图像生成需要扩散。论据是：离散图像 token 丢失太多信息，无法重建细节，而自回归采样会在数千个 token 上累积误差。Stable Diffusion、DALL-E 3、Imagen、Midjourney 都使用某种形式的扩散。Chameleon（第 12.11 课）在小规模上部分否定了这一点，但质量没追上 SDXL。

Emu3 正面攻击这个论据。主张：更好的视觉 tokenizer + 足够的规模 + 下一 token 损失 = 在同一模型中实现击败扩散的图像生成，同时还能做感知。

这个押注在发表时很有争议。两年后，开源统一生成家族（Emu3、Show-o、Janus-Pro、Transfusion）是研究的默认路径；生产级前沿模型似乎也在使用某种变体。

## 概念

### Emu3 的 tokenizer

关键成分是视觉 tokenizer。Emu3 训练了一个定制的 IBQ 类 tokenizer（Inverse Bottleneck Quantizer，SBER-MoVQGAN 家族），每个 token 做 8x8 分辨率降采样。512x512 图像变成 64x64 = 4096 个 token，码本大小 32768。

这比 Chameleon 每张 512x512 图像 1024 个 token、K=8192 更多，但每 token 更便宜（更小的码本查找、更简单的编解码器）。关键指标：重建 PSNR 30.5 dB，与 Stable Diffusion 连续潜空间的 32 dB 有竞争力。

视频方面：3D VQ tokenizer 把时空 patch（4x4x4 像素）编码为一个整数。8 FPS 的 4 秒片段有 32 帧；256x256 下，4x 空间和 4x 时间降采样，token 数为 (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 个 token。

tokenizer 质量就是上限。Emu3 的贡献部分在于「我们训练了一个非常好的 tokenizer」。

### 单损失训练

Emu3 只有一个目标：在文本 token、2D 图像 token 和 3D 视频 token 的共享词表上做下一 token 预测。训练时按模态乘上不同权重以平衡贡献，但损失函数完全相同。

在混合数据上训练：
- 图像生成：`<text caption> <image> image_tokens </image>`
- 图像感知：`<image> image_tokens </image> <question> text_tokens`
- 视频生成：`<text caption> <video> video_tokens </video>`
- 视频感知：类似。
- 纯文本：标准 NTP。

模型从数据分布中学会何时输出图像 token、何时输出文本 token。生成能力从模型在 `<image>` 标签之后预测图像 token 中涌现。

### 无分类器引导与温度

自回归图像生成在推理时使用无分类器引导（CFG）会好很多。Emu3 也用它：生成两次，一次用完整描述，一次用空描述，按引导权重（典型 3.0-7.0）混合 logits。这是扩散所用的同一 CFG 技巧，借到了自回归场景。

温度很重要：太高会产生伪影；太低会模式崩溃。Emu3 推荐感知用 1.0，图像生成用 0.8。

### 三种角色，一个模型

Emu3 以三个功能不同的 API 发布，但底层只有一套权重：

- Emu3-Gen。图像生成。输入文本，输出图像 token。
- Emu3-Chat。VQA 和描述生成。输入图像（token），输出文本。
- Emu3-Stage2。视频生成和视频 VQA。输入文本或视频，输出文本或视频。

没有任务特定头，只是不同的 prompt 模板。同一个 checkpoint。

### 基准

来自 Emu3 论文（2024 年 9 月）：

- 图像生成：在 MJHQ-30K FID 上击败 SDXL（5.4 vs 5.6），GenEval 总体（0.54 vs 0.55——统计上打平），Deep-Eval 综合得分相当。
- 图像感知：VQAv2 上击败 LLaVA-1.6（75.1 vs 72.4），MMMU 上大致相当。
- 视频生成：4 秒片段质量在 FVD 上与 Sora 时代公开基准模型有竞争力。

这些数字并非总是赢——Emu3 在这里让一分、那里赢一分——但「下一 token 预测就是你所需要的一切」这一主张跨模态站得住脚。

### 算力成本

Emu3 用 7B 参数模型训练了约 3000 亿多模态 token。GPU 小时大致与 Llama-2-7B 预训练相当（A100 级芯片上 2000-4000 GPU 年）。Stable Diffusion 3 这类扩散模型的训练预算相近，但需要单独的文本编码器和更复杂的 pipeline。

推理时，Emu3 每张图比 SDXL 慢：4096 个图像 token 以 30 tok/s 计，每张 512x512 图像约 2 分钟，而 SDXL 只要 2-5 秒。投机解码和 KV cache 优化能缩小差距，但无法消除。自回归图像生成是算力密集型的；这是长期权衡。

### 为什么重要

Emu3 的深层贡献是概念上的。如果下一 token 预测能扩展到在图像生成上追平扩散，统一模型路线（一个损失、一个主干、任意模态）就是可行的。未来的模型不需要单独的文本编码器、单独的扩散调度器、单独的 VAE。一个 Transformer，每个模态一个 tokenizer，再加规模。

Show-o、Janus-Pro 和 InternVL-U 都在这篇论文的基础上构建或挑战它。截至 2025 年，中国实验室（BAAI、DeepSeek）在这个方向上发表得比美国实验室更积极。

```figure
l5-emu3-next-token
```

## 使用它

`code/main.py` 构建两个玩具组件：

- 2D vs 3D VQ tokenizer 计数计算器：给定（分辨率、patch、片段长度、FPS），计算图像 vs 视频的 token 数。
- 带无分类器引导和温度的自回归图像 token 采样器。

CFG 实现与 Emu3 的配方一致——用引导权重混合条件与无条件 logits。

## 交付它

本课产出 `outputs/skill-token-gen-cost-analyzer.md`。给定生成产品规格（图像或视频、目标分辨率、质量档位、延迟预算），它计算 token 数、推理成本，并在 Emu3 家族与扩散之间选择。

## 练习

1. Emu3 在 8x8 降采样下每张 512x512 图像产生 4096 个 token。计算 1024x1024 和 2048x2048 的对应值。推理延迟会怎样变化？

2. 阅读 Emu3 第 3.3 节关于视频 tokenizer 的内容。描述 3D VQ patch 的形状，以及为什么是 4x4x4 而不是 8x8x1。

3. 无分类器引导权重 5.0 vs 3.0：视觉上有什么效果？在 `code/main.py` 中追踪数学。

4. 计算 Emu3-7B 在 300B token 上的训练 FLOPs，并与 Stable Diffusion 3 比较。哪个训练成本更高？

5. Emu3 在 FID 上击败 SDXL，但在 VQAv2 上不如专门的 VLM。解释为什么统一损失方案在不同基准上相对专家模型表现出不同优势。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 下一 token 预测 | 「NTP」 | 标准自回归损失：给定 token[0..i] 预测 token[i+1]；token 化后适用于所有模态 |
| IBQ tokenizer | 「逆瓶颈量化器」 | 一类 VQ-VAE，码本更大（32768+），重建质量优于 Chameleon |
| 3D VQ | 「时空量化器」 | 由（时间、行、列）索引的码本；一个 token 覆盖 4x4x4 像素立方体 |
| 无分类器引导 | 「CFG」 | 用权重 gamma 混合条件与无条件 logits；推理时提升图像质量 |
| 统一词表 | 「共享 token」 | 文本 + 图像 + 视频都取自同一整数空间；模型预测接下来出现的模态 |
| MJHQ-30K | 「图像生成基准」 | 含 3 万条 prompt 的 Midjourney 质量基准；Emu3 在此报告 FID |

## 延伸阅读

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
