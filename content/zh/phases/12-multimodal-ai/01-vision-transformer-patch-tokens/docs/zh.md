# Vision Transformer 与 patch-token 原语

> 在做任何多模态工作之前，图像必须先变成 Transformer 能处理的一段 token 序列。2020 年的 ViT 论文用 16x16 像素的 patch、一个线性投影和一个位置嵌入回答了这个问题。五年之后，每一个 2026 年的前沿模型（Claude Opus 4.7 原生 2576px、Gemini 3.1 Pro、Qwen3.5-Omni）仍然这样起步——编码器从 ViT 换成了 DINOv2、又换成 SigLIP 2，加入了 register token，位置方案演变为 2D-RoPE，但这个原语保留了下来。本节课从头到尾读一遍 patch-token 流水线，并用 stdlib Python 实现它，好让第 12 阶段剩下的课程对「视觉 token」有一个具体的心智模型。

**Type:** Learn
**Languages:** Python (stdlib, patch tokenizer + geometry calculator)
**Prerequisites:** Phase 7 (Transformers), Phase 4 (Computer Vision)
**Time:** ~120 minutes

## 学习目标

- 把 HxWx3 图像转换为带正确位置编码的 patch token 序列。
- 对给定（patch 尺寸、分辨率、隐藏维度、深度）的 ViT，计算序列长度、参数数量和 FLOPs。
- 说出把 ViT 从 2020 年的研究推向 2026 年生产的三项升级：自监督预训练（DINO / MAE）、register token，以及原生分辨率打包。
- 为下游任务在 CLS 池化、平均池化和 register token 之间做选择。

## 问题

Transformer 处理的是向量序列。文本本身就是序列（字节或 token）。图像是带三个颜色通道的二维像素网格——不是序列。如果逐像素展平，一张 224x224 的 RGB 图像会变成 150,528 个 token，而在这个长度上做自注意力根本不可行（计算量随序列长度平方增长）。

2020 年之前的做法是在前面接一个 CNN 特征提取器：ResNet 产出一张 7x7 的 2048 维特征图，把这 49 个 token 喂给 Transformer。这样能跑，但继承了 CNN 的归纳偏置（平移等变性、局部感受野），也丢掉了 Transformer 随规模扩展的优势。

Dosovitskiy 等人（2020）问了一个直白的问题：如果跳过 CNN 会怎样？把图像切成固定大小的 patch（比如 16x16 像素），把每个 patch 线性投影成一个向量，加上位置嵌入，再把序列送入一个普通 Transformer。在当时这是异端——没有卷积的视觉。有了足够的数据（JFT-300M，之后是 LAION），它就在 ImageNet 上超过了 ResNet，而且还在继续变好。

到 2026 年，ViT 原语已是无可争议的基础。每个开放权重 VLM 的视觉塔都是它的某个后代（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是「要不要用 patch」，而是「用什么 patch 尺寸、什么分辨率调度、什么预训练目标、什么位置编码」。

## 概念

### Patch 即 token

给定形状为 `(H, W, 3)` 的图像 `x` 和 patch 尺寸 `P`，你把图像切成 `(H/P) x (W/P)` 个互不重叠的 patch 网格。每个 patch 是一个 `P x P x 3` 的像素立方体。把每个立方体展平成 `3 P^2` 维向量。再用一个形状为 `(3 P^2, D)` 的共享线性投影 `W_E`，把每个 patch 映射到模型的隐藏维度 `D`。

对 ViT-B/16 的经典配置：
- 分辨率 224，patch 尺寸 16 → 网格 14x14 → 196 个 patch token。
- 每个 patch 有 `16 x 16 x 3 = 768` 个像素值，投影到 `D = 768`。
- 再加上一个可学习的 `[CLS]` token → 序列长度 197。

patch 投影在数学上等同于一个卷积核大小为 `P`、步长为 `P`、输出通道数为 `D` 的二维卷积。生产代码也正是这样实现的——`nn.Conv2d(3, D, kernel_size=P, stride=P)`。「线性投影」是概念上的说法；「卷积核」是高效的做法。

### 位置嵌入

patch 本身没有顺序——Transformer 看到的是一个无序集合。早期 ViT 加入一个可学习的一维位置嵌入（每个位置一个 768 维向量，共 197 个）。能行，但把模型绑定在训练分辨率上：推理时如果改变网格，就必须对位置表做插值。

现代视觉骨干使用 2D-RoPE（Qwen2-VL 的 M-RoPE、SigLIP 2 的默认方案）或可分解的二维位置。2D-RoPE 根据 patch 的（行、列）索引旋转 query 和 key 向量，于是模型从旋转角度推断出相对二维位置。没有位置表。模型在推理时能处理任意网格尺寸。

### CLS token、池化输出与 register token

图像级表示是什么？三种选择并存：

1. `[CLS]` token。在 patch 序列前插入一个可学习向量。经过所有 Transformer 块后，CLS token 的隐藏状态就是图像表示。继承自 BERT。原始 ViT 和 CLIP 使用它。
2. 平均池化。对 patch token 的输出隐藏状态取平均。SigLIP、DINOv2 和大多数现代 VLM 使用它。
3. Register token。Darcet 等人（2023）观察到，没有显式 sink token 训练出来的 ViT 会形成高范数的「伪影」patch，劫持自注意力。加入 4–16 个可学习的 register token 可以吸收这部分负载，并提升稠密预测质量（分割、深度）。DINOv2 和 SigLIP 2 都自带 register。

这个选择对下游任务很重要。CLS 对分类来说够用。对于把 patch token 喂进 LLM 的 VLM，你完全跳过池化——每个 patch 都成为 LLM 的输入 token。register 在交接前被丢弃（它们是脚手架，不是内容）。

### 预训练：监督、对比、掩码、自蒸馏

2020 年的 ViT 用 JFT-300M 上的监督分类做预训练。很快被以下方法取代：

- CLIP（2021）：在 4 亿对图像-文本上做对比学习。见第 12.02 课。
- MAE（2021，He 等人）：遮住 75% 的 patch，重建像素。自监督，可在纯图像上工作。
- DINO（2021）/ DINOv2（2023）：学生-教师自蒸馏，不用标签，不用描述文本。2023 年的 DINOv2 ViT-g/14 是最强的纯视觉骨干，也是「稠密特征」用例的默认选择。
- SigLIP / SigLIP 2（2023、2025）：用 sigmoid 损失替代 softmax 的 CLIP，并用 NaFlex 支持原生宽高比。2026 年开放 VLM 的主导视觉塔（Qwen、Idefics2、LLaVA-OneVision）。

你的预训练选择决定了这个骨干擅长什么：CLIP/SigLIP 用于与文本做语义匹配，DINOv2 用于稠密视觉特征，MAE 作为下游微调的起点。

### 扩展规律

ViT 扩展（Zhai 等人 2022）确立：ViT 的质量在模型规模、数据规模和计算量上服从可预测的规律。在固定计算量下：
- 更大的模型 + 更多数据 → 更好的质量。
- patch 尺寸是序列长度与保真度之间的杠杆。patch 14（DINOv2/SigLIP SO400m 的典型值）在每张图上比 patch 16 产出更多 token；对 OCR 和稠密任务更好，对速度更差。
- 分辨率是另一个大杠杆。从 224 到 384 再到 512 几乎总是有帮助，代价是 FLOPs 的平方增长。

ViT-g/14（10 亿参数，patch 14，分辨率 224 → 256 个 token）和 SigLIP SO400m/14（4 亿参数，patch 14）是 2026 年开放 VLM 的两款主力编码器。

### ViT 的参数量

完整计算在 `code/main.py` 中。对 224 分辨率的 ViT-B/16：

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

在加载 checkpoint 之前，先这样给每个 ViT 估个大概。骨干规模决定了任何下游 VLM 的显存下限。

### 2026 年的生产配置

2026 年大多数开放 VLM 搭载的编码器是原生分辨率的 SigLIP 2 SO400m/14（NaFlex）。它有：
- 4 亿参数。
- patch 尺寸 14，默认分辨率 384 → 每张图 729 个 patch token。
- 图像级任务用平均池化；做 VQA 时全部 729 个 patch 流入 LLM。
- 4 个 register token，在交给 LLM 前丢弃。
- 带图像级缩放的 2D-RoPE，用于原生宽高比。

这个配置里的每个决定都能追溯到一篇你能读到的论文。

```figure
image-patch-tokens
```

## 使用它

`code/main.py` 是一个 patch 分词器和几何计算器。它接收（图像 H、W，patch P，隐藏维度 D，深度 L）并报告：

- 打 patch 后的网格形状和序列长度。
- 一张合成的 8x8 像素玩具图像的 token 序列（走一遍展平 + 投影的路径）。
- 按 patch 嵌入、位置嵌入、Transformer 块和头部拆解的参数量。
- 目标分辨率下每次前向的 FLOPs。
- 一张对比表：ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384。

运行它。把参数数量与已发表数字对上。调整 patch 尺寸和分辨率，感受 token 数量的代价。

## 交付它

本节课产出 `outputs/skill-patch-geometry-reader.md`。给定一个 ViT 配置（patch 尺寸、分辨率、隐藏维度、深度），它给出 token 数量、参数数量和显存估算，并附上依据。每当你为 VLM 挑选视觉骨干时都用这个 skill——它能避免「token 爆炸、LLM 上下文被填满」这类意外。

## 练习

1. 计算 Qwen2.5-VL 在原生 1280x720 输入、patch 尺寸 14 下的 patch token 序列长度。与只用 CLS 的表示相比如何？

2. 一帧 1080p（1920x1080）画面在 patch 14 下产生多少 token？以 30 FPS 计，一段 5 分钟的视频总共有多少视觉 token？哪种代价节省最多：池化、帧采样，还是 token 合并？

3. 用纯 Python 实现对 patch token 的平均池化。验证：对 DINOv2 输出的 196 个 token 做平均池化，结果与请求池化嵌入时模型 `forward` 返回的一致。

4. 阅读《Vision Transformers Need Registers》（arXiv:2309.16588）第 3 节。用两句话描述 register 吸收了哪种伪影，以及它对下游稠密预测为何重要。

5. 修改 `code/main.py` 以支持 patch-n'-pack：给定一组不同分辨率的图像，产出单个打包序列和块对角注意力掩码。学到第 12.06 课时对照验证。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Patch | 「16x16 像素方块」 | 输入图像中固定大小、互不重叠的区域；变成一个 token |
| Patch embedding | 「线性投影」 | 把展平的 patch 像素映射成 D 维向量的共享学习矩阵（或 stride=P 的 Conv2d） |
| CLS token | 「分类 token」 | 前置的可学习向量，其最终隐藏状态代表整张图像；2026 年已是可选项 |
| Register token | 「Sink token」 | 额外的可学习 token，吸收 ViT 在预训练中形成的高范数注意力伪影 |
| Position embedding | 「位置信息」 | 让序列感知顺序的逐位置向量或旋转；2D-RoPE 是现代默认 |
| Grid | 「Patch 网格」 | 给定分辨率和 patch 尺寸下 (H/P) x (W/P) 的二维 patch 阵列 |
| NaFlex | 「原生灵活分辨率」 | SigLIP 2 特性：单个模型无需重新训练即可服务多种宽高比和分辨率 |
| Backbone | 「视觉塔」 | 预训练图像编码器，其 patch-token 输出供给 VLM 中的 LLM |
| Pooling | 「图像级摘要」 | 把 patch token 变成单个向量的策略：CLS、平均、注意力池化或基于 register |
| Patch 14 vs 16 | 「更细 vs 更粗的网格」 | patch 14 每张图产出更多 token，对 OCR 保真度更好但更慢；patch 16 是经典默认 |

## 延伸阅读

- [Dosovitskiy 等——An Image is Worth 16x16 Words（arXiv:2010.11929）](https://arxiv.org/abs/2010.11929)——ViT 原文。
- [He 等——Masked Autoencoders Are Scalable Vision Learners（arXiv:2111.06377）](https://arxiv.org/abs/2111.06377)——MAE，自监督预训练。
- [Oquab 等——DINOv2（arXiv:2304.07193）](https://arxiv.org/abs/2304.07193)——大规模自蒸馏，无需标签。
- [Darcet 等——Vision Transformers Need Registers（arXiv:2309.16588）](https://arxiv.org/abs/2309.16588)——register token 与伪影分析。
- [Tschannen 等——SigLIP 2（arXiv:2502.14786）](https://arxiv.org/abs/2502.14786)——2026 年的默认视觉塔。
- [Zhai 等——Scaling Vision Transformers（arXiv:2106.04560）](https://arxiv.org/abs/2106.04560)——经验扩展规律。
