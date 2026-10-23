# 条件 GAN 与 Pix2Pix

> 2014-2017 年第一个大突破，是控制 GAN 生成什么。挂上一个标签、一张图或一句话。Pix2Pix 做了图像版本，在窄领域图生图任务上，它至今仍胜过所有通用文生图模型。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 06 (U-Net), Phase 3 · 07 (CNNs)
**Time:** ~75 minutes

## 问题

无条件 GAN 采样任意人脸。做 demo 有用，生产无用。你想要：*把素描映射成照片*、*把地图映射成航拍图*、*把白天场景映射成夜晚*、*给灰度图上色*。所有这些任务，给定输入图像 $x$，必须输出具有某种语义对应的 $y$。每个 $x$ 都有很多合理的 $y$。均方误差把它们压成一团糊。对抗损失不会，因为「看起来真实」是锐利的。

条件 GAN（Mirza & Osindero, 2014）把条件 $c$ 作为输入加进 $G$ 和 $D$。Pix2Pix（Isola et al., 2017）把它专门化：条件是整张输入图像，生成器是 U-Net，判别器是*基于 patch 的*分类器（PatchGAN），损失是对抗损失 + L1。即便在 2026 年，这套配方在窄领域图生图任务上仍胜过从零训练的文生图模型，因为它在*配对数据*上训练——你手里正好有需要的信号。

## 概念

![Pix2Pix：U-Net 生成器，PatchGAN 判别器](../assets/pix2pix.svg)

**条件 G。** $G(x, z) \to y$。在 Pix2Pix 中，$z$ 是 G 内部的 dropout（没有输入噪声——Isola 发现显式噪声会被忽略）。

**条件 D。** $D(x, y) \to [0, 1]$。输入是*成对*的（条件，输出）。这是关键区别：D 必须判断 $y$ 是否与 $x$ 一致，而不只是 $y$ 看起来是否真实。

**U-Net 生成器。** 编码器-解码器，跨瓶颈有跳跃连接。对于输入输出共享低层结构（边缘、轮廓）的任务至关重要。没有跳跃连接，高频细节会消失。

**PatchGAN 判别器。** D 不输出单个真/假分数，而是输出一个 $N \times N$ 的网格，每个格子判断约 70×70 像素的感受野，再取平均。这是一个马尔可夫随机场假设：真实性是局部的。训练快得多、参数更少、输出更锐利。

**损失。**

```math
\begin{aligned}
\text{loss}_G &= -\log D(x, G(x)) + \lambda \cdot \lVert y - G(x) \rVert_1 \\
\text{loss}_D &= -\log D(x, y) - \log(1 - D(x, G(x)))
\end{aligned}
```

L1 项稳定训练，并把 G 推向已知目标。L1 比 L2 给出更锐利的边缘（中位数，而非均值）。$\lambda = 100$ 是 Pix2Pix 的默认值。

## CycleGAN——没有配对数据时

Pix2Pix 需要成对的 $(x, y)$ 数据。CycleGAN（Zhu et al., 2017）以额外一个损失为代价摆脱了这个要求：*循环一致性*（cycle consistency）损失。两个生成器 $G: X \to Y$ 和 $F: Y \to X$。训练它们使得 $F(G(x)) \approx x$ 且 $G(F(y)) \approx y$。这样你就能在没有配对样本的情况下把马变成斑马、把夏天变成冬天。

2026 年，无配对的图生图大多通过扩散（ControlNet、IP-Adapter）完成，而不是 CycleGAN，但循环一致性的思想在几乎每一篇无配对域适应论文中依然存在。

```figure
gx-patchgan
```

## 构建它

`code/main.py` 在一维数据上实现一个迷你条件 GAN。条件 $c$ 是类别标签（0 或 1）。任务是：给定类别，从对应的条件分布中生成样本。

### 第 1 步：把条件拼接到 G 和 D 的输入

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

One-hot 编码是最简单的方式。更大的模型使用学出来的 embedding、FiLM 调制或交叉注意力。

### 第 2 步：训练条件模型

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器必须匹配*给定条件*下的真实分布，而不是边缘分布。

### 第 3 步：验证每个类别的输出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 常见陷阱

- **条件被忽略。** G 学会了边缘化，D 从不惩罚，因为条件信号太弱。修复：更积极地把条件注入 D（放在浅层，而不只是深层），使用投影判别器（Miyato & Koyama 2018）。
- **L1 权重太低。** G 会飘向任意看起来真实的输出，而不是忠实的输出。Pix2Pix 类任务从 $\lambda \approx 100$ 开始。
- **L1 权重太高。** G 输出模糊，因为 L1 仍然是一种 $L_p$ 范数。训练稳定后把权重退火降下来。
- **D 里泄露了 ground truth。** 把 $(x, y)$ 拼接作为 D 的输入，而不只是 $y$。否则 D 无法检查一致性。
- **按类别的模式崩溃。** 每个类别可能各自独立崩溃。做类别条件多样性检查。

## 使用它

2026 年图生图任务的现状：

| 任务 | 最佳方法 |
|------|---------------|
| 素描 → 照片，同领域，有配对数据 | Pix2Pix / Pix2PixHD（依然快，依然锐利） |
| 素描 → 照片，无配对 | 用 Scribble 条件模型的 ControlNet |
| 语义分割 → 照片 | SPADE / GauGAN2 或 SD + ControlNet-Seg |
| 风格迁移 | 带 IP-Adapter 或 LoRA 的扩散；GAN 方法已是遗留 |
| 深度 → 照片 | Stable Diffusion 上的 ControlNet-Depth |
| 超分辨率 | Real-ESRGAN（GAN）、ESRGAN-Plus 或 SD-Upscale（扩散） |
| 上色 | ColTran、基于扩散的上色器，或 Pix2Pix-color |
| 白天 → 夜晚、季节、天气 | CycleGAN 或基于 ControlNet 的方案 |

Pix2Pix 仍然是合适的工具，当：(a) 你有数千对配对样本，(b) 任务狭窄且可重复，(c) 你需要快速推理。在通用开放领域任务上，扩散胜出。

## 交付它

保存 `outputs/skill-img2img-chooser.md`。Skill 接收任务描述、数据可用情况（配对 vs 无配对、N 个样本）以及延迟/质量预算，输出：方案（Pix2Pix、CycleGAN、ControlNet 变体、SDXL + IP-Adapter）、训练数据要求、推理成本，以及评估协议（LPIPS、FID、任务特定指标）。

## 练习

1. **简单。** 修改 `code/main.py` 加入第三个类别。确认 G 仍能把每个类别的噪声映射到正确的模式。
2. **中等。** 在一维设定下把 L1 换成感知风格的损失（例如用一个小型冻结 D 作为特征提取器）。这会改变条件分布的锐利度吗？
3. **困难。** 在一维设定下草拟一个 CycleGAN：两个分布、两个生成器、循环损失。证明它能在没有配对数据的情况下学会在两个分布之间映射。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Conditional GAN | 「带标签的 GAN」 | G(z, c)、D(x, c)。两个网络都能看到条件。 |
| Pix2Pix | 「图生图 GAN」 | 配对 cGAN，U-Net 做 G，PatchGAN 做 D，加 L1 损失。 |
| U-Net | 「带跳跃连接的编码器-解码器」 | 对称卷积网络；跳跃连接保留高频信息。 |
| PatchGAN | 「局部真实性分类器」 | D 输出逐 patch 分数，而不是全局分数。 |
| CycleGAN | 「无配对图像翻译」 | 两个 G + 循环一致性损失；不需要配对数据。 |
| SPADE | 「GauGAN」 | 用语义图归一化中间激活；分割图到图像。 |
| FiLM | 「特征级线性调制」 | 从条件生成逐特征仿射变换；廉价的条件注入方式。 |

## 生产说明：Pix2Pix 作为延迟受限场景的基线

当你有配对数据且任务狭窄时（素描 → 渲染、语义图 → 照片、白天 → 夜晚），Pix2Pix 的单次推理在延迟上比扩散好一个数量级。生产上的比较通常是：

| 路径 | 步数 | 单张 L4 上 512² 的典型延迟 |
|------|-------|----------------------------------------|
| Pix2Pix（U-Net 前向） | 1 | ~30 ms |
| SD-Inpaint 或 SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

在静态 batch 下（每个请求 FLOPs 相同），Pix2Pix 在吞吐量上胜出。扩散在质量和泛化性上胜出。现代的打法往往是：为窄任务上线一个 Pix2Pix 式的蒸馏模型，再为长尾输入保留一个扩散兜底。

## 延伸阅读

- [Mirza & Osindero（2014）。Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — cGAN 论文。
- [Isola 等（2017）。Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix。
- [Zhu 等（2017）。Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN。
- [Wang 等（2018）。High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD。
- [Park 等（2019）。Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN。
- [Miyato & Koyama（2018）。cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — 投影判别器。
