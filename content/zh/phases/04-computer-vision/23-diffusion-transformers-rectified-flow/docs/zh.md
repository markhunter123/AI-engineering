# Diffusion Transformer 与 Rectified Flow

> U-Net 并非扩散模型的核心秘诀。把它换成 transformer，将噪声调度改为直线 flow，你就得到了 SD3、FLUX 以及 2026 年几乎所有的文本到图像模型。

**Type:** Learn + Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 10 (Diffusion DDPM), Phase 4 Lesson 14 (ViT), Phase 7 Lesson 02 (Self-Attention)
**Time:** ~75 minutes

## 学习目标

- 梳理从 U-Net DDPM（第 10 课）到 Diffusion Transformer（DiT）、MMDiT（SD3）、单双流 DiT（FLUX）的演进
- 解释 rectified flow：为什么噪声与数据之间的直线路径能让模型在 20 步而非 1000 步内采样
- 实现一个微型 DiT 块和一个 rectified-flow 训练循环，均不到 100 行
- 按架构、参数量与许可证区分 SD3、FLUX.1-dev、FLUX.1-schnell、Z-Image、Qwen-Image 等模型变体

## 问题

第 10 课用 U-Net 去噪器构建了 DDPM。这一配方在 2020-2023 年占主导：U-Net + beta 调度 + 噪声预测损失。它催生了 Stable Diffusion 1.5、2.1 和 DALL-E 2。

2026 年每个 SOTA 文本到图像模型都已超越它。Stable Diffusion 3、FLUX、SD4、Z-Image、Qwen-Image、Hunyuan-Image 都不使用 U-Net。它们使用 Diffusion Transformer（DiT）。SD3 和 FLUX 还把 DDPM 噪声调度换成了 rectified flow，它把从噪声到数据的路径拉直，使一致性模型或蒸馏变体能在 1-4 步内推理。

这次转变很重要，因为它是基于扩散的图像生成变得可控、提示准确（SD3/SD4 解决了文字渲染）且生产级快速的原因。理解 DiT + rectified flow，就是理解 2026 年的生成式图像栈。

## 概念

### 从 U-Net 到 transformer

```mermaid
flowchart LR
    subgraph UNET["DDPM U-Net (2020)"]
        U1["Conv encoder"] --> U2["Conv bottleneck"] --> U3["Conv decoder"]
    end
    subgraph DIT["DiT (2023)"]
        D1["Patch embed"] --> D2["Transformer blocks"] --> D3["Unpatchify"]
    end
    subgraph MMDIT["MMDiT (SD3, 2024)"]
        M1["Text stream"] --> M3["Joint attention<br/>(separate weights per modality)"]
        M2["Image stream"] --> M3
    end
    subgraph FLUX["FLUX (2024)"]
        F1["Double-stream blocks<br/>(text + image separate)"] --> F2["Single-stream blocks<br/>(concat + shared weights)"]
    end

    style UNET fill:#e5e7eb,stroke:#6b7280
    style DIT fill:#dbeafe,stroke:#2563eb
    style MMDIT fill:#fef3c7,stroke:#d97706
    style FLUX fill:#dcfce7,stroke:#16a34a
```

- **DiT**（Peebles & Xie，2023）—— 用类似 ViT 的 transformer 替换 U-Net，在 latent patch 上操作。通过 adaptive layer norm（AdaLN）进行条件化。
- **MMDiT**（SD3，Esser et al.，2024）—— 文本和图像 token 使用独立的权重流，共享一次 joint attention。
- **FLUX**（Black Forest Labs，2024）—— 前 N 个块像 SD3 一样双流，后面把 token 拼接并共享权重（单流），以在更大深度上提高效率。
- **Z-Image**（2025）—— 一个 6B 参数的高效单流 DiT，挑战「规模至上」。

### 一段话说清 rectified flow

DDPM 把前向过程定义为噪声 SDE，`x_t` 被逐步污染。学到的反向是第二条 SDE，需要 1000 个小步求解。

Rectified flow 定义了干净数据与纯噪声之间的**直线**插值：

```math
x_t = (1 - t) \, x_0 + t \, \epsilon, \qquad t \in [0, 1]
```

训练网络预测速度 `v_theta(x_t, t) = epsilon - x_0`——沿从干净数据到噪声的直线路径的前向方向（`dx_t/dt`）。采样时，把这个速度反向积分，从噪声走向数据。得到的 ODE 更接近直线，因此采样所需积分步数远少于 DDPM。

SD3 称其为 **Rectified Flow Matching**。FLUX、Z-Image 和大多数 2026 年模型使用相同目标。典型推理：20-30 个 Euler 步（确定性），对比旧 DDPM 机制下 50+ 个 DDIM 步。蒸馏 / turbo / schnell / LCM 变体把它降到 1-4 步。

### AdaLN 条件化

DiT 通过**adaptive layer norm**对时间步和类别 / 文本进行条件化：从条件向量预测 `scale` 和 `shift`，在 LayerNorm 之后应用。比 U-Net 中的 FiLM 式调制更简洁，也是每个现代 DiT 的默认做法。

```
cond -> MLP -> (scale, shift, gate)
norm(x) * (1 + scale) + shift, then residual add * gate
```

### SD3 与 FLUX 中的文本编码器

- **SD3** 使用三个文本编码器：两个 CLIP 模型 + T5-XXL。Embedding 拼接后作为文本条件输入图像流。
- **FLUX** 使用一个 CLIP-L + T5-XXL。
- **Qwen-Image / Z-Image** 变体使用自研文本编码器，与其基础 LLM 对齐。

文本编码器是 SD3/FLUX 的提示理解远超 SD1.5 的重要原因。仅 T5-XXL 就有 4.7B 参数。

### 无分类器引导仍然适用

Rectified flow 改变的是采样器，不是条件化机制。无分类器引导（训练时以 10% 概率丢弃文本，推理时混合条件与无条件预测）与 rectified flow 配合方式完全相同。大多数 2026 年模型使用 3.5-5 的引导尺度，低于 SD1.5 的 7.5，因为 rectified-flow 模型默认更紧密地遵循提示。

### Consistency、Turbo、Schnell、LCM

四个名字，同一思想：把慢速多步模型蒸馏成快速少步模型。

- **LCM（Latent Consistency Model）** —— 训练一个学生，从任意中间 `x_t` 一步预测最终 `x_0`。
- **SDXL Turbo / FLUX schnell** —— 用对抗扩散蒸馏训练的 1-4 步模型。
- **SD Turbo** —— OpenAI 风格 Consistency Model 在 latent 扩散上的适配。

任何新模型的生产部署都会同时发布「完整质量」checkpoint 和「turbo / schnell」变体。Schnell（德语「快速」，Black Forest Labs 的命名惯例）可在 1-4 步内运行，适合实时流程。

### 2026 年模型格局

| Model | Size | Architecture | License |
|-------|------|--------------|---------|
| Stable Diffusion 3 Medium | 2B | MMDiT | SAI Community |
| Stable Diffusion 3.5 Large | 8B | MMDiT | SAI Community |
| FLUX.1-dev | 12B | Double + Single Stream DiT | non-commercial |
| FLUX.1-schnell | 12B | same, distilled | Apache 2.0 |
| FLUX.2 | — | iterated FLUX.1 | mixed |
| Z-Image | 6B | S3-DiT (Scalable Single-Stream) | permissive |
| Qwen-Image | ~20B | DiT + Qwen text tower | Apache 2.0 |
| Hunyuan-Image-3.0 | ~80B | DiT | research |
| SD4 Turbo | 3B | DiT + distillation | SAI Commercial |

FLUX.1-schnell 是 2026 年开源默认。Z-Image 是效率领袖。FLUX.2 和 SD4 是当前质量顶尖。

### 为什么这次范式转变重要

DDPM + U-Net 能工作。DiT + rectified flow **工作得更好、更快、扩展更干净**。这次转变类似 NLP 中从 RNN 到 transformer 的过渡：两种架构都能解决同一问题，但 transformer 能规模化，于是占据主导。2026 年每篇关于图像、视频或 3D 生成的论文都使用 DiT 形状的去噪器，并且通常使用 rectified flow 目标。U-Net DDPM 现在主要用于教学（第 10 课）。

```figure
cv3-rectified-flow
```

## 构建它

### 第 1 步：带 AdaLN 的 DiT 块

```python
import torch
import torch.nn as nn


class AdaLNZero(nn.Module):
    """
    Adaptive LayerNorm with a gate. Predicts (scale, shift, gate) from the conditioning.
    Init such that the whole block starts as identity ("zero init").
    """

    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Linear(cond_dim, dim * 3)
        nn.init.zeros_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)

    def forward(self, x, cond):
        scale, shift, gate = self.mlp(cond).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return h, gate.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, dim=192, heads=3, mlp_ratio=4, cond_dim=192):
        super().__init__()
        self.adaln1 = AdaLNZero(dim, cond_dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.adaln2 = AdaLNZero(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, cond):
        h, gate1 = self.adaln1(x, cond)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1 * a
        h, gate2 = self.adaln2(x, cond)
        x = x + gate2 * self.mlp(h)
        return x
```

`AdaLNZero` 因为 MLP 权重初始化为零，整体以恒等映射起步。训练把块从恒等映射推开；这对稳定深度 transformer 扩散模型效果显著。

### 第 2 步：微型 DiT

```python
def timestep_embedding(t, dim):
    import math
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TinyDiT(nn.Module):
    def __init__(self, image_size=16, patch_size=2, in_channels=3, dim=96, depth=4, heads=3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, cond_dim=dim) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.head = nn.Linear(dim, patch_size * patch_size * in_channels)

    def forward(self, x, t):
        n = x.size(0)
        x = self.patch(x)
        x = x.flatten(2).transpose(1, 2) + self.pos
        t_emb = self.time_mlp(timestep_embedding(t, self.pos.size(-1)))
        for blk in self.blocks:
            x = blk(x, t_emb)
        x = self.norm_out(x)
        x = self.head(x)
        return self._unpatchify(x, n)

    def _unpatchify(self, x, n):
        p = self.patch_size
        h = w = int(self.num_patches ** 0.5)
        x = x.view(n, h, w, p, p, -1).permute(0, 5, 1, 3, 2, 4).reshape(n, -1, h * p, w * p)
        return x
```

### 第 3 步：Rectified flow 训练

```python
import torch.nn.functional as F

def rectified_flow_train_step(model, x0, optimizer, device):
    model.train()
    x0 = x0.to(device)
    n = x0.size(0)
    t = torch.rand(n, device=device)
    epsilon = torch.randn_like(x0)
    x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * epsilon

    target_velocity = epsilon - x0
    pred_velocity = model(x_t, t)

    loss = F.mse_loss(pred_velocity, target_velocity)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

与 DDPM 的噪声预测损失（第 10 课）对比：结构相同，目标不同。我们不是预测噪声 `epsilon`，而是预测**速度** `epsilon - x_0`，它沿直线插值从数据指向噪声。

### 第 4 步：Euler 采样器

Rectified flow 是一个 ODE。Euler 方法最简单，对于训练良好的 rectified-flow 模型，20+ 步时几乎与高阶求解器一样精确。

```python
@torch.no_grad()
def rectified_flow_sample(model, shape, steps=20, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    t = torch.ones(shape[0], device=device)
    for _ in range(steps):
        v = model(x, t)
        x = x - dt * v
        t = t - dt
    return x
```

20 步。在训练好的模型上，可产生与 1000 步 DDPM 相媲美的样本。

### 第 5 步：端到端冒烟测试

```python
import numpy as np

def synthetic_blobs(num=200, size=16, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((num, 3, size, size), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    for i in range(num):
        cx, cy = rng.uniform(4, size - 4, size=2)
        r = rng.uniform(2, 4)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        colour = rng.uniform(-1, 1, size=3)
        for c in range(3):
            out[i, c][mask] = colour[c]
    return torch.from_numpy(out)
```

用 rectified flow 在这个数据集上训练 `TinyDiT`。500 步后，采样输出应呈现淡淡的彩色 blob。

## 使用它

对于用 FLUX / SD3 / Z-Image 进行真实图像生成，`diffusers` 为每个模型提供统一 API：

```python
from diffusers import FluxPipeline, StableDiffusion3Pipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
).to("cuda")

out = pipe(
    prompt="a golden retriever surfing a tsunami, hyperrealistic, studio lighting",
    guidance_scale=0.0,           # schnell was trained without CFG
    num_inference_steps=4,
    max_sequence_length=256,
).images[0]
out.save("surf.png")
```

三行代码，`FLUX.1-schnell` 只需 4 步。把模型 ID 换成 `black-forest-labs/FLUX.1-dev` 可获得更高质量，20-30 步并配合 CFG。

对于 SD3：

```python
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.bfloat16,
).to("cuda")
out = pipe(prompt, guidance_scale=3.5, num_inference_steps=28).images[0]
```

## 交付它

本节课产出：

- `outputs/prompt-dit-model-picker.md` —— 根据质量、延迟和许可证约束，在 SD3、FLUX.1-dev、FLUX.1-schnell、Z-Image、SD4 Turbo 之间做选择。
- `outputs/skill-rectified-flow-trainer.md` —— 编写完整的 rectified flow 训练循环，包括 AdaLN DiT 和 Euler 采样。

## 练习

1. **（简单）** 按上面方法在合成 blob 数据集上训练 TinyDiT 500 步。比较 10、20、50 个 Euler 步产生的样本。
2. **（中等）** 添加文本条件化：把一个可学习的类别 embedding 拼接到时间 embedding（按颜色划分 10 个 blob「类别」）。对类别 0、5、9 采样，验证颜色匹配。
3. **（困难）** 计算 rectified-flow 版本与 DDPM 版本在相同规模网络、相同数据、相同步数下生成样本之间的 Fréchet 距离（FID 代理）。报告哪个收敛更快。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| DiT | "Diffusion transformer" | Transformer that replaces the U-Net as the diffusion denoiser; operates on patchified latents |
| AdaLN | "Adaptive layer norm" | Timestep/text conditioning via learned scale, shift, gate applied after LayerNorm; standard in every modern DiT |
| MMDiT | "Multi-modal DiT (SD3)" | Separate weight streams for text and image tokens that share a joint self-attention |
| Single-stream / double-stream | "FLUX trick" | First $N$ blocks double-stream (separate weights per modality), later blocks single-stream (concat + shared weights) for efficiency |
| Rectified flow | "Straight-line noise-to-data" | Linear interpolation between data and noise; network predicts velocity; fewer ODE steps needed at inference |
| Velocity target | "epsilon - x_0" | The regression target in rectified flow; points from clean data to noise |
| CFG guidance | "classifier-free guidance" | Mix conditional and unconditional predictions; still used in rectified-flow models |
| Schnell / turbo / LCM | "1-4 step distillation" | Small-step variants distilled from full-quality models; production real-time |

## 延伸阅读

- [Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748) —— DiT 论文
- [Scaling Rectified Flow Transformers (Esser et al., SD3 paper)](https://arxiv.org/abs/2403.03206) —— 大规模 MMDiT 与 rectified flow
- [FLUX.1 model card and technical report (Black Forest Labs)](https://huggingface.co/black-forest-labs/FLUX.1-dev) —— 双流 + 单流细节
- [Z-Image: Efficient Image Generation Foundation Model (2025)](https://arxiv.org/html/2511.22699v1) —— 6B 单流 DiT
- [Elucidating the Design Space of Diffusion (Karras et al., 2022)](https://arxiv.org/abs/2206.00364) —— 扩散设计权衡的权威参考
- [Latent Consistency Models (Luo et al., 2023)](https://arxiv.org/abs/2310.04378) —— LCM-LoRA 如何实现 4 步推理
