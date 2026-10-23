# 图像生成 —— 扩散模型

> 扩散模型学习去噪。训练它从一张带噪图像中去除一点点噪声，再反向重复上千次，你就得到了一个图像生成器。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 07 (U-Net), Phase 1 Lesson 06 (Probability), Phase 3 Lesson 06 (Optimizers)
**Time:** ~75 minutes

## 学习目标

- 推导前向加噪过程 `x_0 -> x_1 -> ... -> x_T`，并解释闭式 `q(x_t | x_0)` 为何对任意 t 都成立
- 实现 DDPM 风格的训练目标：回归每一步加入的噪声，并构造从纯噪声走回图像的采样器
- 构建一个可在 CPU 上训练的时间条件 U-Net，对任意时间步预测噪声
- 解释 DDPM 与 DDIM 采样的区别，以及各自适用的场景（第 23 课会更深入地讲 flow matching 与 rectified flow）

## 问题

GAN 一次性生成：噪声进，图像出，一次前向传播。它很快，但很难训练。扩散模型迭代生成：从纯噪声出发，一小步一小步去噪，图像逐渐显现。它很慢，但很容易训练。过去五年，后一个特性占据了主导：任何小团队都能训练扩散模型并得到合理样本；GAN 训练则是一门需要多年失败经验才能掌握的技艺。

除了训练稳定，扩散的迭代结构还解锁了现代图像生成的全部能力：文本条件、inpainting、图像编辑、超分辨率、可控风格。采样循环的每一步都是注入新约束的位置。这就是 Stable Diffusion、Imagen、DALL-E 3、Midjourney，以及你使用的所有可控图像模型都基于扩散的原因。

本节课构建最小 DDPM：前向加噪、反向去噪、训练循环。下一课（Stable Diffusion）会把它接入一个生产系统，包含 VAE、文本编码器和 classifier-free guidance。

## 概念

### 前向过程

取一张图像 `x_0`，加入少量高斯噪声得到 `x_1`；再加一点得到 `x_2`；持续 $T$ 步，直到 `x_T` 几乎与纯高斯噪声无法区分。

```math
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \; \beta_t I\right)
```

`beta_t` 是一个小的方差调度，通常从 0.0001 线性增加到 0.02，$T=1000$ 步。每一步都轻微压缩信号并注入新噪声。

### 闭式跳跃

一步一步加噪是一个马尔可夫链，但数学上可以折叠：你可以直接从 `x_0` 一步采样出 `x_t`。

```math
\begin{aligned}
&\text{Define } \alpha_t = 1 - \beta_t \\
&\text{Define } \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s \\[6pt]
&\text{Then: } q(x_t \mid x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} \, x_0, \; (1 - \bar{\alpha}_t) I\right) \\[6pt]
&\text{Equivalently: } x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon \quad \text{where } \epsilon \sim \mathcal{N}(0, I)
\end{aligned}
```

正是这个等式让扩散模型变得实用。训练时你随机选一个 `t`，直接从 `x_0` 采样出 `x_t`，一步完成训练——无需模拟完整马尔可夫链。

### 反向过程

前向过程是固定的。反向过程 `p(x_{t-1} | x_t)` 才是神经网络学习的。扩散模型并不直接预测 `x_{t-1}`，而是预测第 $t$ 步加入的噪声 `epsilon`，再用公式从噪声推出 `x_{t-1}`。

```mermaid
flowchart LR
    X0["x_0<br/>(clean image)"] --> Q1["q(x_t|x_0)<br/>add noise"]
    Q1 --> XT["x_t<br/>(noisy)"]
    XT --> MODEL["model(x_t, t)"]
    MODEL --> EPS["predicted epsilon"]
    EPS --> LOSS["MSE against<br/>true epsilon"]

    XT -.->|sampling| STEP["p(x_{t-1}|x_t)"]
    STEP -.-> XT1["x_{t-1}"]
    XT1 -.->|repeat 1000x| X0S["x_0 (sampled)"]

    style X0 fill:#dcfce7,stroke:#16a34a
    style MODEL fill:#fef3c7,stroke:#d97706
    style LOSS fill:#fecaca,stroke:#dc2626
    style X0S fill:#dbeafe,stroke:#2563eb
```

### 训练损失

每个训练步骤：

1. 采样一张真实图像 `x_0`。
2. 从 $[1, T]$ 均匀采样一个时间步 `t`。
3. 采样噪声 `epsilon ~ N(0, I)`。
4. 计算 `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`。
5. 用网络预测 `epsilon_theta(x_t, t)`。
6. 最小化 `|| epsilon - epsilon_theta(x_t, t) ||^2`。

仅此而已。神经网络学习预测任意时间步的噪声。损失是 MSE。没有对抗博弈，没有坍塌，没有振荡。

### 采样器（DDPM）

生成时：从 `x_T ~ N(0, I)` 出发，一步一步反向走。

```
for t = T, T-1, ..., 1:
    eps = model(x_t, t)
    x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps) + sqrt(beta_t) * z
    where z ~ N(0, I) if t > 1, else 0
return x_0
```

关键是：尽管反向条件一般情况下没有闭式解，但对这种特定的高斯前向过程它有。那些看起来复杂的系数就是贝叶斯规则给出的结果。

### 为什么是 1000 步

前向噪声调度的设计目标是：每一步只加足够的噪声，使得反向一步仍近似高斯。步数太少，反向步远离高斯，网络难以建模；步数太多，采样变得昂贵而收益递减。$T=1000$ 配合线性调度是 DDPM 默认。

### DDIM：采样速度提升 20 倍

训练不变。采样改变。DDIM（Song 等人，2020）定义了一个确定性反向过程，可以在不重训练的情况下跳过时间步。用 DDIM 50 步采样即可达到近 1000 步 DDPM 的质量。每个生产系统都使用 DDIM 或更快的变体（DPM-Solver、Euler ancestral）。

### 时间条件

网络 `epsilon_theta(x_t, t)` 需要知道当前在去除哪个时间步的噪声。现代扩散模型通过正弦时间嵌入（与 Transformer 位置编码相同思想）注入 `t`，并在 U-Net 的每一层加到特征图上。

```
t_embedding = sinusoidal(t)
feature_map += MLP(t_embedding)
```

没有时间条件，网络只能从图像本身猜测噪声水平，这虽然也能工作，但样本效率低得多。

```figure
cv-diffusion-image
```

## 构建它

### 第 1 步：噪声调度

```python
import torch

def linear_beta_schedule(T=1000, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)


def precompute_schedule(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
    }

schedule = precompute_schedule(linear_beta_schedule(T=1000))
```

一次性预计算，训练和采样时按索引取出。

### 第 2 步：前向扩散（q_sample）

```python
def q_sample(x0, t, noise, schedule):
    sqrt_a = schedule["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
    return sqrt_a * x0 + sqrt_one_minus_a * noise
```

一行闭式表达。`t` 是一个时间步 batch，batch 中每张图像对应一个时间步。

### 第 3 步：一个迷你时间条件 U-Net

```python
import torch.nn as nn
import torch.nn.functional as F
import math

def timestep_embedding(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    return emb


class TinyUNet(nn.Module):
    def __init__(self, img_channels=3, base=32, t_dim=64):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, base * 4),
            nn.SiLU(),
            nn.Linear(base * 4, base * 4),
        )
        self.t_dim = t_dim
        self.enc1 = nn.Conv2d(img_channels, base, 3, padding=1)
        self.enc2 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.mid = nn.Conv2d(base * 2, base * 2, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec2 = nn.Conv2d(base * 2, img_channels, 3, padding=1)
        self.time_proj = nn.Linear(base * 4, base * 2)

    def forward(self, x, t):
        t_emb = timestep_embedding(t, self.t_dim)
        t_emb = self.t_mlp(t_emb)
        t_proj = self.time_proj(t_emb)[:, :, None, None]

        h1 = F.silu(self.enc1(x))
        h2 = F.silu(self.enc2(h1)) + t_proj
        h3 = F.silu(self.mid(h2))
        d1 = F.silu(self.dec1(h3))
        d2 = torch.cat([d1, h1], dim=1)
        return self.dec2(d2)
```

两层 U-Net，在瓶颈处注入时间条件。真实图像需要加大深度和宽度。

### 第 4 步：训练循环

```python
def train_step(model, x0, schedule, optimizer, device, T=1000):
    model.train()
    x0 = x0.to(device)
    bs = x0.size(0)
    t = torch.randint(0, T, (bs,), device=device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, schedule)
    pred = model(x_t, t)
    loss = F.mse_loss(pred, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

这就是整个训练循环。没有 GAN 博弈，没有特殊损失，只有一个 MSE 调用。

### 第 5 步：采样器（DDPM）

```python
@torch.no_grad()
def sample(model, schedule, shape, T=1000, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    betas = schedule["betas"].to(device)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"].to(device)
    sqrt_recip_alphas = schedule["sqrt_recip_alphas"].to(device)

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        coef = betas[t] / sqrt_one_minus_a[t]
        mean = sqrt_recip_alphas[t] * (x - coef * eps)
        if t > 0:
            x = mean + torch.sqrt(betas[t]) * torch.randn_like(x)
        else:
            x = mean
    return x
```

生成一批样本需要 1000 次前向传播。真实代码中你会把它换成 DDIM 50 步采样器。

### 第 6 步：DDIM 采样器（确定性，约 20 倍更快）

```python
@torch.no_grad()
def sample_ddim(model, schedule, shape, steps=50, T=1000, device="cpu", eta=0.0):
    model.eval()
    x = torch.randn(shape, device=device)
    alphas_cumprod = schedule["alphas_cumprod"].to(device)

    ts = torch.linspace(T - 1, 0, steps + 1).long()
    for i in range(steps):
        t = ts[i]
        t_prev = ts[i + 1]
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        a_t = alphas_cumprod[t]
        a_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
        sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        dir_xt = torch.sqrt(1 - a_prev - sigma ** 2) * eps
        noise = sigma * torch.randn_like(x) if eta > 0 else 0
        x = torch.sqrt(a_prev) * x0_pred + dir_xt + noise
    return x
```

`eta=0` 是完全确定性的（相同噪声输入总是产生相同输出）。`eta=1` 退化为 DDPM。

## 使用它

生产工作请使用 `diffusers`：

```python
from diffusers import DDPMScheduler, UNet2DModel

unet = UNet2DModel(sample_size=32, in_channels=3, out_channels=3, layers_per_block=2)
scheduler = DDPMScheduler(num_train_timesteps=1000)
```

这个库提供现成的调度器（DDPM、DDIM、DPM-Solver、Euler、Heun）、可配置 U-Net、文生图和图生图 pipeline，以及 LoRA 微调辅助工具。

研究方面，`k-diffusion`（Katherine Crowson）有最忠实的参考实现和最好的采样变体。

## 交付它

本节课产出：

- `outputs/prompt-diffusion-sampler-picker.md` —— 一个 prompt，根据质量目标、延迟预算和条件类型选择 DDPM / DDIM / DPM-Solver / Euler。
- `outputs/skill-noise-schedule-designer.md` —— 一个 skill，给定 T 和目标破坏程度，生成线性、cosine 或 sigmoid beta 调度，并绘制随时间变化的信噪比诊断图。

## 练习

1. **（简单）** 可视化前向过程：取一张图像，绘制 `t in [0, 100, 250, 500, 750, 1000]` 时的 `x_t`。验证 `x_1000` 看起来像纯高斯噪声。
2. **（中等）** 用合成圆形数据集训练 TinyUNet 20 个 epoch，并采样 16 个圆形。比较 DDPM（1000 步）和 DDIM（50 步）采样——从相同噪声种子出发，它们是否产生相似图像？
3. **（困难）** 实现 cosine 噪声调度（Nichol & Dhariwal，2021）：`alpha_bar_t = cos^2((t/T + s) / (1 + s) * pi / 2)`。用相同模型分别在线性和 cosine 调度下训练，证明 cosine 在低步数下产生更好样本。

## 关键术语

| 术语 | 常说的 | 实际含义 |
|---|---|---|
| 前向过程（Forward process） | "随时间加噪声" | 固定的马尔可夫链，把图像在 $T$ 步内腐蚀成高斯噪声 |
| 反向过程（Reverse process） | "一步步去噪" | 学习的分布，从噪声走回图像 |
| epsilon 预测（Epsilon prediction） | "预测噪声" | 训练目标：`epsilon_theta(x_t, t)` 预测第 $t$ 步加入的噪声 |
| beta 调度（Beta schedule） | "噪声量" | 由 $T$ 个小方差组成的序列，定义每步注入多少噪声 |
| $\bar{\alpha}_t$ | "累积保留因子" | 到时间 $t$ 为止 $(1 - \beta_s)$ 的乘积；t 越大，剩余信号越少 |
| DDPM 采样器 | "祖先采样、随机" | 从条件高斯中逐个采样 $x_{t-1}$；1000 步 |
| DDIM 采样器 | "确定性、快速" | 把采样重写为确定性 ODE；20-100 步即可达到相近质量 |
| 时间条件（Time conditioning） | "告诉模型当前是 t" | 把 $t$ 的正弦嵌入注入 U-Net，使其知道噪声水平 |

## 延伸阅读

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) —— 让扩散模型实用化并在 FID 上击败 GAN 的论文
- [Improved DDPM (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672) —— cosine 调度与 v-parameterisation
- [DDIM (Song, Meng, Ermon, 2020)](https://arxiv.org/abs/2010.02502) —— 让实时推理成为可能的确定性采样器
- [Elucidating the Design Space of Diffusion (Karras et al., 2022)](https://arxiv.org/abs/2206.00364) —— 对扩散模型各种设计选择的统一视角；当前最佳参考
