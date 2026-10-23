# 3D 视觉 —— 点云与 NeRF

> 3D 视觉有两种形态：点云是传感器的原始输出，NeRF 是学习得到的体积场。两者都在回答「空间中有什么、在哪里」。

**Type:** Learn + Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 1 Lesson 12 (Tensor Operations)
**Time:** ~45 minutes

## 学习目标

- 区分显式 3D 表示（点云、网格、体素）与隐式 3D 表示（有符号距离场、NeRF），以及各自的适用场景
- 理解 PointNet 的对称函数技巧：如何让神经网络对无序点集保持置换不变性
- 追踪 NeRF 的一次前向传播：光线投射、体积渲染、位置编码、MLP 密度+颜色头
- 使用 `nerfstudio` 或 `instant-ngp`，从少量带位姿的图像进行预训练式 3D 重建

## 问题

相机产出 2D 图像。LIDAR 产出没有顺序的一组 3D 点。运动恢复结构（SfM）pipeline 产出稀疏的 3D 关键点云。NeRF 从少数带位姿的图像重建出完整 3D 场景。这些都是「视觉」，但都不像 CNN 想要的稠密张量。

3D 视觉之所以重要，是因为几乎所有高价值机器人任务都在 3D 中运行：抓取、避障、导航、AR 遮挡、3D 内容捕获。只懂 2D 图像的视觉工程师会被挡在增长最快的领域之外（AR/VR 内容、机器人、自动驾驶堆栈、房地产或建筑领域的 NeRF 3D 重建）。

两种表示因不同原因占据主导。点云是传感器免费给你的东西。NeRF 及其后继者（3D 高斯泼溅、神经 SDF）是你让神经网络学习一个场景时得到的东西。

## 概念

### 点云

点云是 $\mathbb{R}^3$ 中 $N$ 个点的无序集合，每个点可选地带有特征（颜色、强度、法向）。

```
cloud = [
  (x1, y1, z1, r1, g1, b1),
  (x2, y2, z2, r2, g2, b2),
  ...
  (xN, yN, zN, rN, gN, bN),
]
```

没有网格，没有连接关系。两个性质让神经网络难以处理：

- **置换不变性** —— 输出不能依赖点的顺序。
- **可变的 N** —— 同一个模型必须处理不同大小的点云。

PointNet（Qi 等人，2017）用一个想法同时解决了两个问题：对每个点应用共享 MLP，再用对称函数（max pool）聚合。结果是一个不依赖顺序的固定长度向量。

```math
f(P) = \max_{p \in P} \mathop{\mathrm{MLP}}(p)
```

这就是 PointNet 的核心。更深层的变体（PointNet++、Point Transformer）加入了分层采样和局部聚合，但对称函数技巧始终不变。

### PointNet 架构

```mermaid
flowchart LR
    PTS["N points<br/>(x, y, z)"] --> MLP1["shared MLP<br/>(64, 64)"]
    MLP1 --> MLP2["shared MLP<br/>(64, 128, 1024)"]
    MLP2 --> MAX["max pool<br/>(symmetric)"]
    MAX --> FEAT["global feature<br/>(1024,)"]
    FEAT --> FC["MLP classifier"]
    FC --> CLS["class logits"]

    style MLP1 fill:#dbeafe,stroke:#2563eb
    style MAX fill:#fef3c7,stroke:#d97706
    style CLS fill:#dcfce7,stroke:#16a34a
```

「共享 MLP」指同一个 MLP 独立运行在每一点上。为效率起见，实现时把它作为沿点维度做的 1x1 卷积。

### 神经辐射场（NeRF）

NeRF（Mildenhall 等人，2020）把「能否从 N 张照片重建 3D 场景？」这个问题，回答成「让神经网络本身就是场景」。网络把 `(x, y, z, viewing_direction)` 映射为 `(density, colour)`。渲染新视角就是对网络做光线投射循环。

```
NeRF MLP:  (x, y, z, theta, phi) -> (sigma, r, g, b)

To render a pixel (u, v) of a new view:
  1. Cast a ray from the camera through pixel (u, v)
  2. Sample points along the ray at distances t_1, t_2, ..., t_N
  3. Query the MLP at each point
  4. Composite the colours weighted by (1 - exp(-sigma * dt))
  5. The sum is the rendered pixel colour
```

损失函数比较渲染像素与训练照片中的真实像素。反向传播穿过渲染步骤更新 MLP。没有 3D 真实标签，没有显式几何——场景就存储在 MLP 权重中。

### NeRF 中的位置编码

一个普通 MLP 直接处理 `(x, y, z)` 时，由于 MLP 在频谱上偏向低频，无法表示高频细节。NeRF 的做法是在进入 MLP 前，把每个坐标编码成傅里叶特征向量：

```math
\gamma(p) = \left(\sin(2^0 \pi p), \cos(2^0 \pi p), \sin(2^1 \pi p), \cos(2^1 \pi p), \ldots\right)
```

最高到 $L=10$ 个频率级别。这与 transformer 用于位置编码的技巧相同，也会在扩散时间条件（第 10 课）中再次出现。没有它，NeRF 看起来会很糊。

### 体积渲染

```math
\begin{aligned}
C(r) &= \sum_i T_i \left(1 - \exp(-\sigma_i \delta_i)\right) c_i \\[4pt]
T_i &= \exp\left(-\sum_{j<i} \sigma_j \delta_j\right) \\
\delta_i &= t_{i+1} - t_i
\end{aligned}
```

`T_i` 是透光率——有多少光能活到第 $i$ 个点。`(1 - exp(-sigma_i * delta_i))` 是第 $i$ 个点的不透明度。`c_i` 是颜色。最终像素是沿光线加权求和。

### 什么接替了 NeRF

纯 NeRF 训练慢（小时级）、渲染慢（秒/图）。后续发展：

- **Instant-NGP**（2022）—— 用哈希网格编码替换 MLP 的位置输入；秒级训练。
- **Mip-NeRF 360** —— 处理无界场景和抗锯齿。
- **3D 高斯泼溅**（2023）—— 用数百万个 3D 高斯替换体积场；分钟级训练、实时渲染。当前生产默认。

2026 年几乎每一个真正的 NeRF 产品，实际都是 3D 高斯泼溅。但心智模型仍然是 NeRF。

### 数据集与基准

- **ShapeNet** —— 以点云形式对 3D CAD 模型做分类和分割。
- **ScanNet** —— 真实室内扫描，用于分割。
- **KITTI** —— 户外 LIDAR 点云，用于自动驾驶。
- **NeRF Synthetic** / **Blended MVS** —— 用于视图合成的带位姿图像数据集。
- **Mip-NeRF 360 数据集** —— 无界真实场景。

```figure
nerf-rays
```

## 构建它

### 第 1 步：PointNet 分类器

```python
import torch
import torch.nn as nn

class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),    nn.BatchNorm1d(64),   nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),   nn.BatchNorm1d(64),   nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),  nn.BatchNorm1d(128),  nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 512),   nn.BatchNorm1d(512),  nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),    nn.BatchNorm1d(256),  nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (N, 3, num_points) — transposed for Conv1d
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = torch.max(x, dim=-1)[0]       # (N, 1024)
        return self.head(x)

pts = torch.randn(4, 3, 1024)
net = PointNet(num_classes=10)
print(f"output: {net(pts).shape}")
print(f"params: {sum(p.numel() for p in net.parameters()):,}")
```

约 160 万参数。每朵点云 1024 个点。

### 第 2 步：位置编码

```python
def positional_encoding(x, L=10):
    """
    x: (..., D) -> (..., D * 2 * L)
    """
    freqs = 2.0 ** torch.arange(L, dtype=x.dtype, device=x.device)
    args = x.unsqueeze(-1) * freqs * 3.141592653589793
    sinc = torch.cat([args.sin(), args.cos()], dim=-1)
    return sinc.reshape(*x.shape[:-1], -1)

x = torch.randn(5, 3)
y = positional_encoding(x, L=10)
print(f"input:  {x.shape}")
print(f"encoded: {y.shape}     # (5, 60)")
```

乘以 `2^l * pi` 可得到越来越高的频率。

### 第 3 步：迷你 NeRF MLP

```python
class TinyNeRF(nn.Module):
    def __init__(self, L_pos=10, L_dir=4, hidden=128):
        super().__init__()
        self.L_pos = L_pos
        self.L_dir = L_dir
        pos_dim = 3 * 2 * L_pos
        dir_dim = 3 * 2 * L_dir
        self.trunk = nn.Sequential(
            nn.Linear(pos_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
        )
        self.sigma = nn.Linear(hidden, 1)
        self.color = nn.Sequential(
            nn.Linear(hidden + dir_dim, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 3), nn.Sigmoid(),
        )

    def forward(self, x, d):
        x_enc = positional_encoding(x, self.L_pos)
        d_enc = positional_encoding(d, self.L_dir)
        h = self.trunk(x_enc)
        sigma = torch.relu(self.sigma(h)).squeeze(-1)
        rgb = self.color(torch.cat([h, d_enc], dim=-1))
        return sigma, rgb

nerf = TinyNeRF()
x = torch.randn(128, 3)
d = torch.randn(128, 3)
s, c = nerf(x, d)
print(f"sigma: {s.shape}   rgb: {c.shape}")
```

相比原始 NeRF（两个深度为 8 的 MLP trunk）非常迷你，但足够展示架构。

### 第 4 步：沿光线做体积渲染

```python
def volumetric_render(sigma, rgb, t_vals):
    """
    sigma: (..., N_samples)
    rgb:   (..., N_samples, 3)
    t_vals: (N_samples,) distances along the ray
    """
    delta = torch.cat([t_vals[1:] - t_vals[:-1], torch.full_like(t_vals[:1], 1e10)])
    alpha = 1.0 - torch.exp(-sigma * delta)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    weights = alpha * trans
    rendered = (weights.unsqueeze(-1) * rgb).sum(dim=-2)
    depth = (weights * t_vals).sum(dim=-1)
    return rendered, depth, weights


N = 64
t_vals = torch.linspace(2.0, 6.0, N)
sigma = torch.rand(N) * 0.5
rgb = torch.rand(N, 3)
rendered, depth, weights = volumetric_render(sigma, rgb, t_vals)
print(f"rendered colour: {rendered.tolist()}")
print(f"depth:           {depth.item():.2f}")
```

一条光线，64 个采样点，合成为一个 RGB 像素和一个深度值。

## 使用它

实际工作中：

- `nerfstudio`（Tancik 等人）—— NeRF / Instant-NGP / 高斯泼溅的当前参考库。命令行 + web 查看器。
- `pytorch3d`（Meta）—— 可微渲染、点云工具、网格操作。
- `open3d` —— 点云处理、配准、可视化。

部署方面，3D 高斯泼溅已大幅取代纯 NeRF，因为渲染速度快了约 100 倍。重建质量相当。

## 交付它

本节课产出：

- `outputs/prompt-3d-task-router.md` —— 一个 prompt，根据任务和输入数据把任务路由到正确的 3D 表示（点云、网格、体素、NeRF、高斯泼溅）。
- `outputs/skill-point-cloud-loader.md` —— 一个 skill，为 .ply / .pcd / .xyz 文件写出正确的 PyTorch `Dataset`，包含归一化、中心化和点采样。

## 练习

1. **（简单）** 证明 PointNet 是置换不变的：对同一朵点云运行两次，其中一次打乱点顺序。验证两次输出在浮点误差范围内相同。
2. **（中等）** 实现一个最小光线生成函数：给定相机内参和位姿，为 H x W 图像的每个像素生成光线原点和方向。
3. **（困难）** 在彩色立方体渲染视图的合成数据集（通过可微渲染或简单光线追踪生成）上训练 TinyNeRF。报告第 1、10、100 个 epoch 的渲染损失。到第几个 epoch 模型能产生可辨认的视角？

## 关键术语

| 术语 | 常说的 | 实际含义 |
|---|---|---|
| 点云（Point cloud） | "LIDAR 出来的 3D 点" | 无序的 (x, y, z) 集合，每点可选带特征 |
| PointNet | "第一个点云神经网络" | 每点共享 MLP + 对称 max pool；结构本身就置换不变 |
| NeRF | "MLP 即场景" | 网络把 (x, y, z, dir) 映射为 (density, colour)；通过光线投射渲染 |
| 位置编码（Positional encoding） | "傅里叶特征" | 把每个坐标编码成多频率 sin/cos，克服 MLP 低频偏向 |
| 体积渲染（Volumetric rendering） | "光线积分" | 用透光率和 alpha 把沿光线的采样点合成为一个像素 |
| Instant-NGP | "哈希网格 NeRF" | 用多分辨率哈希网格替换 NeRF 的坐标 MLP；快 100-1000 倍 |
| 3D 高斯泼溅（3D Gaussian splatting） | "几百万个高斯" | 场景 = 一堆 3D 高斯；实时渲染、分钟级训练 |
| SDF | "有符号距离场" | 返回到最近表面带符号距离的函数；另一种隐式表示 |

## 延伸阅读

- [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593) —— 置换不变的点云分类器
- [NeRF (Mildenhall et al., 2020)](https://arxiv.org/abs/2003.08934) —— 让从照片做 3D 重建变成神经网络问题
- [Instant-NGP (Müller et al., 2022)](https://arxiv.org/abs/2201.05989) —— 哈希网格，1000 倍加速
- [3D Gaussian Splatting (Kerbl et al., 2023)](https://arxiv.org/abs/2308.04079) —— 在生产中取代 NeRF 的架构
