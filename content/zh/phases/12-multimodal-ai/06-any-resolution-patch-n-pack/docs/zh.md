# 任意分辨率视觉：patch-n'-pack 与 NaFlex

> 真实图像不是 224x224 的正方形。收据是 9:16，图表是 16:9，医学扫描可能是 4096x4096，手机截图是 9:19.5。2024 年之前 VLM 的答案——把所有东西缩放成固定正方形——丢掉了让 OCR、文档理解和高分辨率场景解析得以工作的信号。NaViT（Google，2023）表明你可以把可变分辨率的 patch 打包进单个 Transformer batch，并使用块对角掩码。Qwen2-VL 的 M-RoPE（2024）完全放弃了绝对位置表。LLaVA-NeXT 的 AnyRes 把高分辨率图像平铺成一个基图 + 子图。SigLIP 2 的 NaFlex 变体（2025）现在是开放 VLM 的默认编码器，它们希望用单个 checkpoint 服务所有宽高比。本节课端到端实现 patch-n'-pack。

**Type:** Build
**Languages:** Python (stdlib, patch packer + block-diagonal mask)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 12 · 05 (LLaVA)
**Time:** ~120 minutes

## 学习目标

- 把一个 batch 中可变分辨率图像的 patch 打包成一个序列，并构建块对角注意力掩码。
- 为给定任务在 AnyRes 平铺（LLaVA-NeXT）、NaFlex（SigLIP 2）和 M-RoPE（Qwen2-VL）之间选择。
- 在不缩放的前提下，计算 OCR、图表和摄影的 token 预算。
- 说出方形缩放的三类失败模式：文字被压扁、内容被裁掉、填充浪费 token。

## 问题

Transformer 期望一个序列。一个 batch 是一叠等长序列。如果你的图像都是 224x224，每次都得到 196 个 patch token，不需要填充，搞定。用 224 训练，用 224 推理，再也不用想分辨率。

世界不配合。文档是纵向的（8.5x11 英寸，约 2:3）。图表的截图是横向的（16:9）。收据又高又窄（1:3）。医学影像以 2048x2048 或更大交付。移动设备截图是 1170x2532（0.46:1）。

2024 年前的三个选项，以及各自为何失败：

1. 缩放到固定正方形（224x224 或 336x336）。挤压会扭曲文字和面孔。降采样会毁掉图表标签和 OCR 内容。这是 LLaVA-1.5 之前的标准做法。
2. 裁剪到固定宽高比。你丢掉图像的大部分，而且挑选裁剪位置本身就是一个视觉问题。
3. 填充到最长边。修正了扭曲，但对纵向图像会在填充上浪费 50%+ 的 token。所有这些填充 token 都带来平方级的注意力成本。

2024–2025 的答案：让 Transformer 按图像原生分辨率吃 patch，并想办法把异构 batch 打包进一个序列而不浪费计算。

## 概念

### NaViT 与 patch-n'-pack

NaViT（Dehghani 等，2023）是证明这在规模上可行的论文。思路是机械性的：

1. 对 batch 中每张图，在选定 patch 尺寸（比如 14）下计算它的原生 patch 网格。
2. 把每张图的 patch 展平成它自己的可变长序列。
3. 把 batch 中所有图像的 patch 拼接成一条长序列。
4. 构建块对角注意力掩码，让图像 A 的 patch 只在图像 A 内关注。
5. 携带逐 patch 的位置信息（2D RoPE 或分数位置嵌入）。

三张图分别为 336x336（576 token）、224x224（256 token）、448x336（768 token）的 batch，变成一条 1600 token 的序列，配 1600x1600 的块对角掩码。没有填充。没有浪费计算。Transformer 处理任意宽高比。

NaViT 还引入了训练时的分数 patch 丢弃——在 batch 内随机丢弃 50% 的 patch——这既能正则化又能加速训练。SigLIP 2 继承了这一点。

### AnyRes（LLaVA-NeXT）

LLaVA-NeXT 的 AnyRes 是务实的替代方案。给定一张高分辨率图像和一个固定编码器（336 的 CLIP 或 SigLIP），把图像平铺：

1. 从预定义集合中选一个网格布局——(1x1)、(1x2)、(2x1)、(1x3)、(3x1)、(2x2) 等——最贴合图像宽高比的那个。
2. 把完整图像按网格平铺；每个瓦片变成 336x336 裁剪。
3. 同时生成缩略图：整张图缩放到 336x336，作为全局上下文 token。
4. 通过冻结的 336 编码器编码每个瓦片。拼接瓦片 token + 缩略图 token。

672x672 图像用 2x2 网格加缩略图：4 * 576 + 576 = 2880 个视觉 token。昂贵但有效——LLM 既看到局部细节也看到全局上下文。

当你的编码器冻结且只支持一种分辨率时，AnyRes 是首选路线。它会让大图的 token 数爆炸（1344x1344 图像用 4x4 网格是 9216 + 576 ≈ 9800 个 token，占满 8k LLM 上下文的大部分）。

### M-RoPE（Qwen2-VL）

Qwen2-VL 引入了多模态旋转位置嵌入。不用 NaViT 的分数位置，也不用 AnyRes 的平铺加缩略图，每个 patch 携带一个三维位置（时间、高度、宽度）。query/key 的旋转处理任意 H、W 和时间长度。

M-RoPE 无需重训即可原生支持动态分辨率。推理时你喂任意 HxW 图像，patch 嵌入器产出 H/14 x W/14 个 token，每个 token 得到它的 (t=0, r=row, c=col) 位置，RoPE 用正确的频率旋转注意力，完成。Qwen2.5-VL 和 Qwen3-VL 延续了这一路线。InternVL3 的 V2PE 是同样的思路，只是按模态使用可变编码。

与 AnyRes 不同，M-RoPE 在原生分辨率下是 $O(H \times W / P^2)$ 个 token——没有倍乘的瓦片开销。与 NaViT 不同，它仍然期望每次前向只有一张图。跨分辨率批处理仍需要在上面叠加 patch-n'-pack。

### NaFlex（SigLIP 2）

NaFlex 是 SigLIP 2 checkpoint 的原生灵活模式。单个模型在推理时服务多种序列长度（256、729、1024 个 token）。内部在训练时使用 NaViT 风格的 patch-n'-pack，并为每个 patch 使用绝对分数位置。它的关键优势是：一个 checkpoint，推理时按任务挑 token 预算。

语义任务（分类、检索）用 256 个 token。OCR 或图表理解用 1024 个 token。无需重训。

### 打包掩码

块对角掩码是大多数实现出错的地方。对一条长度为 `N_total` 的打包序列，覆盖图像 `i=0..B-1`，各图长度为 `n_i`，掩码 `M` 的形状为 `(N_total, N_total)`：当两个索引落在同一张图像的块内时为 1，否则为 0。你可以从累积长度列表构建它：

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

在 PyTorch 里用 `torch.block_diag` 或显式 gather 就是一行。FlashAttention 的可变长路径（`cu_seqlens`）完全跳过掩码，直接用累积长度张量在序列内部做注意力——对典型 batch 比稠密掩码快约 10 倍。

### Token 预算

按任务选策略：

- OCR / 文档：1024–4096 个 token。SigLIP 2 NaFlex 用 1024，或 AnyRes 3x3 + 缩略图。
- 图表和 UI：384–448 原生分辨率下 729–1024 个 token。Qwen2.5-VL 动态分辨率配最大像素上限。
- 自然照片：256–576 个 token 就够了。下游 LLM 看到的信息足够。把 token 花在内容密度高的地方。
- 视频：空间池化后每帧 64–128 个 token，2–8 FPS。第 12.17 课讲这个。

2026 年的生产规则：为每个任务设一个最大像素上限，在上限内按原生宽高比编码，打包 batch，跳过填充。Qwen2.5-VL 正好为此暴露了 `min_pixels` 和 `max_pixels` 两个旋钮。

```figure
mm-patch-n-pack
```

## 使用它

`code/main.py` 为带整数像素坐标的异构图像 batch 实现 patch-n'-pack。它：

- 接收 (H, W) 图像尺寸列表。
- 计算每张图在 patch 尺寸 14 下的 patch 序列长度。
- 把它们打包成总长度为 `sum(n_i)` 的单条序列。
- 构建块对角注意力掩码（稠密，以求清晰）。
- 比较打包成本 vs 方形缩放 vs AnyRes 平铺。
- 为混合 batch（收据、图表、截图、照片）打印 token 预算表。

运行它。算出来的数字就是 2026 年每个开放 VLM 都使用 patch-n'-pack 的原因。

## 交付它

本节课产出 `outputs/skill-resolution-budget-planner.md`。给定混合宽高比的工作负载（OCR、图表、照片、视频帧）和总 token 预算，它挑选正确策略（NaFlex、AnyRes、M-RoPE 或固定正方形）并给出每请求配置。为产品做 VLM 容量规划时用这个 skill——它能防止悄悄杀死延迟预算的 10 倍 token 爆炸。

## 练习

1. 一张收据是 600x1500（1:2.5）。patch 尺寸 14 下，原生分辨率有多少 token？方形缩放到 336 后呢？实践中哪个损失更多 OCR 准确率？

2. 为四张长度分别为 256、576、729、1024 的图像构建块对角掩码。验证注意力矩阵是 2585x2585，且恰好有 `256^2 + 576^2 + 729^2 + 1024^2` 个非零元素。

3. 对于 1792x896 图像、patch 14，比较：(a) 方形缩放到 336 再编码，(b) AnyRes 2x1 + 缩略图，(c) 原生 M-RoPE。哪个用的 token 最少？哪个保留细节最多？

4. 实现分数 patch 丢弃：给定打包序列，均匀随机丢弃 50% 的 token，并相应更新块对角掩码。测量掩码稀疏度的变化。

5. 阅读 Qwen2-VL 论文（arXiv:2409.12191）第 3.2 节。用两句话描述 `min_pixels` 和 `max_pixels` 控制什么，以及为什么两个边界都重要。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Patch-n'-pack | 「NaViT 式打包」 | 把不同图像的可变长 patch 序列拼接进一个 batch 维度 |
| Block-diagonal mask | 「打包掩码」 | 把每张图的 patch 限制为只关注自身、不关注打包中邻居的注意力掩码 |
| AnyRes | 「LLaVA-NeXT 平铺」 | 把高分辨率图像切成固定大小瓦片网格加一张全局缩略图；用固定编码器编码每个瓦片 |
| NaFlex | 「SigLIP 2 原生灵活」 | 单个 SigLIP 2 checkpoint 在推理时无需重训即可服务 256/729/1024 token 预算 |
| M-RoPE | 「多模态 RoPE」 | 三维旋转位置编码（时间、行、列），无需位置表即可处理任意 H、W、T |
| cu_seqlens | 「FlashAttention 打包」 | FlashAttention varlen 路径用累积长度张量代替稠密块对角掩码 |
| min_pixels / max_pixels | 「分辨率边界」 | Qwen2.5-VL 的逐请求旋钮，限制极小或极大输入的 token 数 |
| Visual token budget | 「每图多少 token」 | 每张图发出的 patch token 大致数量；决定 LLM 的 prompt 预算和注意力成本 |

## 延伸阅读

- [Dehghani 等——Patch n' Pack: NaViT（arXiv:2307.06304）](https://arxiv.org/abs/2307.06304)
- [Wang 等——Qwen2-VL（arXiv:2409.12191）](https://arxiv.org/abs/2409.12191)
- [Laurençon 等——What matters when building vision-language models?（Idefics2，arXiv:2405.02246）](https://arxiv.org/abs/2405.02246)
- [Tschannen 等——SigLIP 2（arXiv:2502.14786）](https://arxiv.org/abs/2502.14786)
- [Qwen Team——Qwen2.5-VL Technical Report（arXiv:2502.13923）](https://arxiv.org/abs/2502.13923)
