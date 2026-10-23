# 视频-语言模型：时间 token 与定位

> 视频不是一叠照片。一个 5 秒片段包含因果顺序、动作动词和事件时序，这些是图像模型无法表示的。Video-LLaMA（Zhang 等，2023 年 6 月）交付了第一个具备音视频定位能力的开放视频 LLM。VideoChat 和 Video-LLaVA 扩大了这一范式。到 2025 年，Qwen2.5-VL 的 TMRoPE 追平了前沿专有模型。各系统解决时间 token 的方式各不相同——每个片段一个 Q-former、每帧池化拼接、每个 token 一个 TMRoPE。本课解读这些范式，构建一个均匀 vs 动态的帧采样器，并在时间定位任务上做评估。

**Type:** Build
**Languages:** Python (stdlib, frame sampler + temporal-grounding evaluator)
**Prerequisites:** Phase 12 · 08 (LLaVA-OneVision)
**Time:** ~180 minutes

## 学习目标

- 解释为什么时间位置编码会独立于视觉编码器改变视频 VLM 的性能。
- 在每秒 token 数与定位准确率上，比较均匀采样、动态 FPS 和事件驱动采样。
- 描述每片段 Q-former（Video-LLaMA）、每帧池化（Video-LLaVA）和每 token M-RoPE（Qwen2.5-VL）的设计。
- 说出四个视频基准：VideoMME、TempCompass、EgoSchema、Video-MMMU。

## 问题

1 分钟、30 FPS 的视频有 1800 帧。按每帧 196 个视觉 token 计算（224 分辨率的 ViT-B），就是 352k token——超过 2024 年任何 LLM 的上下文。

存在三种压缩策略：

1. 抽帧（根据内容取 1-8 FPS）。
2. 对每帧的 patch token 做激进池化（3x3 或 4x4 双线性池化）。
3. 通过 Q-former 压缩：输入 16 帧片段，输出 64 个 token。

每种权衡都不同。抽帧丢失时间细节。池化丢失空间细节。Q-former 两者都丢一点，但节省 token。

时间位置编码是另一个维度：模型如何知道第 5 帧在第 6 帧之前？可选方案包括简单的一维时间 RoPE（Video-LLaMA）、学习式时间嵌入（Video-LLaVA）和 TMRoPE（Qwen2.5-VL，完整三维）。

## 概念

### Video-LLaMA：每片段 Q-former + 音频分支

Video-LLaMA（2023）是第一个开放视频 LLM。架构：

- 16 帧片段，2 FPS（即 8 秒）。
- 逐帧 ViT 特征 -> Video Q-former 对全部 16 帧做交叉注意力 -> 32 个学习查询 -> LLM。
- 并行的音频分支：波形 -> ImageBind 音频编码器 -> Audio Q-former -> 32 个查询 -> LLM。

优势：音视频联合推理。劣势：片段长度固定，无法做任意时间定位。

### VideoChat 与 Video-LLaVA

VideoChat 保留了 Video-LLaMA 的思路，但去掉了音频并做了简化。Video-LLaVA（Lin 等，2023）在图像和视频帧上训练同一个视觉编码器（「先对齐后投影」），得到统一表示。两者都是冻结 CLIP 编码器 + MLP + LLM。

两者都不处理长视频。都是 8-16 帧的系统。

### Qwen2.5-VL 与 TMRoPE

Qwen2.5-VL 引入了 TMRoPE——Temporal-Modality Rotary Position Embedding（时间-模态旋转位置嵌入）。每个 patch token 携带一个 (t, h, w) 位置，其中 t 是真实时间戳（而非帧索引）。

与简单时间嵌入的关键差异：

- 绝对时间，而非索引。模型看到的是「在 4.2 秒处」，而不是「在第 15 帧」。
- 逐 token 旋转，而非逐片段。每个视觉 token 按自己的时间戳独立旋转。
- 兼容动态 FPS。如果这里按 2 FPS 采样、那里按 4 FPS 采样，TMRoPE 原生处理不均匀的间隔。

TMRoPE 让「猫在第几秒跳起来？」这类查询成为可能。模型可以输出「在 4.2 秒」。Video-LLaMA 只能说「在片段前段」。

### 帧采样策略

均匀采样：在时长上等间隔取 N 帧。简单，但会错过运动峰值。

动态 FPS：根据运动强度自适应采样。光流或帧差法挑出高运动片段做更密集的采样。Qwen2.5-VL 按此训练。

事件驱动：跑一个轻量检测器，在动作发生处多采样。VideoAgent 使用。

关键帧 + 上下文：在镜头边界采样，再加几帧相邻帧。用于影视内容。

### 逐帧池化

在 1 FPS、每帧 576 token 下，5 分钟片段是 172,800 token。Qwen2.5-VL-72B 的 128k 上下文可以处理，但很昂贵。

3x3 双线性池化把每帧降到 64 token -> 5 分钟为 19,200 token。这是大多数任务的最佳区间。

对于空间细节不那么重要的 Agent 工作流，可以更激进地池化（6x6 -> 每帧 16 token）。

### 四个视频基准

- VideoMME：综合视频理解，短、中、长视频。
- TempCompass：细粒度时间推理，「之前」/「之后」类问题。
- EgoSchema：长时程第一人称视频。
- Video-MMMU：多模态、多学科的视频问答。

一次完整的视频 VLM 评估应当覆盖全部四个。它们考察不同的维度——TempCompass 关注顺序，EgoSchema 关注 3 分钟以上的推理，VideoMME 横跨不同时长。

### 定位输出格式

时间定位的输出格式：

- 自由文本：「猫大约在第 4 秒处跳起。」容易解析但不精确。
- 结构化 JSON：`{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL 按此训练。
- 基于 token：把特殊的 `<time>4.1</time>` token 交错在答案中。这是 Qwen2.5-VL 的内部格式。

基于 token 的方式对下游使用最准确。Qwen2.5-VL 的 JSON 输出格式可直接解析。

### 2026 年的最佳实践

2026 年的视频 VLM：

- 编码器：SigLIP 2，搭配 M-RoPE 或 TMRoPE（Qwen2.5-VL）。
- 帧采样：动态 FPS（按运动情况取 1-4），并设最大帧数上限。
- 逐帧池化：3x3 双线性。
- 输出：带时间 + 事件字段的结构化 JSON。
- 基准：通用场景用 VideoMME + TempCompass；长时程用 EgoSchema。

```figure
video-temporal-patches
```

## 使用它

`code/main.py` 包含：

- 均匀采样和动态 FPS 帧采样器。
- 一个玩具级时间定位评估器：给定时刻 T 的「真值」事件和模型输出，在容差内打准确率分。
- Video-LLaMA（16 帧，Q-former）、Video-LLaVA（8 帧，MLP）、Qwen2.5-VL（动态 FPS + TMRoPE）三者的对比。

## 交付它

本课产出 `outputs/skill-video-vlm-frame-planner.md`。给定一个视频任务（监控、动作识别、时间定位、摘要），它选择帧采样器、池化系数、输出格式和预期的准确率档位。

## 练习

1. 对于一个 3 分钟的烹饪演示，选择均匀采样还是动态 FPS？用 token 数量论证。

2. TMRoPE 具体增加了什么，是简单的时间嵌入表做不到的？

3. 写一个 VLM 能学会输出的时间定位 JSON schema。包含错误情况。

4. 读 Video-LLaVA 第 3 节「Alignment Before Projection」。为什么这比分别训练图像编码器和视频编码器更好？

5. 根据 VideoMME 排行榜，截至 2026 年，最强开放模型与最强专有模型之间差距多大？其中有多少可归因于时间编码，多少可归因于基础 LLM 的规模？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 时间定位 | 「时间局部化的答案」 | VLM 输出事件发生的具体时间戳区间 |
| TMRoPE | 「Time-Multimodal RoPE」 | 带绝对时间戳的三维旋转位置编码，Qwen2.5-VL 使用 |
| 动态 FPS | 「运动感知采样」 | 高运动片段多采帧，静止片段少采帧 |
| 帧池化 | 「逐帧空间压缩」 | 在送入 LLM 前用双线性插值减少每帧的 patch 数 |
| Video Q-former | 「片段压缩器」 | 把 N 帧映射到 K 个学习查询的交叉注意力瓶颈 |
| VideoMME | 「视频基准」 | 覆盖短/中/长视频的综合基准，2500+ 样本 |

## 延伸阅读

- [Zhang 等 — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li 等 — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin 等 — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin 等 — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
