# Omni 模型：Qwen2.5-Omni 与 Thinker-Talker 分离架构

> GPT-4o 在 2024 年 5 月的产品演示之所以具有颠覆性，不是因为底层模型，而是因为产品形态——一种语音接口：你说话，模型看到摄像头所见，并在 250ms 内回话。开源生态在 2024 年余下时间和 2025 年一路追赶这个产品形态。Qwen2.5-Omni（2025 年 3 月）是参考性的开放设计：一个 Thinker（大型文本生成 Transformer）加一个 Talker（并行的语音生成 Transformer），通过流式语音 token 连接。Mini-Omni 简化了它，Moshi 追平了它的延迟，GLM-4-Voice 把它扩展到中文。本课解读 Thinker-Talker 架构，以及让流式实时对话成立的那份延迟预算。

**Type:** Build
**Languages:** Python (stdlib, streaming pipeline latency simulator + VAD loop)
**Prerequisites:** Phase 12 · 19 (audio-LLMs), Phase 12 · 16 (any-to-any)
**Time:** ~180 minutes

## 学习目标

- 把推理 pipeline 拆成 Thinker（文本推理）和 Talker（语音合成），并解释并行流式为何可行。
- 逐组件计算一次对话交互的首个音频字节时间（TTFAB）预算。
- 描述 TMRoPE 在 Thinker 内部跨越视觉、音频和文本的时间对齐位置编码。
- 说出三种实时对话模式：半双工、轮流发言、全双工。

## 问题

一个实时语音助手必须在很短时间内完成很多事：

1. 听用户。实时语音分词、语音活动检测（VAD）判断用户何时说完。
2. 可选地看。以 2-4 FPS 输入摄像头画面，与音频一起流式送入 Thinker。
3. 思考。基于对话历史组织回答。
4. 说话。合成音频 token，解码为波形，流式送到用户的扬声器。

每一步都会增加延迟。对话感要求总往返时间 < 500ms——低于这个值，用户就不再察觉卡顿。GPT-4o 声称约 250ms。Moshi 约 160ms。Qwen2.5-Omni 约 350-500ms。

每个组件都必须支持流式。任何环节都不能「先全部组批、再解码」。

## 概念

### Thinker 与 Talker

Qwen2.5-Omni 的分解：

- Thinker：一个 7B-80B 的文本生成 Transformer。消费交错的文本 + 图像 + 音频 token。输出表示「该说什么」的文本 token。
- Talker：一个更小的语音生成 Transformer（200M-1B）。消费 Thinker 输出的文本 token 以及最近的语音上下文 token。输出离散语音 token（残差 VQ 索引）。
- 语音解码器：流式波形解码器（SNAC、MoVQGAN 系列），实时把语音 token 转成音频采样。

这种分离很关键。Thinker 必须够大才能推理得好。Talker 可以很小，因为它的任务是局部的——把文本转成语音 token。更大的 Talker 不会更富表现力，只会更慢。

两者并行运行：

1. Thinker 输出文本 token $t_i$。
2. Talker 消费 $t_i$（通过流式），并输出语音 token $s_i, s_{i+1}, \ldots, s_{i+k}$。
3. 语音解码器随到随消费语音 token，输出音频采样。
4. 当 Thinker 推进到文本 token $t_{i+3}$ 时，Talker 已经为 $t_0, \ldots, t_{i+2}$ 流出了音频。

### TMRoPE——时间对齐的多模态位置

Thinker 需要整合图像帧（比如以 4 FPS 到达）、音频帧（以每秒 50 帧到达）和对话历史中的文本。朴素的序列顺序（先所有图像、再所有音频、最后文本）会丢失时间对齐。

TMRoPE 给每个 token 分配绝对时间戳。视觉 token 在 t=2.3s。音频 token 在 t=2.32s。用户说的「停」文本 token 在 t=2.35s。RoPE 按时间戳旋转注意力；模型将它们视为在时间上同时发生。

这就是「他一边挥手一边说你好」能够成立的基础设施——模型在同一概念时刻看到视频帧和音频。

### 流式语音合成

语音 token 必须流式输出。Mini-Omni（Xie & Wu，2024）提出了「语言模型可以在流式思考的同时听和说」：Thinker 的输出 token 和 Talker 的输出 token 交错在同一条序列里。Thinker 一旦确定下一个文本 token，Talker 立即启动。没有 batch 边界。

Moshi（Défossez 等，2024 年 10 月）是最快的开源实现。单张 A100 上 TTFAB 为 160ms。架构：单个 7B Transformer 在交替位置上输出文本和语音 token，并用「内心独白」把思考流和说话流分开。这实际上是通过精心训练把 Thinker + Talker 融为一个模型。

### VAD 与轮流发言

语音活动检测运行在输入端。两种模式：

- 半双工：用户说，模型听。模型说，用户听。通过 VAD 静音检测（约 200ms）明确交接。
- 全双工：双方可以同时说话。模型可以应和（「嗯哼」）或打断。难得多。Moshi 支持。

Qwen2.5-Omni 默认支持半双工，通过静音阈值实现轮流发言。全双工需要应用层处理。

### Qwen3-Omni（2025 年 11 月）

继任者。Qwen3-80B Thinker，更大的 Talker，改进的 TMRoPE-v2。延迟接近 GPT-4o 的 250ms。开放权重。在 OmniBench 上的基准与 Gemini 2.0 Live 相当。

### 生产延迟预算

对于典型的流式交互：

- 麦克风 -> 音频 token：40-80ms。
- 预填充（prompt + 历史）：7B 上 100-200ms，70B 上要长得多。
- Thinker 首个文本 token：40ms。
- Talker 处理首个文本 token：20ms。
- 首批语音 token 确定：40ms。
- 残差 VQ 解码：30ms。
- 语音波形解码：50-80ms。

TTFAB 总计：7B 上 320-510ms，70B 上 600-900ms。前沿质量通常意味着 70B+；因此存在前沿延迟差距。

### Token 速率算术

16kHz 语音配 50 Hz 基础语音 token，每秒输出需要 50 个语音 token。Talker 必须达到 ≥50 tok/s 才能跟上。在 H100 上典型的 LLM 吞吐为 30-80 tok/s，小型（200-300M）Talker 足够快；7B 的 Talker 会跟不上。

这就是为什么存在专门的小型 Talker 模型，而不是「直接用主模型」。

```figure
l5-thinker-talker
```

## 使用它

`code/main.py`：

- 用模拟的 token 输出速率仿真 Thinker-Talker pipeline。
- 为可配置的模型规模和麦克风采样率计算 TTFAB。
- 演示带 VAD 静音阈值的半双工轮流发言。

## 交付它

本课产出 `outputs/skill-omni-streaming-budget.md`。给定实时语音产品的目标 TTFAB 和功能集（视觉输入、双语、全双工），选择 Qwen2.5-Omni、Qwen3-Omni、Moshi 或 Mini-Omni，并确定 Thinker/Talker 的规模。

## 练习

1. 你的目标 TTFAB 是 300ms。在 7B Thinker 和 300M Talker 上，写出每个组件的延迟。

2. Qwen2.5-Omni 使用 TMRoPE。描述用户从 t=1s 开始说话、摄像头在 t=1.2s 捕捉到一个手势时，模型看到的内容。

3. 全双工支持要求模型在听的同时输出音频。提出一种能教会这一点的训练数据格式。

4. 阅读 Moshi 论文第 4 节。描述「内心独白」的分离方式，以及它为什么能避开 Thinker-Talker 拆分。

5. 计算吞吐预算：Talker 必须以多快的速度输出 token，才能跟上 50 基础层 token/秒的 16kHz 语音？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Thinker | 「推理大脑」 | 大型文本生成 Transformer，产出该说什么 |
| Talker | 「语音生成嘴」 | 小型 Transformer，从 Thinker 的文本产出离散语音 token |
| TTFAB | 「延迟预算」 | 首个音频字节时间：从用户说话结束到第一个音频采样输出 |
| TMRoPE | 「时间对齐 RoPE」 | 跨视觉、音频、文本使用绝对时间戳的位置编码 |
| 半双工 | 「轮流发言」 | 用户和模型交替；VAD 静音检测判断用户说完 |
| 全双工 | 「同时进行」 | 模型可以同时说和听；具备应和能力 |
| 内心独白 | 「Moshi 分离」 | 单模型设计，思考流和说话流交错 |

## 延伸阅读

- [Xu et al. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez et al. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng et al. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)
