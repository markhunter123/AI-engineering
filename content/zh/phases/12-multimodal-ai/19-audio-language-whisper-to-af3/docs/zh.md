# 音频-语言模型：从 Whisper 到 Audio Flamingo 3 的演进弧线

> Whisper（Radford 等，2022 年 12 月）解决了语音识别——68 万小时弱监督多语种语音、一个简单的编码器-解码器 Transformer、一个让后续每一次 ASR 发布都要引用它的基准。但识别不等于推理。问「这段录音里有哪些乐器」「说话人表达了什么情绪」「第 3 分钟发生了什么」，需要的是音频理解，而不是转写。Qwen-Audio、SALMONN、LTU 和 NVIDIA 的 Audio Flamingo 3（AF3，2025 年 7 月）逐步搭建起了这套栈：保留 Whisper 级别的编码器，外挂 Q-former，在音频-文本指令数据上训练，再加上思维链推理。本课走一遍这条演进弧线。

**Type:** Build
**Languages:** Python (stdlib, log-Mel spectrogram + audio Q-former skeleton)
**Prerequisites:** Phase 6 (Speech and Audio), Phase 12 · 03 (Q-Former)
**Time:** ~180 minutes

## 学习目标

- 从波形计算 log-Mel 频谱图：加窗、FFT、滤波器组、对数变换。
- 比较编码器选项：Whisper 编码器、BEATs、AF-Whisper 混合。各自在什么情况下胜出。
- 构建音频 Q-former：N 个可学习查询对频谱图 patch 做交叉注意力。
- 解释级联式（Whisper 再接 LLM）与端到端音频-LLM 训练：为什么端到端在推理任务上扩展性更好。

## 问题

语音识别已经被 Whisper 解决。音频的 OCR 已经商品化。但「商品化」止步于转写。如果模型无法对听到的内容做推理——时间、说话人、情绪、音乐结构、环境音——仅靠转写就无法驱动产品功能。

三条显而易见的路线：

1. 级联：Whisper 转写，LLM 对文本做推理。适用于纯语音场景。在音乐、环境音、多说话人重叠、情绪上失效。

2. 端到端音频-LLM：音频编码器把音频 token 直接送入 LLM，跳过转写。保留声学信息（情绪、说话人、环境）。需要新的训练数据。

3. 混合：音频编码器 + 文本解码器，既能转写又能推理。Qwen-Audio 和 Audio Flamingo 选择了这条路。

## 概念

### Log-Mel 频谱图：输入特征

每个音频编码器都从同一个特征开始：log-Mel 频谱图。

1. 重采样到 16 kHz。
2. 25ms 窗口、10ms 步长的短时傅里叶变换。
3. 取 FFT 结果的幅度。
4. 应用 Mel 滤波器组（通常 80 个滤波器，在 0-8000 Hz 上对数间隔），变换到感知频率。
5. 对数压缩 $(\log(1 + x))$ 以扩大动态范围。

结果：形状为 (T, 80) 的二维数组，T 是时间帧数。对于 100 Hz 帧率的 30 秒音频：(3000, 80)。

### Whisper 的编码器

Whisper 的编码器是一个 12 层 ViT 风格 Transformer，把 log-Mel 频谱图当作时间帧序列处理。输出：每个时间帧一个隐藏状态向量。

用于 ASR 时，Whisper 的解码器是一个交叉注意力 Transformer，以编码器输出为条件生成文本 token。标准的编码器-解码器结构。

对于 ALM（音频-LLM），你需要把编码器输出作为另一个 LLM 的输入。模式是：Whisper 编码器冻结，Q-former 可训练，LLM 冻结或微调。

### BEATs 与音频专用编码器

Whisper 是在以语音为主的数据上训练的。它在音乐和环境音上较弱。

BEATs（Chen 等，2022）是在 AudioSet 上训练的自监督 Transformer。在同等参数量下，它对音乐和环境音的捕捉优于 Whisper。

AF-Whisper（Audio Flamingo 3 的混合方案）：把 Whisper 和 BEATs 的特征拼接作为音频输入。Whisper 承载语言信号，BEATs 承载声学信号。

### 音频 Q-former

模式与 BLIP-2 的视觉 Q-former 相同。固定数量的可学习查询（通常是 32 或 64 个）对音频编码器的输出帧做交叉注意力。这些查询成为 LLM 消费的音频 token。

训练对齐阶段：只训练 Q-former，在音频-文本对上使用对比损失 + 描述损失（AudioCaps、Clotho）。指令阶段：端到端，解冻 LLM，在指令数据上训练。

### 演进弧线——SALMONN、Qwen-Audio、AF3

SALMONN（Tang 等，2023）：Whisper + BEATs + Q-former + LLaMA。第一个具备严肃推理能力的开源音频-LLM。在 MMAU 上的综合分约为 0.55。

Qwen-Audio（Chu 等，2023）：架构类似，在更丰富的数据集上训练，针对多轮对话调优。MMAU 约 0.60。

LTU——Listen, Think, Understand（Gong 等，2023）：显式的推理数据，专注于音频片段上的思维链。规模更小但更聚焦。

Audio Flamingo 3（Goel 等，2025 年 7 月）：当前开源 SOTA。8B LLM 骨干（Qwen2 7B），Whisper-large 编码器拼接 BEATs，64 查询的 Q-former，在 100 万+ 音频-文本指令对上训练。MMAU 0.72，在某些子任务上匹敌闭源前沿模型。

AF3 还为音频引入了按需思维链：模型可以选择在最终答案之前输出思考 token（「让我先识别一下乐器：……」）。启用思考后，复杂推理任务的准确率提升 3-5 个点。

### 级联 vs 端到端

级联式流水线：

1. Whisper 把音频转写为文本。
2. LLM 对文本做推理。

对于「总结这段播客」效果完美。但在以下场景失效：
- 「这首歌的情绪是什么？」——情绪在声音里，不在文字里。
- 「说话的是 Alice 还是 Bob？」——需要说话人识别。
- 「爆炸发生在第几秒？」——时间定位在文本中丢失了。
- 「这是真实音频还是生成音频？」——深度伪造检测需要声学特征。

端到端保留声学信号。Qwen-Audio 和 AF3 原生处理音乐、环境和情绪。

### 2026 年的生产配方

对于一个新音频理解产品：

- 级联，如果：目标是转写，没有音乐，不需要情绪推断。
- AF3 / Qwen-Audio 系列，如果：涉及音乐、情绪、多说话人或复杂音频推理。

级联更便宜也更简单。端到端能力更强。

### MMAU——音频推理基准

MMAU（Massive Multimodal Audio Understanding）是 2024-2025 年的音频推理基准：

- 10,000 个音频-文本问答对，覆盖语音、音乐、环境音。
- 涵盖分类、时间推理、因果推理、开放式问答。
- 测试的正是级联流水线系统性缺失的能力。

开源 SOTA（AF3）为 0.72；闭源前沿约 0.78（Gemini 2.5 Pro、Claude Opus 4.7）。这一差距小于 VideoMME 的开源-闭源差距，说明音频-LLM 正在成熟。

```figure
audio-text-ctc
```

## 使用它

`code/main.py`：

- 用 stdlib 实现 log-Mel 频谱图计算：加窗、朴素 DFT、Mel 滤波器组。
- 音频 Q-former 骨架：给定编码器输出帧，计算 Q、K、V、注意力，并输出 N 个 token。
- 在玩具任务上做级联 vs 端到端的对比。

## 交付它

本课产出 `outputs/skill-audio-llm-pipeline-picker.md`。给定一个音频任务（转写、音乐打标签、情绪推断、多说话人分离、环境分类），它选择级联、端到端 AF3 或混合方案。

## 练习

1. 计算 16kHz、25ms 窗口、10ms 步长、80 个 Mel 频带的 30 秒音频的 log-Mel 频谱图维度。在 48kHz 下这个维度如何变化？

2. 为什么 Whisper 在音乐上表现不佳？BEATs 捕捉到了哪些 Whisper 没有的音频特征？

3. 64 个查询 vs 32 个查询的音频 Q-former：在什么任务复杂度下 64 个才划算？32 个在什么情况下节省算力？

4. 阅读 AF3 关于按需思考的第 4 节。提出三个思维链帮助最大的音频任务。

5. 使用 AF3 的输出实现一个最小的说话人分离流水线。你如何标记说话人切换？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Log-Mel 频谱图 | 「Mel 特征」 | 经过 Mel 滤波器组后的对数幅度二维（时间、频率）数组 |
| 音频 Q-former | 「音频 Perceiver」 | 从音频编码器输出到固定长度查询的交叉注意力瓶颈，查询送入 LLM |
| 级联 | 「ASR 再接 LLM」 | Whisper 转写、文本 LLM 推理的流水线；会丢失声学信息 |
| 端到端 | 「音频-LLM」 | 音频特征通过 Q-former 直接进入 LLM；保留声学信号 |
| BEATs | 「AudioSet 音频编码器」 | 在 AudioSet 上训练的自监督 Transformer；在音乐 + 环境音上表现强 |
| MMAU | 「音频推理基准」 | 1 万个问答对，覆盖语音、音乐、环境；2024 年的评测标准 |
| 按需思考 | 「音频 CoT」 | 模型可在最终答案前选择性输出推理 token，准确率提升 3-5 点 |

## 延伸阅读

- [Radford et al. — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu et al. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel et al. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang et al. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong et al. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
