# 百万 token 上下文下的长视频理解

> 1 小时 24 FPS 的 4K 视频，经过切块和嵌入，会产生约 6000 万个 token。一集 2 小时播客转成文字是 3 万个 token。一部完整的蓝光剧情片，即便用激进池化压缩，也是几十万个 token。Google 的 Gemini 1.5（2024 年 3 月）以 1000 万 token 上下文开启了这个时代，能在长达一小时的视频里做可靠的大海捞针召回。LWM（Liu 等，2024 年 2 月）展示了 ring attention 的扩展路径。LongVILA 和 Video-XL 进一步扩展了摄取规模。VideoAgent 用 Agent 式检索取代原始上下文。每种方案在算力、召回率和工程复杂度上做了不同权衡。本课将它们并排解读。

**Type:** Build
**Languages:** Python (stdlib, needle-in-haystack simulator + agentic-retrieval router)
**Prerequisites:** Phase 12 · 17 (video temporal tokens)
**Time:** ~180 minutes

## 学习目标

- 计算不同 FPS 和池化设置下长视频的视觉 token 总量。
- 解释三条扩展路径：蛮力上下文（Gemini 1.5）、ring attention（LWM）、token 压缩（LongVILA / Video-XL）。
- 在准确率和延迟上比较原始上下文视频 VLM 与 Agent 式检索视频 VLM（VideoAgent）。
- 为一段 30 分钟视频设计大海捞针测试，并测量指定分钟处的召回率。

## 问题

Qwen2.5-VL 尺寸的 patch 在 384 原生分辨率下，单帧约 729 个 token。按 3x3 池化，每帧 81 个 token。30 分钟片段按 1 FPS 是 1800 帧 = 145,800 个 token。2025 年的开放 VLM 勉强能处理，余量很紧。按 2 FPS 则是 291,600 个 token——只有最大的上下文才装得下。

一部 2 小时电影按 1 FPS 是 583k token。超出 2026 年大多数开放模型的能力；需要 Gemini 2.5 Pro，或者更激进的池化。

由此出现了三条扩展路径。

## 概念

### 路径 1：蛮力上下文（Gemini 1.5、Claude Opus）

用硬件硬扛。把上下文扩展到数百万 token，一次前向传播处理全部内容。

Gemini 1.5 Pro 发布时是 1M token；Gemini 1.5 Ultra 到 10M；2026 年的 Gemini 2.5 Pro 能可靠处理数小时的视频。论文（arXiv:2403.05530）记录了在约 950 万 token 以内大海捞针召回率达到 99.7%。

工程实现：定制注意力实现，带内存层级（局部 + 全局 + 稀疏），再加 MoE 专家路由以提升长上下文效率。论文没有完整公开细节。非开源。

### 路径 2：Ring attention（LWM、LongVILA）

Ring attention 把长序列分布到多台设备上，形成一个「环」，每台设备持有一个分块。全序列上的注意力通过以下方式完成：每台设备按环状模式把自己的分块发给下一台，计算部分注意力，再聚合。

LWM（Liu 等，2024）用这种方式训练了一个 1M token 上下文的模型。训练算力随上下文线性增长，而非平方增长——注意力上的平方开销被分摊到环上的多台设备。

LongVILA（arXiv:2408.10188）把这一模式适配到 VLM。1400 帧视频、每帧 192 token = 268k 上下文，用 8 路并行下的 ring attention 训练。

### 路径 3：Token 压缩（Video-XL、LongVA）

比蛮力上下文便宜：在 LLM 看到序列之前就激进压缩。

Video-XL（arXiv:2409.14485）使用视觉摘要 token：每 N 帧组成的片段产生一个「摘要」token，它对该片段的 N 帧做注意力。推理时，LLM 每个片段只看一个摘要 token，上下文大幅缩小。

LongVA 用「长上下文迁移」技术把 LLM 上下文从 200k 扩展到 2M。先在长上下文文本上训练，再通过共享表示迁移到长上下文视频。

Token 压缩用特定时间戳上的召回率换取可扩展性。模型大致知道发生了什么，但有时会漏掉确切的帧。

### 路径 4：Agent 式检索（VideoAgent）

不要把整段视频喂给 LLM。相反，把视频当成数据库，用 LLM 去查询它。

VideoAgent（arXiv:2403.10517）：

1. LLM 读问题。
2. LLM 向检索工具请求相关片段（「给我看有猫的片段」）。
3. 工具返回匹配片段的起止时间。
4. LLM 通过 VLM 读取这些片段。
5. LLM 组织答案，或发起追问。

这是把 LLM-as-agent 模式应用到长视频上。推理更便宜（只编码相关片段），工程更难（检索质量成为瓶颈）。

### 大海捞针基准

标准的长上下文测试：在视频的随机位置插入一个独特的视觉或文本标记，然后提出一个必须回忆该标记才能回答的问题。

指标：在不同视频长度和标记位置下的 Recall@k。

Gemini 2.5 Pro 在最长 90 分钟的视频上召回率超过 99%。开放的 72B 模型（Qwen2.5-VL-72B、InternVL3-78B）在 30 分钟时约 85-90%，超过 60 分钟后明显退化。

VideoAgent 在 2 小时以上的内容上可以持平或超过原始上下文模型，因为只要检索工具够好，就能命中那根针。

### 该选哪条路径

15 分钟片段、要求前沿准确率：开放 72B + 原生上下文通常够用。选 Qwen2.5-VL-72B。

30 分钟到 1 小时的内容：开放方案用 LongVILA 或 Video-XL；闭源用 Gemini 2.5 Pro。质量要求高时走闭源。

2 小时以上的内容：VideoAgent 或类似的检索模式。或者先摘要成更小的块，再喂分层摘要。

### 2026 年的生产模式

实践中，生产级长视频 pipeline 是混合式的：

1. 对整段视频跑动态 FPS 采样 + 激进池化（得到 100k token 的全局表示）。
2. 交给 72B VLM 生成全局摘要。
3. 如果用户提出细节问题，以摘要为索引运行 Agent 式检索。

这样把蛮力上下文用于全局理解，把检索用于局部细节。

```figure
mm-video-token-budget
```

## 使用它

`code/main.py`：

- 计算从 1 分钟到 3 小时的视频在不同 FPS + 池化设置下的 token 预算。
- 模拟一次大海捞针运行：在随机时间戳注入标记，提问，计算召回。
- 包含一个 Agent 式检索路由器模拟器，挑选特定片段喂给下游 VLM。

跑一下预算表，感受规模差距。

## 交付它

本课产出 `outputs/skill-long-video-strategy-planner.md`。给定视频时长和查询复杂度，它在蛮力上下文、压缩和 Agent 式检索之间做选择，并计算延迟与质量预期。

## 练习

1. 一段 45 分钟的讲座，1 FPS，每帧 81 个 token。总 token 数是多少？能装进哪些模型的上下文？

2. 设计一个大海捞针测试：在第几分钟注入标记，确切的查询格式是什么？

3. 在 1 小时视频上比较蛮力上下文的 Qwen2.5-VL-72B（80k 上下文）与 VideoAgent（Claude 3.5 + 检索）。哪个召回率更高？哪个延迟更低？

4. Ring attention 的显存开销随序列长度线性增长，也随设备数量线性增长。解释原因，并说明如果去掉环旋转阶段会出什么问题。

5. 读 Gemini 1.5 第 5 节关于大海捞针的内容。论文在 1M 与 10M token 边界上关于召回率发现了什么？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 蛮力上下文 | 「就是堆更多 token」 | 把 LLM 上下文扩展到数百万 token；一次处理全部内容 |
| Ring attention | 「LWM 式并行」 | 每台设备持有一个分块并轮转的分布式注意力模式 |
| Token 压缩 | 「摘要 token」 | 在送入 LLM 前用学习式压缩器减少每片段 token 数 |
| 大海捞针 | 「NIH 测试」 | 在随机位置插入独特标记，测试时要求模型回忆它 |
| Agent 式检索 | 「LLM 作为查询规划器」 | LLM 向检索工具请求相关片段，用 VLM 读取，再组织答案 |
| VideoAgent | 「视频检索模式」 | 规范的 Agent 式检索设计：问题 -> 工具 -> 片段 -> 答案 |

## 延伸阅读

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu 等 — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue 等 — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu 等 — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang 等 — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)
