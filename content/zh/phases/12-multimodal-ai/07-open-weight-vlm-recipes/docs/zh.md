# 开放权重 VLM 配方：真正重要的是什么

> 2024-2026 年的开放权重 VLM 文献是一片消融表格的森林。Apple 的 MM1 测试了图像编码器、连接器和数据混合的 13 种组合。Allen AI 的 Molmo 证明，详细的人类标注描述（caption）胜过 GPT-4V 蒸馏。Cambrian-1 跑了 20 多组编码器对比。Idefics2 把设计空间形式化为五个轴。Prismatic VLMs 在受控基准上比较了 27 种训练配方。在所有这些噪声中，有一小组结论跨论文稳定成立：图像编码器比连接器架构更重要，数据混合比两者都更重要，而详细的人类标注描述胜过蒸馏的合成数据。这节课替你把这些表格读完。

**Type:** Learn + lab
**Languages:** Python (stdlib, ablation table parser + recipe picker)
**Prerequisites:** Phase 12 · 05 (LLaVA baseline)
**Time:** ~180 minutes

## 学习目标

- 说出 VLM 的五轴设计空间：图像编码器、连接器、LLM、数据混合、分辨率调度。
- 读懂 MM1 / Idefics2 / Cambrian-1 的消融表，并预测哪个旋钮会推动某个给定基准的分数。
- 在给定算力预算和任务组合时，为新 VLM 挑选一套配方（编码器、连接器、数据、分辨率）。
- 解释为什么在相同 token 数下，详细的人类描述胜过 GPT-4V 蒸馏。

## 问题

开放权重的 VLM 有数百个。从「不错」到「最先进」的大部分差距并不来自架构，而是来自数据、分辨率调度和编码器选择。当模型表现不佳时，知道先转哪个旋钮，能帮你省下一次 500 万 GPU 小时的错误。

2023 年的浪潮（LLaVA-1.5、InstructBLIP、MiniGPT-4）靠描述对预训练加 LLaVA-Instruct-150k 跑起来。这是不错的基线，但 MMMU 最高只到 35% 左右。

2024 年的浪潮（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs）做了穷举式消融。结果既出人意料，又很实用。

## 概念

### 五轴设计空间

Idefics2（Laurençon 等，2024）为这些轴命名：

1. 图像编码器。CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。编码器在 patch 大小、分辨率和预训练目标上各不相同。
2. 连接器。MLP（2-4 层）、Q-Former（32 个 query + 交叉注意力）、Perceiver Resampler（64 个 query）、C-Abstractor（卷积 + 双线性池化）。
3. 语言模型。Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM 规模是参数成本的主要来源。
4. 训练数据。描述对（CC3M、LAION）、交错图文（OBELICS、MMC4）、指令数据（LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. 分辨率调度。固定 224/336/448、AnyRes、原生动态分辨率。训练中逐步爬升或保持不变。

每个生产级 VLM 都要在每条轴上做出选择。MMMU 分数的大部分方差由轴 1、4、5 解释，而不是由你选了哪个连接器解释。

### 轴 1：编码器 > 连接器

MM1 第 3.2 节表明：从 CLIP ViT-L/14 换到 SigLIP SO400m/14，MMMU 提升 3 分以上。把连接器从 MLP 换成 Perceiver Resampler，提升不到 1 分。Idefics2 复现了这一点：SigLIP > CLIP，在相同 token 数下 Q-Former ≈ MLP ≈ Perceiver。

Cambrian-1 的「Cambrian Vision Encoders Match-Up」（Tong 等，2024）在视觉中心基准（CV-Bench）上跑了 20 多种编码器。榜首是 DINOv2 与 SigLIP 的混合；CLIP 处在中游；ImageBind 和 ViT-MAE 更低。从 CLIP ViT-L 到 DINOv2 ViT-g/14，在 CV-Bench 上差距约 5-7 分。

2026 年开放 VLM 的默认编码器是 SigLIP 2 SO400m/14，用于语义 + 稠密特征；如果你需要分割/定位，还可以与 DINOv2 ViT-g/14 的特征拼接（Cambrian 的「Spatial Vision Aggregator」就是这么做的）。

### 轴 2：连接器设计差别不大

MM1、Idefics2、Prismatic 和 MM-Interleaved 都得出同一结论：在固定视觉 token 数下，连接器架构几乎不重要。在相同 token 预算下，对均值池化 patch 做 2 层 MLP，与 32-query Q-Former 的性能差距在 1 分以内。

真正重要的是 token 数。更多视觉 token = 更多 LLM 算力 = 性能提升，直到某一点后收益递减。每图 64 个 token 对 OCR 来说太少。576-1024 个 token 是大多数开放 VLM 的最佳区间。只有文档和图表才需要 2048+。

Q-Former vs MLP 是成本问题，不是质量问题：Q-Former 不管图像分辨率多高，都把 token 上限卡在 32-64；MLP 则输出所有 patch token。高分辨率输入时，Q-Former 节省 LLM 上下文；低分辨率时，差别只是噪声。

### 轴 3：LLM 规模决定上限

把 LLM 从 7B 翻倍到 13B，在每篇 VLM 论文中都能稳定带来 MMMU 上 2-4 分的提升。到 70B 时，大多数基准趋于饱和。VLM 的多模态推理上限就是 LLM 的文本推理上限——视觉编码器只能向它提供输入，不能替它推理。

这就是为什么 Qwen2.5-VL-72B 和 Claude Opus 4.7 能在 MMMU-Pro 和 ScreenSpot-Pro 上碾压对手：语言大脑足够大。7B VLM 无法靠巧妙的连接器设计替代 70B VLM。

### 轴 4：数据——详细的人类描述胜过蒸馏

Molmo + PixMo（Deitke 等，2024）是每个人都该读的 2024 年成果。Allen AI 让人类标注员用 1-3 分钟的密集语音转文字来描述图像，产出了 712K 张密集标注的图像。训练数据中完全没有 GPT-4V 蒸馏。

Molmo-72B 在 11/11 个基准上击败 Llama-3.2-90B-Vision。差距不在架构，而在描述质量。详细的人类描述每图包含的信息量是简短网络描述文字的 5-10 倍，并且在 GPT-4V 蒸馏产生幻觉的地方仍然事实可靠。

ShareGPT4V（Chen 等，2023）和 Cauldron（Idefics2）沿用了同样的打法，混合人类与 GPT-4V 描述。趋势很清楚：对 2026 年的前沿来说，描述密度 > 描述数量 > 蒸馏的便利性。

### 轴 5：分辨率及其调度

Idefics2 的消融：384 -> 448 增加 1-2 分。448 -> 980 配合图像切分（AnyRes）在 OCR 基准上再增加 3-5 分。固定分辨率训练会在中等精度处平台化；分辨率爬升（从 224 开始，到 448 或原生分辨率结束）训练更快、最终精度更高。

Cambrian-1 研究了分辨率与 token 数的权衡：在固定算力下，你可以选择低分辨率配更多 token，或高分辨率配更少 token。OCR 上高分辨率胜出；通用场景理解上低分辨率多 token 胜出。

2026 年的生产配方：Stage 1 固定 384 训练，Stage 2 用动态分辨率最高到 1280，用于 OCR 密集任务。

### Prismatic 的受控比较

Prismatic VLMs（Karamcheti 等，2024）是控制了所有轴的论文。同样的 13B LLM、同样的指令数据、同样的评估——一次只变一条轴。结果：

- 每图视觉 token 数解释了约 60% 的方差。
- 编码器选择解释了约 20%。
- 连接器架构解释了约 5%。
- 其他所有因素（数据混合、调度器、LR）占剩下的约 15%。

这是一个粗略的分解，但它是文献中对「我应该先消融什么」最干净的回答。

### 2026 年的配方选择器

综合这些证据，2026 年新项目的默认开放 VLM 配方是：

- 编码器：原生分辨率下的 SigLIP 2 SO400m/14，配合 NaFlex；如果你需要分割/定位，再与 DINOv2 ViT-g/14 拼接以获得稠密特征。
- 连接器：对 patch token 做 2 层 MLP。除非 token 受限，否则跳过 Q-Former。
- LLM：Qwen2.5 / Llama-3.1 / Gemma 2，7B 省成本，70B 保质量，按目标延迟挑选。
- 数据：PixMo + ShareGPT4V + Cauldron，再补充任务特定的指令数据。
- 分辨率：动态（长边最小 256 像素，最大 1280 像素）。
- 调度：Stage 1 对齐（仅训练 projector），Stage 2 全量微调，Stage 3 任务特定微调。

上述每个默认值都能追溯到本课末尾所引论文中经过测量的消融。

```figure
l5-vlm-recipe-knobs
```

## 使用它

`code/main.py` 是一个消融表解析器和配方选择器。它编码了 MM1 和 Idefics2 的消融表（精简版），让你可以查询：

- 「给定预算 X 和任务 Y，哪套配方胜出？」
- 「如果我在 7B Llama 上把 SigLIP 换成 CLIP，预期的 MMMU 变化是多少？」
- 「为了得到 80% 置信度的答案，我应该先消融哪条轴？」

输出是一份排序后的配方列表，附带预期基准变化和「优先消融」建议。

## 交付它

本课产出 `outputs/skill-vlm-recipe-picker.md`。给定目标任务组合、算力预算和延迟目标，它会输出完整配方（编码器、连接器、LLM、数据混合、分辨率调度），并引用为每个选择提供依据的消融实验。它能避免工程师每次启动新的 VLM 项目时都重新发明 Idefics2 的消融表。

## 练习

1. 阅读 MM1 第 3.2 节。对于固定的 2B LLM 和 5000 万张图像的预算，哪个编码器胜出？换到 13B LLM 时答案会翻转吗？为什么？

2. Cambrian-1 发现 DINOv2 + SigLIP 拼接在视觉中心基准上优于任一单独使用，但在 MMMU 上没有额外信号。预测哪些基准会受益、哪些会持平。

3. 你的目标是 2B LLM 上的移动端 UI 智能体。选择编码器、连接器、分辨率和数据混合，并用具体的消融表为每个选择辩护。

4. Molmo 发布了 4B 和 72B 模型。4B 与闭源 7B VLM 有竞争力；72B 在 11/11 个基准上击败 Llama-3.2-90B-Vision。这对 LLM 规模平台期假设说明了什么？

5. 设计一个消融表，在 7B VLM 上把数据混合质量与编码器质量隔离开来。最少需要多少次训练？给出四个轴的设置。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 消融（Ablation） | 「只转一个旋钮」 | 多组训练运行，恰好在一个设计空间轴上不同，其余全部保持不变 |
| 连接器（Connector） | 「桥」/「projector」 | 把视觉编码器输出映射到 LLM token 空间的可训练模块（MLP、Q-Former、Perceiver） |
| 详细人类描述（Detailed human caption） | 「密集描述」 | 多句的人类撰写描述（通常 80-300 个 token），比网络 alt 文本信息更丰富 |
| 蒸馏（Distillation） | 「GPT-4V 描述」 | 由更强的专有 VLM 生成的训练数据；方便，但容易继承幻觉 |
| AnyRes / 动态分辨率 | 「高分辨率路径」 | 通过切块或 M-RoPE 喂入大于编码器原生分辨率的图像 |
| 分辨率爬升（Resolution ramp） | 「课程学习」 | 从低分辨率开始逐步提高的训练调度，加快对齐学习 |
| 视觉中心基准（Vision-centric bench） | 「CV-Bench / BLINK」 | 强调细粒度视觉感知而非语言推理的评估 |
| PixMo | 「Molmo 的数据」 | Allen AI 的 712K 密集标注图像数据集；由人类语音转写为密集描述 |

## 延伸阅读

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
