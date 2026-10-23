# LLaVA 与视觉指令微调

> LLaVA（2023 年 4 月）是地球上被复制最多的多模态架构。它用 2 层 MLP 取代了 BLIP-2 的 Q-Former，用朴素的 token 拼接取代了 Flamingo 的门控交叉注意力，并在 15.8 万轮由 GPT-4 根据纯文本标题生成的视觉指令上训练。2023 到 2026 年间构建 VLM 的任何从业者，建的都是某种 LLaVA 变体。LLaVA-1.5 加入了 AnyRes。LLaVA-NeXT 提升了分辨率。LLaVA-OneVision 把图像、多图和视频统一进一个配方。本节课读这个配方，实现投影器，并解释为什么「更简单的一方赢了」。

**Type:** Build
**Languages:** Python (stdlib, projector + instruction-template builder)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 11 (LLM Engineering — instruction tuning)
**Time:** ~180 minutes

## 学习目标

- 构建一个 2 层 MLP 投影器，把 ViT patch 嵌入（维度 1024）映射到 LLM 的嵌入维度（维度 4096）。
- 走一遍 LLaVA 的两阶段配方：(1) 在 55.8 万对标题上做投影器对齐，(2) 在 15.8 万轮 GPT-4 生成的对话上做视觉指令微调。
- 构造一个 LLaVA 格式的 prompt，包含图像 token 占位符、系统 prompt 和用户/助手轮次。
- 解释为什么社区从 Q-Former 转向 MLP，尽管 Q-Former 在 token 预算上占优。

## 问题

BLIP-2 的 Q-Former（第 12.03 课）把图像压缩成 32 个 token。干净、高效、基准成绩好。但它有两个问题。

第一，Q-Former 可训练，但它的损失不是最终任务。阶段 1 训练 ITC+ITM+ITG。阶段 2 训练 LM 损失。query 学到某种中间表示，然后 LLM 必须把它解码出来。信息在瓶颈处丢失。

第二，Q-Former 有 1.88 亿参数，在 LLaVA 的 2023 年规模下，你必须把它与目标 LLM 一起协同设计。换 LLM，重训 Q-Former。换视觉编码器，重训。每种组合都是一个独立的研发项目。

LLaVA 的答案简单得令人难堪：拿 ViT 的 576 个 patch token，让每个通过一个 2 层 MLP（`1024 → 4096 → 4096`），把全部 576 个塞进 LLM 的输入序列。没有瓶颈。没有在奇怪目标上的阶段 1 预训练。只在直接的 LM 损失上训练 MLP。

数据从哪来？LLaVA 的第二个洞见：用（纯文本的）GPT-4 生成指令数据。把一张图的 COCO 标题和边界框数据喂给 GPT-4，让它生成对话、描述和复杂推理问题。15.8 万轮指令-回复，免费。没有人工标注。

结果：一个 VLM，在 8 张 A100 上跑一天，在 MMMU 上击败 Flamingo，并发布了可被社区扩展的开放 checkpoint。到 2023 年底，它已经衍生出 50 多个 fork。

## 概念

### 架构

13B 的 LLaVA-1.5：
- 视觉编码器：CLIP ViT-L/14 @ 336（阶段 1 冻结，阶段 2 可选解冻）。
- 投影器：带 GELU 激活的 2 层 MLP，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后来是 Llama-3.1-8B）。

图像 + 文本 prompt 的前向：

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

图像占据 LLM 上下文的 576 个 token。在 2048 上下文下，文本还剩 1472 个 token。在 32k 上下文下，这只是舍入误差。

### 阶段 1：投影器对齐

冻结 ViT。冻结 LLM。只训练 2 层 MLP。数据集：55.8 万对图像-标题（LAION-CC-SBU）。损失：在标题上的语言建模，以投影后的图像 token 为条件。

batch 128、单 epoch，几个小时就能完成。投影器学会把 ViT 空间映射到 LLM 空间。没有任务特定的监督。

### 阶段 2：视觉指令微调

解冻投影器（仍可训练）。解冻 LLM（通常全量，有时 LoRA）。在 15.8 万轮视觉指令上训练。

指令数据才是诀窍。Liu 等人是这样生成的：
1. 取一张 COCO 图像。
2. 提取文本描述（5 条人工标题 + 边界框列表）。
3. 用三种 prompt 模板发给 GPT-4：
   - 对话：「围绕这张图生成一段用户与助手之间的来回对话。」
   - 详细描述：「对这张图给出丰富、详细的描述。」
   - 复杂推理：「提出一个需要就图像进行推理的问题，然后回答它。」
4. 解析 GPT-4 的输出为（指令，回复）对。

这些都没有直接接触图像——只有文本描述。GPT-4 会针对这些描述产生看似合理的幻觉。有些噪声，但有效：15.8 万轮足以解锁对话能力。

### 为什么社区照抄

- 没有阶段 1 专属的损失要调。全程 LM 损失。
- 投影器数小时就能训完，不是数天。
- LLM 可以换（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3），只需重训投影器。
- 视觉指令数据流水线用 GPT-4，针对新领域重新生成很便宜。

### LLaVA-1.5 与 LLaVA-NeXT

LLaVA-1.5（2023 年 10 月）新增：
- 学术任务数据（VQA、OKVQA、RefCOCO）混入指令微调。
- 更好的系统 prompt。
- 2048 → 32k 上下文。

LLaVA-NeXT（2024 年 1 月）新增：
- AnyRes：把高分辨率图像切成 2x2 或 1x3 的 336x336 裁剪网格，外加一张全局低分辨率缩略图。每个裁剪变成 576 个 token；每张图总共约 2880 个视觉 token。OCR 和图表任务大幅提升。
- 更好的指令数据混合，加入 ShareGPT4V（高质量的 GPT-4V 标题）。
- 更强的基座 LLM（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

第 12.08 课深入讲 OneVision。简短版：相同的投影器，但用一套课程训练，在单个模型里覆盖单图、多图和视频，共享视觉 token 预算。

### 与 Q-Former 的对比

| | Q-Former（BLIP-2） | MLP（LLaVA） |
|---|---|---|
| 每图视觉 token | 32 | 576（基础）或 2880（AnyRes） |
| 可训练参数 | 1.88 亿 + LM | 4000 万 + LM |
| 阶段 1 损失 | ITC+ITM+ITG | 仅 LM |
| LLM 即插即用 | 需要重训 | 最小重训即可更换 |
| 多图 | 别扭 | 自然（拼接） |
| 视频 | 别扭 | 自然（逐帧拼接） |
| token 预算 | 小 | 大 |

MLP 在简洁性和 token 灵活性上胜出。Q-Former 在 token 预算上胜出。到 2023 年底，token 预算不再是硬约束（LLM 上下文增长到 32k–128k+），简洁性占据主导。

### prompt 格式

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` 是一个占位 token。在分词之前，它被替换成 576 个视觉 token（AnyRes 下是 2880 个）。分词器看到的序列比训练时略长，但 LLM 能处理这种新输入，因为阶段 1 教会了它。

### 参数经济性

LLaVA-1.5-7B 拆解：
- CLIP ViT-L/14 @ 336：303M（阶段 1 冻结，阶段 2 常解冻）。
- 投影器（2x 线性）：约 22M 可训练。
- Llama-7B：7B。
- 总计：7.3B 参数。阶段 2 可训练：完整 7B + 22M 投影器。

阶段 2 的训练成本：8xA100 上约 20 小时。这是关键数字——一天、一个节点、可复现。这就是 LLaVA 传播开来的原因。

```figure
mm-llava-projector
```

## 使用它

`code/main.py` 实现：

1. 纯 Python 的 2 层 MLP 投影器（玩具规模 dim 16 → 32 → 32）。
2. prompt 构建流水线：系统 prompt + `<image>` 替换为 N 个投影 token + 用户轮次 + 助手生成占位。
3. 一个可视化器，展示 576-token 视觉块在 LLM 上下文中的样子（占 2k / 32k / 128k 上下文的百分比）。

## 交付它

本节课产出 `outputs/skill-llava-vibes-eval.md`。给定一个 LLaVA 家族 checkpoint，它运行一套 10 条 prompt 的 vibes-eval（3 条描述、3 条 VQA、2 条推理、2 条拒答），并报告人类可读的评分卡。这不是基准；这是一个冒烟测试，用来确认投影器和 LLM 衔接良好。

## 练习

1. 计算 `1024 → 4096 → 4096` 的 2 层 MLP 投影器的可训练参数量。带 GELU 和偏置，它占 LLaVA-13B 的多大比例？

2. 为一个「拒答」案例构造 LLaVA prompt——图中包含一位私人个体。写出预期的助手回复。为什么 LLaVA 应该零样本拒答？需要什么训练数据来强化这种拒答？

3. 阅读 LLaVA-NeXT 博客的 AnyRes 一节。计算一张 1344x672 图像在 AnyRes 下的视觉 token 数。与 336x336 基础 576 token 比较。

4. LLaVA 阶段 1 的投影器用标题上的 LM 损失训练。如果跳过阶段 1 直接进入阶段 2（视觉指令微调）会怎样？引用 Prismatic VLMs 的消融实验（arXiv:2402.07865）给出答案。

5. LLaVA-Instruct-150k 用 GPT-4 加 COCO 标题生成指令。对于一个新领域（医学 X 光、卫星影像），描述生成领域指令的四步数据流水线。每一步可能出什么问题？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Projector | 「MLP 桥梁」 | 带 GELU 的 2 层 MLP，把 ViT 维度映射到 LLM 维度 |
| Image token | 「<image> 占位符」 | 推理前被替换为 N 个投影视觉 token 的 prompt 标记 |
| Visual instruction tuning | 「LLaVA 阶段 2」 | 在 GPT-4 生成的（图像，指令，回复）三元组上训练 |
| Stage 1 alignment | 「投影器预训练」 | 冻结 ViT 和 LLM，用标题上的 LM 损失训练投影器 |
| AnyRes | 「多裁剪平铺」 | 把高分辨率图像切成瓦片网格，并拼接每个瓦片的视觉 token |
| LLaVA-Instruct | 「GPT-4 生成」 | 由 COCO 标题 + GPT-4 合成的 15.8 万条指令-回复对 |
| Vision encoder freeze | 「骨干锁定」 | CLIP 权重在阶段 1 不更新，阶段 2 有时也不更新 |
| ShareGPT4V | 「更好的标题」 | GPT-4V 生成的 100 万条稠密标题，用于更高质量的对齐 |
| VQA | 「视觉问答」 | 回答关于图像的自由形式问题的任务 |
| Prismatic VLMs | 「设计空间论文」 | Karamcheti 2024 的消融，系统测试投影器和数据选择 |

## 延伸阅读

- [Liu 等——Visual Instruction Tuning（arXiv:2304.08485）](https://arxiv.org/abs/2304.08485)——LLaVA 论文。
- [Liu 等——Improved Baselines with Visual Instruction Tuning（arXiv:2310.03744）](https://arxiv.org/abs/2310.03744)——LLaVA-1.5。
- [Chen 等——ShareGPT4V（arXiv:2311.12793）](https://arxiv.org/abs/2311.12793)——稠密标题数据集。
- [Karamcheti 等——Prismatic VLMs（arXiv:2402.07865）](https://arxiv.org/abs/2402.07865)——设计空间消融。
- [Li 等——LLaVA-OneVision（arXiv:2408.03326）](https://arxiv.org/abs/2408.03326)——统一单图、多图、视频。
