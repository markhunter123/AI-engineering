# 从 CLIP 到 BLIP-2——Q-Former 作为模态桥梁

> CLIP 能对齐图像和文本，却不会生成标题、回答问题或进行对话。BLIP-2（Salesforce，2023）用一个小型可训练桥梁解决了这个问题：32 个可学习的 query 向量通过交叉注意力关注冻结 ViT 的特征，然后直接插入冻结 LLM 的输入流。1.88 亿参数的桥梁把 11B 的 LLM 连到 ViT-g/14 上。到 2026 年，所有基于适配器的 VLM——MiniGPT-4、InstructBLIP、LLaVA 一系的模型——都是它的后代。本节课解读 Q-Former 的架构，解释它的两阶段训练，并构建一个玩具版本，把视觉 token 送进冻结的文本解码器。

**Type:** Build
**Languages:** Python (stdlib, cross-attention + learnable-query demo)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 7 (Transformers)
**Time:** ~180 minutes

## 学习目标

- 解释为什么在冻结视觉编码器和冻结 LLM 之间放一个可训练瓶颈，在成本和稳定性上都优于端到端微调。
- 实现一个交叉注意力块，让一组固定的可学习 query 关注外部图像特征。
- 走一遍 BLIP-2 的两阶段预训练：表征阶段（ITC + ITM + ITG），然后生成阶段（带冻结解码器的 LM 损失）。
- 比较 Q-Former 与 LLaVA 中更简单的 MLP 投影器，并论证各自何时占优。

## 问题

你有一个冻结的 ViT，每张图产出 256 个 1408 维的 patch token。你有一个冻结的 7B LLM，期望 4096 维的 token 嵌入。显然的桥梁——一个从 1408 到 4096 的线性层——能行，但把所有 256 个 patch token 喂进 LLM 的上下文，每张图要花掉 256 个额外 token。一批 32 张图，仅视觉模态就消耗 8192 个 token。

BLIP-2 的问题：你能不能把 256-token 的图像表示压缩成少得多的 token（比如 32 个），同时保留足够的信息，让 LLM 能描述、回答问题并推理图像？而且你能否在不动冻结骨干的情况下训练这个桥梁，把训练成本限制在桥梁自身的参数上？

答案：Q-Former。32 个可学习的「query」向量交叉关注 ViT 的 patch token，产生一个 32-token 的视觉摘要供 LLM 消费。总共 1.88 亿参数。在接触 LLM 之前，先用对比、匹配和生成目标训练好。

## 概念

### 可学习 query

Q-Former 的核心技巧：不让 LLM 的文本 token 去关注图像 patch，而是引入一组新的 32 个可学习 query 向量 `Q`，让它们去关注图像 patch。query 是模型的参数——训练中学习得到，所有图像共用同样的 32 个 query。

经过交叉注意力后，每个 query 都持有图像的一个压缩摘要——「描述主要物体」「描述背景」「数一数物体」等等。query 并不会真的按语义标签专门化；它们学到的是任何能让下游损失下降的编码。

### 架构

Q-Former 是一个小型 Transformer（12 层，约 1 亿参数），有两条路径：

1. Query 路径：32 个 query 向量先经过自注意力（彼此之间），再对冻结 ViT 的 patch token 做交叉注意力，然后过 FFN。
2. 文本路径：一个类 BERT 的文本编码器，与 query 路径共享自注意力和 FFN 权重。文本路径的交叉注意力被禁用。

训练时两条路径都运行。query 和文本通过共享的自注意力交互，这意味着在需要时（ITM、ITG），query 可以以文本为条件。推理时交给 VLM 的只有 query，产出 32 个视觉 token。

### 两阶段训练

BLIP-2 分两个阶段预训练：

阶段 1：表征学习（不涉及 LLM）。三个损失：
- ITC（图像-文本对比）：在池化后的 query token 与文本 CLS token 之间做 CLIP 式对比。
- ITM（图像-文本匹配）：二元分类器——这对图文是否匹配？难负样本挖掘。
- ITG（图像接地的文本生成）：在文本上接因果 LM 头，以 query 为条件。迫使 query 编码可生成文本的内容。

只有 Q-Former 训练。ViT 冻结。不涉及 LLM。

阶段 2：生成学习。接上冻结的 LLM（OPT-2.7B 或 Flan-T5-XL 等）。用一个小的线性层把 32 个 query 输出投影到 LLM 的嵌入维度。把它们前置到文本 prompt。只在拼接后的 prompt + 图像 + 标题序列上，用 LM 损失训练线性投影和 Q-Former。

阶段 2 之后，Q-Former + 投影就是完整的视觉适配器。推理时：图像 → ViT → Q-Former → 线性投影 → 前置到文本 → 冻结 LLM 输出。

### 参数经济学

BLIP-2 用 ViT-g/14（1.1B，冻结）+ OPT-6.7B（6.7B，冻结）+ Q-Former（188M，训练）= 总共 8B，训练 188M。仅 Q-Former 就约占整个技术栈参数的 2.4%。训练成本反映了这一点：几张 A100 上跑几天，而端到端要几周。

质量：BLIP-2 在零样本 VQA 上追平或超过 Flamingo-80B，而体积小 50 倍。这个桥梁有效。

### InstructBLIP 与指令感知的 Q-Former

InstructBLIP（2023）给 Q-Former 增加了一个额外输入：指令文本本身。在交叉注意力时，query 现在既能访问图像 patch，也能访问指令。query 可以按指令专门化（「数一数汽车」「描述氛围」），而不是学习单一的固定摘要。在留出任务上带来基准提升。

### MiniGPT-4 与仅投影器路线

MiniGPT-4 保留 Q-Former，但只训练输出线性投影，其余全部冻结。便宜，但代价是质量——query 是 BLIP-2 的，不是你的。适合快速迭代，但不是最佳架构。

### LLaVA 为什么走向更简单

LLaVA（2023，第 12.05 课）用普通的 2 层 MLP 取代 Q-Former，把每个 ViT patch token 投影到 LLM 空间——24x24 网格每张图 576 个 token，全部喂给 LLM。压缩更差，但让 LLM 能关注原始 patch。当时这有争议；到 2023 年底它成了主导，因为视觉指令数据（LLaVA-Instruct-150k）证明 MLP 可以被训练到保留足够的信号。权衡是：LLaVA 的上下文填得更快，但它能自然扩展到多图和视频。

到 2026 年，领域出现分化：Q-Former 在 token 预算重要的地方存活（长视频、多图）；MLP 投影器在优先考虑每个 token 的原始质量时占主导。

### 门控交叉注意力：祖先 Flamingo

Flamingo（第 12.04 课）早于 BLIP-2，使用了相同的交叉注意力思路，但放在冻结 LLM 的每一层，而不是作为单一桥梁。BLIP-2 表明你可以只压缩到输入层并且仍然有效。Gemini 和 Idefics 将两者结合：交错的输入 token，加上可选的 gated cross-attention 用于上下文少样本。

### 2026 年的后代

- Q-Former：BLIP-2、InstructBLIP、MiniGPT-4，以及大多数视频-语言模型（出于 token 预算考虑）。
- Perceiver resampler：Flamingo 的变体（第 12.04 课）；Idefics 家族、Eagle、OmniMAE。
- MLP projector：LLaVA、LLaVA-NeXT、LLaVA-OneVision、Cambrian-1。
- Attention pool：VILA、PaliGemma。

四种都成立。决定性的问题是：你的约束在 token 预算上，还是在每 token 质量上。

```figure
modality-projection
```

## 使用它

`code/main.py` 构建一个 stdlib 的 Q-Former 风格交叉注意力：

1. 模拟 256 个图像 patch token（维度 128）。
2. 实例化 32 个可学习 query（维度 128）。
3. 运行缩放点积交叉注意力（Q 来自 query，K/V 来自 patch）。
4. 通过线性层投影到 LLM 维度（512）。
5. 输出 32 个可供 LLM 使用的视觉 token。

所有数学都用纯 Python（对向量做嵌套循环）。玩具级别，但形状正确。注意力权重矩阵会被打印出来，让你看到每个 query 从哪些 patch 取信息。

## 交付它

本节课产出 `outputs/skill-modality-bridge-picker.md`。给定目标 VLM 配置（视觉编码器 token 数、LLM 上下文预算、部署约束、质量目标），它推荐 Q-Former、MLP 或 Perceiver resampler，附简短理由和每种桥梁的参数估算。

## 练习

1. 用 PyTorch 实现交叉注意力块。验证：32 个 query、256 个 key/value 时，注意力权重矩阵是 32 x 256，且 softmax 后每行之和为 1。

2. 在 BLIP-2 阶段 1 中，Q-Former 同时运行三个损失：ITC、ITM、ITG。用伪代码写出各自的前向签名。哪一个需要文本编码器路径处于激活状态？

3. 比较参数量：Q-Former（12 层，768 隐藏维度）vs 2 层 MLP 投影器（1408 → 4096，两层）。在多大的 LLM 规模下，1.88 亿 Q-Former 的成本能在训练效率上收回？

4. 阅读 BLIP-2 论文（arXiv:2301.12597）第 3.2 节关于 Q-Former 初始化的内容。解释为什么从 BERT-base 初始化（而非随机）能加速收敛。

5. 对于一段 10 分钟、以 1 FPS 采样到 60 帧的视频，计算两种方案的每帧 token 成本：(Q-Former → 32 token/帧) vs (MLP 投影器 → 576 token/帧)。哪一种能放进 128k token 的 LLM 上下文窗口？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Q-Former | 「查询 Transformer」 | 小型 Transformer，带 32 个可学习 query 向量，交叉关注冻结 ViT 特征 |
| Learnable queries | 「视觉的软 prompt」 | 作为交叉注意力 query 侧的一组固定参数；按模型学习，跨所有输入共享 |
| Cross-attention | 「Q 来自这里，K/V 来自那里」 | query、key、value 来自不同来源的注意力；query 据此从 ViT patch 取信息 |
| ITC | 「图像-文本对比」 | 施加在 Q-Former 池化 query 与文本 CLS 上的 CLIP 式损失 |
| ITM | 「图像-文本匹配」 | 在难负样本挖掘对上的二元分类器；迫使 query 区分细粒度不匹配 |
| ITG | 「图像接地的文本生成」 | 以 query 为条件生成文本的因果 LM 损失；迫使 query 编码可被文本解码的内容 |
| Two-stage pretraining | 「先表征后生成」 | 阶段 1 单独训练 Q-Former（ITC/ITM/ITG）；阶段 2 接冻结 LLM，只训投影 + Q-Former |
| Frozen backbone | 「不微调」 | 视觉编码器和 LLM 权重固定；只有桥梁训练 |
| Projection head | 「线性映射到 LLM 维度」 | 把 Q-Former 输出映射到 LLM 嵌入维度的最后线性层 |
| Perceiver resampler | 「Flamingo 的版本」 | 类似的可学习 query 交叉注意力，Flamingo 在每一层使用，而非作为单一桥梁 |

## 延伸阅读

- [Li 等——BLIP-2（arXiv:2301.12597）](https://arxiv.org/abs/2301.12597)——核心论文。
- [Li 等——BLIP（arXiv:2201.12086）](https://arxiv.org/abs/2201.12086)——带有 ITC/ITM/ITG 三件套的前身。
- [Li 等——ALBEF（arXiv:2107.07651）](https://arxiv.org/abs/2107.07651)——「先对齐再融合」，阶段 1 训练的思想祖先。
- [Dai 等——InstructBLIP（arXiv:2305.06500）](https://arxiv.org/abs/2305.06500)——指令感知的 Q-Former。
- [Zhu 等——MiniGPT-4（arXiv:2304.10592）](https://arxiv.org/abs/2304.10592)——仅投影器路线。
- [Jaegle 等——Perceiver IO（arXiv:2107.14795）](https://arxiv.org/abs/2107.14795)——可学习 query 交叉注意力的通用架构。
