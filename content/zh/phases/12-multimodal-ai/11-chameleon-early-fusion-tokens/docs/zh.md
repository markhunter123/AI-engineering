# Chameleon 与早期融合的纯 token 多模态模型

> 我们目前看到的每个 VLM 都把图像和文本分开。视觉 token 来自视觉编码器，流入投影器，然后在 LLM 内部与文本相遇。视觉和文本词表从不重叠。Chameleon（Meta，2024 年 5 月）问道：如果它们重叠呢？训练一个 VQ-VAE，把图像变成来自共享词表的离散 token 序列。于是每个多模态文档就是一个序列——文本 token 与图像 token 交错，单一自回归损失。副作用：模型能生成混合模态输出——在一次推理调用中交替输出文本和图像 token。本课解读早期融合的论点，并端到端构建一个玩具版本。

**Type:** Build
**Languages:** Python (stdlib, VQ-VAE tokenizer + interleaved decoder)
**Prerequisites:** Phase 12 · 05, Phase 8 (Generative AI)
**Time:** ~180 minutes

## 学习目标

- 解释共享词表 + 单一损失为什么改变了模型能做的事。
- 描述 VQ-VAE 如何把图像 token 化成与 Transformer 下一 token 目标兼容的离散序列。
- 说出 Chameleon 的训练稳定性技巧：QK-Norm、dropout 放置、LayerNorm 顺序。
- 比较 Chameleon 与 BLIP-2 的 Q-Former 方案，并说明各自何时是正确选择。

## 问题

基于适配器的 VLM（LLaVA、BLIP-2、Qwen-VL）把文本和图像当作两种不同东西。文本 token 走 `embed(text_token)`；图像走 `visual_encoder(image) → projector → ... pseudo_tokens`。模型有两条输入路径，在中途汇合。

三个后果：

1. LLM 只能消费图像，不能输出图像。输出只有文本。
2. 混合模态文档（像文章那样段落与图像交替）很别扭——你要么在模型外解析多模态输入，要么串接多次生成。
3. 分布不匹配。视觉 token 和文本 token 处在隐藏空间的不同区域，造成微妙的对齐问题。

Chameleon 拒绝这个前提：图像不过是由共享词表中的离散 token 组成的序列。在交错文档上训练模型，一个损失、一个自回归解码器，就能免费解锁混合模态生成。

## 概念

### VQ-VAE 作为图像 tokenizer

tokenizer 是一个向量量化变分自编码器。架构：

- 编码器：CNN + ViT，把图像映射到空间特征图，比如 32x32 个 256 维特征。
- 码本：K 个学习到的向量组成的词表（Chameleon 用 8192），同样是 256 维。
- 量化：对每个空间特征，按 L2 距离查找最近的码本项。用整数索引替换连续特征。
- 解码器：CNN，把量化特征还原为像素。

训练：VAE 重建损失 + commitment 损失 + 码本损失。码本索引构成图像的离散字母表。

对 Chameleon 而言：一张图变成 32*32 = 1024 个 token，取自 8192 的词表。与文本 token（来自 LLM 的 BPE 词表，比如 32000）拼接。最终词表：40192。Transformer 看到一个序列、一个损失。

### 共享词表

Chameleon 的词表结合了文本 token、图像 token 和模态分隔符。每个 token 有一个 ID。输入 embedding 层把每个 ID 映射为 D 维隐藏向量。输出投影把隐藏向量映射回词表 logits。Softmax 选出下一个 token，无论什么模态。

分隔符很重要：`<image>` 和 `</image>` 标签括住图像 token 序列。生成时，如果模型输出 `<image>`，下游软件就知道接下来的 1024 个 token 是 VQ 索引，要送给解码器渲染像素。

### 混合模态生成

推理就是在共享词表上做下一 token 预测。示例提示：「画一只猫并描述它。」Chameleon 输出：

```
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

模型自主决定顺序——它可能先图后文、先文后图，或交错。同一个解码器，同一个损失。

对比生成只输出文本的适配器 VLM，Chameleon 重新打开了模型输出模态这个问题。

### 训练稳定性——QK-Norm、dropout、LayerNorm 顺序

早期融合训练在大规模下不稳定。Chameleon 论文记录了三个技巧：

- QK-Norm。在注意力内部、点积之前，对 query 和 key 投影施加 LayerNorm。防止深层 logit 幅度爆炸。2024 年之后的大型模型普遍采用。
- Dropout 放置。在每个残差相加之后放 dropout，而不只是注意力和 MLP 之后。当图像 token 的梯度可能占主导时，需要更多正则。
- LayerNorm 顺序。残差分支上用 Pre-LN（标准做法），再在最后一个 block 的跳跃连接上加一个额外 LN。稳定最后一层的梯度流。

没有这些技巧，34B 参数的 Chameleon 训练在多个 checkpoint 处发散。有了它们，训练收敛。训练配方与架构同样是贡献的一部分。

### tokenizer 的重建上限

VQ-VAE 是有损的。8192 个码本项、每张 512x512 图像 1024 个 token 时，重建 PSNR 上限约 26-28 dB。这足以生成可辨认的图像，但明显差于连续空间扩散（Stable Diffusion 3 达到 32+ dB）。

tokenizer 就是瓶颈。更好的 tokenizer（MAGVIT-v2、IBQ、SBER-MoVQGAN）能抬高上限。Emu3（第 12.12 课）仅凭更好的 tokenizer 就达到了 SDXL 质量的生成。

### Chameleon 与 BLIP-2 / LLaVA 的对比

Chameleon（早期融合，共享词表）：
- 一个损失，一个解码器。
- 生成混合模态输出。
- tokenizer 是质量上限。
- 昂贵：推理路径上每生成一张图都要跑 VQ-VAE 解码器。

BLIP-2 / LLaVA（后期融合，分离塔）：
- 视觉进，只出文本。
- 复用预训练 LLM。
- 理解任务上没有 tokenizer 瓶颈。
- 便宜：单次前向传播。

按任务选择。需要图像生成，选 Chameleon 家族。只需要理解，适配器 VLM 更简单，复用更多预训练算力。

### Fuyu 与 AnyGPT

Fuyu（Adept，2023）是相关方案：完全跳过独立的视觉编码器，把原始图像 patch 像 token 一样送入 LLM 的输入投影，不需要 tokenizer。比 Chameleon 简单，但失去了共享词表的输出生成能力。

AnyGPT（Zhan 等，2024）把 Chameleon 扩展到四种模态：文本、图像、语音、音乐。每种都用同样的 VQ-VAE 技巧，共享 Transformer。任意到任意生成。第 12.16 课会讲更多。

```figure
vq-codebook
```

## 使用它

`code/main.py` 构建一个玩具级端到端早期融合模型：

- 一个微型 VQ-VAE 风格量化器，把 8x8 patch 映射为码本索引（K=16）。
- 共享词表：(文本 id 0..31) + (图像 id 32..47) + (分隔符 48, 49)。
- 一个玩具自回归解码器（bigram 表），在合成描述 + 图像 token 序列上训练。
- 采样循环，给定提示后输出交替的文本 + 图像 token。

代码有意把 Transformer 保持得很小（bigram），这样你可以端到端追踪信号流。

## 交付它

本课产出 `outputs/skill-tokenizer-vs-adapter-picker.md`。给定产品规格（只需理解 vs 理解 + 生成、所需图像质量、成本预算），它在 Chameleon 家族（早期融合）和 LLaVA 家族（后期融合）之间选择，并用定量经验规则说明理由。

## 练习

1. Chameleon 使用 K=8192 个码本项、每张 512x512 图像 1024 个 token。估计相对 24 位 RGB 图像的压缩比。有损吗？损失多大？

2. 一张 4K 图像（3840x2160）在相同 VQ-VAE 密度下产生多少图像 token？Chameleon 风格模型能在一次推理调用中生成 4K 图像吗？先坏掉的是什么——上下文、tokenizer 质量还是 KV cache？

3. 用纯 Python 实现 QK-Norm。给定 64 维 query 和 key，展示 LayerNorm 前后的点积。为什么深层需要控制幅度？

4. 阅读 Chameleon 第 2.3 节关于训练稳定性的内容。描述论文在没有 QK-Norm 的 34B 模型上观察到的确切失效模式。「范数爆炸」的特征是什么？

5. 扩展玩具解码器，让它在纯文本提示下输出混合模态响应。在训练数据分布为 60% 文本优先 / 40% 图像优先时，测量模型选择图像优先 vs 文本优先的频率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 早期融合 | 「统一 token」 | 图像被转换为离散 token，从第一步起共享 Transformer 的词表 |
| VQ-VAE | 「图像 tokenizer」 | CNN + ViT + 码本，把图像映射为 Transformer 可预测的整数索引 |
| 共享词表 | 「一本字典」 | 覆盖文本 + 图像 + 模态分隔符的单一 token ID 空间 |
| QK-Norm | 「注意力稳定器」 | 在 query 和 key 点积之前施加 LayerNorm，防止范数爆炸 |
| 混合模态生成 | 「文本 + 图像输出」 | 一次推理中自主产生交错文本和图像 token |
| 码本大小 | 「K 个条目」 | VQ-VAE 可量化到的离散向量数量；在压缩与保真度之间权衡 |
| tokenizer 上限 | 「重建极限」 | 解码 VQ token 能达到的最佳 PSNR；限制模型的图像质量 |

## 延伸阅读

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
