# CLIP 与对比式视觉-语言预训练

> OpenAI 的 CLIP（2021）证明了一个足够大的想法足以支撑之后五年：只用嘈杂的网络图像-标题对和对比损失，就能把图像编码器和文本编码器对齐到同一个向量空间。零监督标签。4 亿对。由此得到的嵌入空间可以做零样本分类、图像-文本检索，并在 2026 年作为视觉塔接入每一个 VLM。SigLIP 2（2025）用 sigmoid 取代 softmax，以更低成本扩展到 CLIP 之上。本节课从 InfoNCE 推导到 sigmoid 成对损失，并用 stdlib Python 构建训练步骤。

**Type:** Build
**Languages:** Python (stdlib, InfoNCE + sigmoid loss implementations)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 7 (Transformers)
**Time:** ~180 minutes

## 学习目标

- 从互信息推导 InfoNCE 损失，并实现数值稳定的向量化版本。
- 解释 sigmoid 成对损失（SigLIP）为何能扩展到 batch 32768 以上，而不需要 softmax 所要求的 all-gather 开销。
- 通过构造文本模板（`a photo of a {class}`）并对余弦相似度取 argmax，运行零样本 ImageNet 分类。
- 说出 CLIP / SigLIP 预训练给你的四个杠杆：batch 大小、温度、prompt 模板、数据质量。

## 问题

CLIP 之前的视觉是监督式的。收集带标签的数据集（ImageNet：120 万张图像、1000 个类别），训练 CNN，发布。标签很贵，标签会偏向标注者能达成一致的东西，而且不经过微调，标签就无法迁移到新任务。

网络上的图像-标题有十亿以上个松散标注的对，免费。一张金毛寻回犬的照片配上 alt 文本「my dog Max in the park」就带有监督信号——文本描述了图像。问题是：你能把它变成有用的训练吗？

CLIP 的回答：把图像-标题对当成匹配任务。给定一批 N 张图像和 N 条标题，学会在 N-1 个干扰项中把每张图像与它自己的标题匹配起来。监督信号是「这两样东西属于一起；那 N-1 个不属于」。没有类别标签。没有人工标注。只有一个对比损失。

得到的嵌入空间做到的比 CLIP 训练时更多。ImageNet 零样本之所以奏效，是因为「a photo of a cat」的嵌入靠近那些从未被显式标注为猫的图片。这就是催生 2026 年每一个 VLM 的那次押注。

## 概念

### 双编码器

CLIP 有两个塔：

- 图像编码器 `f`：ViT 或 ResNet，每张图像输出一个 D 维向量。
- 文本编码器 `g`：小型 Transformer，每条标题输出一个 D 维向量。

两个塔都把输出归一化为单位长度。相似度为 `cos(f(x), g(y)) = f(x)^T g(y)`，因为二者都是单位范数。

对一批 $N$ 个（图像，标题）对，构建形状为 `(N, N)` 的相似度矩阵 `S`：

```math
S[i, j] = \cos(f(x_i), g(y_j)) / \tau
```

其中 `tau` 是可学习的温度（CLIP 初始化为 0.07；在对数空间中学习）。

### InfoNCE 损失

CLIP 对行和列做对称的交叉熵：

```
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。交叉熵里的 softmax 迫使每张图像与自己的标题的匹配程度高于 batch 中其他任何标题。「负样本」就是 batch 里的其他所有项。更大的 batch = 更多负样本 = 更强的信号。CLIP 用 batch 32k 训练；规模很重要。

### 温度

`tau` 控制 softmax 的锐度。低 $\tau$ → 分布尖锐，有难负样本挖掘的效果。高 $\tau$ → 分布平缓，所有样本都贡献。CLIP 学习 $\log(1/\tau)$，并做裁剪以防坍缩。SigLIP 2 固定初始 $\tau$，改用可学习的偏置。

### 为什么 sigmoid 扩展性更好（SigLIP）

softmax 要求整个相似度矩阵同步。在分布式训练中，你必须把每个嵌入 all-gather 到每个副本，再做 softmax。通信量随世界规模的平方增长。

SigLIP 用逐元素 sigmoid 取代 softmax：对每一对 `(i, j)`，损失是对「这两者是不是匹配对？」的二元分类，正类标签是对角线，其余全是负类。损失是：

```math
L = -\frac{1}{N} \sum_{(i, j)} \Big[ y_{ij} \log \mathop{\mathrm{sigmoid}}(S[i,j]) + (1-y_{ij}) \log \mathop{\mathrm{sigmoid}}(-S[i,j]) \Big]
```

`y_ij = 1` 当 `i == j`，否则为 0。每一对的损失彼此独立。不需要 all-gather。每张 GPU 计算自己的局部块再求和。SigLIP 2 能以很低成本扩展到 batch 32k–512k，而 CLIP 则需要按比例增加通信。

### 零样本分类

给定 N 个类别名，为每个类别构造一个文本模板：

```
"a photo of a {class}"
```

用文本编码器嵌入每个模板。用图像编码器嵌入你的图像。余弦相似度的 argmax 就是预测类别。没有在目标类别上训练。

prompt 模板很重要。CLIP 原论文为每个类别用了 80 个模板（plain、artistic、photo、painting 等）并对嵌入取平均。ImageNet 提升 3 个点。现代用法通常选一两个模板。

### 线性探针与微调

零样本是一个基线。线性探针（在冻结的 CLIP 特征上为目标类别训练一个线性层）在域内任务上超过零样本。全量微调在域内超过线性探针，但可能损害零样本迁移。三种范式，三种权衡。

### SigLIP 2：NaFlex 与稠密特征

SigLIP 2（2025）新增：
- NaFlex：单个模型处理可变的宽高比和分辨率。
- 更好的稠密特征，用于分割和深度估计，目标是在 VLM 中作为冻结骨干使用。
- 多语言：在 100 多种语言上训练，而 CLIP 只支持英语。
- 10 亿参数规模，而 CLIP 最高到 4 亿。

在 2026 年的开放 VLM 中，SigLIP 2 SO400m/14 是默认视觉塔。当 LAION-2B 的特定训练分布与你的查询模式匹配时，CLIP 仍然是纯图像-文本检索的默认选择。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google，2021）：与 CLIP 相同的思路，18 亿对规模，90% 是噪声。证明了噪声数据可以扩展。OpenCLIP（LAION）：在 LAION-400M / 2B 上对 CLIP 的开放复现，多种规模，是开放 checkpoint 的首选。EVA-CLIP：从掩码图像建模初始化；VLM 的强骨干。BASIC：Google 的 CLIP+ALIGN 混合。都属于同一家族，数据与调参不同。

### 零样本的天花板

CLIP 级模型在 ImageNet 零样本上大约封顶 76%（CLIP-G、OpenCLIP-G）。再往上要么需要多得多的数据（SigLIP 2 达到 80%+），要么需要架构改动（监督头、更多参数）。这个基准正在饱和；真正的价值在于下游 VLM 所消费的嵌入空间。

```figure
multimodal-fusion
```

## 使用它

`code/main.py` 实现：

1. 一个玩具双编码器（基于哈希的图像特征、文本字符特征），让你不用 numpy 也能看到 InfoNCE 的形状。
2. 纯 Python 的 InfoNCE 损失（通过 log-sum-exp 保证数值稳定）。
3. 用于对比的 sigmoid 成对损失。
4. 一个零样本分类例程：对一组文本 prompt 计算余弦相似度，argmax 给出预测。

运行它并观察损失曲线。绝对值是玩具级别；形状与真实 CLIP 训练器输出的一致。

## 交付它

本节课产出 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径）和一个目标类别列表，它用 CLIP 模板构造文本 prompt，用指定的 checkpoint（例如 `openai/clip-vit-large-patch14`）嵌入两侧，并返回 top-1 / top-5 预测及相似度分数。对不在 prompt 列表中的类别，这个 skill 拒绝做出任何断言。

## 练习

1. 手算 4 对样本的 InfoNCE。构造 4x4 相似度矩阵，跑 softmax，取出对角线，计算交叉熵。用这个手算结果验证你的 Python 实现。

2. SigLIP 在温度之外还使用一个偏置参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当 batch 存在较大的类别不平衡（每行的负样本远多于正样本）时，`b` 起什么作用？阅读 SigLIP 第 3 节（arXiv:2303.15343）。

3. 为猫 vs 狗构建一个零样本分类器。尝试两个 prompt 模板：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图像上测量准确率。模板集成能胜过单个模板吗？

4. 计算 512 张 GPU、batch 32k 运行时，softmax InfoNCE 与 sigmoid 成对损失的通信成本。哪个按 $O(N)$ 扩展，哪个按 $O(N^2)$ 扩展？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP 扩展规律论文（arXiv:2212.07143，Cherti 等）。从图中复现他们关于数据扩展的结论：在固定模型规模下，ImageNet 零样本准确率与训练数据量之间是什么对数线性关系？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| InfoNCE | 「对比损失」 | 对 batch 相似度矩阵的交叉熵；每一项的正样本是它的配对项，负样本是其余全部 |
| Sigmoid loss | 「SigLIP 损失」 | 逐对二元交叉熵；没有 softmax，没有 all-gather，分布式训练扩展成本低 |
| Temperature | 「tau」 | 在 softmax/sigmoid 之前缩放 logit 的标量；控制分布的锐度 |
| Zero-shot | 「免微调分类」 | 用文本 prompt 构造类别嵌入，按余弦相似度分类；不在目标类别上训练 |
| Prompt template | 「a photo of a ...」 | 围绕类别名的文本脚手架；对零样本准确率有 1–5 个点的影响 |
| Dual encoder | 「双塔」 | 一个图像编码器 + 一个文本编码器，输出位于共享的 D 维空间 |
| Hard negative | 「难干扰项」 | 与正样本足够相似的负样本，模型必须费力才能区分 |
| Linear probe | 「冻结 + 一层」 | 只在冻结特征之上训练一个线性分类器；用于衡量特征质量 |
| NaFlex | 「原生灵活分辨率」 | SigLIP 2 的能力：无需缩放即可摄入任意宽高比和分辨率的图像 |
| Temperature scaling | 「对数参数化 tau」 | CLIP 参数化 `log(1/tau)` 以让梯度表现正常；裁剪以防止 tau 坍缩到接近零 |

## 延伸阅读

- [Radford 等——Learning Transferable Visual Models From Natural Language Supervision（arXiv:2103.00020）](https://arxiv.org/abs/2103.00020)——CLIP 论文。
- [Zhai 等——Sigmoid Loss for Language Image Pre-Training（arXiv:2303.15343）](https://arxiv.org/abs/2303.15343)——SigLIP。
- [Tschannen 等——SigLIP 2（arXiv:2502.14786）](https://arxiv.org/abs/2502.14786)——多语言 + NaFlex。
- [Jia 等——ALIGN（arXiv:2102.05918）](https://arxiv.org/abs/2102.05918)——用噪声网络数据做规模。
- [Cherti 等——Reproducible scaling laws for contrastive language-image learning（arXiv:2212.07143）](https://arxiv.org/abs/2212.07143)——OpenCLIP 扩展规律。
