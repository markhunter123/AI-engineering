# 用于文本的 CNN 与 RNN

> 卷积学习 n-gram。循环网络记忆。两者都被注意力取代。两者在受限硬件上仍然重要。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 11 (PyTorch Intro), Phase 5 · 03 (Word Embeddings), Phase 4 · 02 (Convolutions from Scratch)
**Time:** ~75 minutes

## 问题

TF-IDF 和 Word2Vec 生成忽略词序的平面向量。基于它们的分类器无法区分 `dog bites man` 和 `man bites dog`。词序有时承载信号。

在 Transformer 出现之前，两类架构填补了这一空白。

**用于文本的卷积网络（TextCNN）。** 在词嵌入序列上应用 1D 卷积。宽度为 3 的滤波器就是一个可学习的 trigram 检测器：它覆盖三个词并输出一个分数。堆叠不同宽度（2、3、4、5）的滤波器以检测多尺度模式。通过最大池化得到固定大小的表示。扁平、并行、快速。

**循环网络（RNN、LSTM、GRU）。** 逐个处理 token，维护一个向前传递信息的隐藏状态。序列化、带记忆、输入长度灵活。它们在 2014 到 2017 年间主导序列建模，随后注意力出现。

本课构建这两种模型，然后点出促使注意力诞生的失败。

## 概念

**TextCNN**（Kim, 2014）。token 先被嵌入。宽度为 `k` 的 1D 卷积在连续的 `k`-gram 嵌入上滑动一个滤波器，产生一个 feature map。在该 map 上做全局最大池化，取最强激活。把多个滤波器宽度的池化输出拼接起来，再送入分类头。

它为什么有效。一个滤波器就是一个可学习的 n-gram。最大池化是位置不变的，因此 "not good" 无论在评论开头还是中间都会触发同一个特征。三个宽度各 100 个滤波器，就得到 300 个学习好的 n-gram 检测器。训练是并行的，没有序列依赖。

**RNN。** 在每个时间步 `t`，隐藏状态 `h_t = f(W * x_t + U * h_{t-1} + b)`。跨时间共享 `W`、`U`、`b`。最后时刻 `T` 的隐藏状态是整个前缀的摘要。用于分类时，对 `h_1 ... h_T` 做池化（最大、平均或取最后）。

普通 RNN 存在梯度消失问题。**LSTM** 增加了输入门、遗忘门和输出门以及一个细胞状态，能在长序列上稳定梯度。**GRU** 把 LSTM 简化为两个门；参数更少，表现相近。

**双向 RNN** 同时运行一个前向和一个后向 RNN，并把隐藏状态拼接起来。每个 token 的表示都能看到左右两侧的上下文。对标注任务至关重要。

```figure
rnn-unroll
```

## 构建它

### 第 1 步：PyTorch 中的 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 把 `[batch, seq_len, embed_dim]` 重塑为 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 把中间轴当作通道。池化后的输出是固定大小，与输入长度无关。

### 第 2 步：LSTM 分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

对序列做最大池化，而不是只取最后时刻。用于分类时，最大池化通常优于取最后的隐藏状态，因为长序列末尾的信息往往会主导最后状态。

### 第 3 步：梯度消失演示（直观解释）

没有门控的普通 RNN 无法学习长程依赖。考虑一个玩具任务：判断序列中是否出现过 token `A`。如果 `A` 在位置 1，而序列长度为 100，那么损失梯度必须反向流过 99 次循环权重的连乘。如果权重小于 1，梯度消失；如果大于 1，梯度爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM 通过一条**细胞状态（cell state）** 来解决这个问题：它在网络中只进行加法交互（遗忘门会乘法缩放，但梯度仍能沿“高速公路”流动）。GRU 用更少的参数实现了类似效果。两者都能让 100 步以上的序列稳定训练。

### 第 4 步：为什么这仍然不够

即便有 LSTM，三个问题依然存在。

1. **序列瓶颈。** 在长度为 1000 的序列上训练 RNN 需要 1000 个串行的前向/反向步骤，无法跨时间并行。
2. **编码器-解码器中的固定大小上下文向量。** 解码器只能看到编码器的最终隐藏状态，整个输入被压缩进一个向量。长输入会丢失细节。第 09 课直接讨论这一点。
3. **远距离依赖准确率天花板。** LSTM 优于普通 RNN，但仍难以把特定信息传播过 200 步以上。

注意力解决了全部三个问题。Transformer 则彻底抛弃了循环。第 10 课是转折点。

## 使用它

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 都已可用于生产。训练代码是标准写法。

Hugging Face 提供预训练嵌入，可以直接作为输入层：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

在约束合适时使用它们的清单。

- **边缘 / 端侧推理。** 搭配 GloVe 嵌入的 TextCNN 比 Transformer 小 10–100 倍。如果部署目标是手机，这就是你的技术栈。
- **流式 / 在线分类。** RNN 一次处理一个 token，而 Transformer 需要整条序列。对实时到来的文本，LSTM 仍然占优。
- **极小基线模型。** 在新任务上快速迭代。用 CPU 在 5 分钟内训练一个 TextCNN。
- **数据有限的序列标注。** BiLSTM-CRF（第 06 课）在 1k–10k 条标注句子的 NER 任务上仍是生产级架构。

其余情况都用 Transformer。

## 交付它

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 练习

1. **简单。** 在一个 3 类玩具数据集（自己构造数据）上训练 TextCNN。验证滤波器宽度 (2, 3, 4) 的平均 F1 优于单一宽度 (3)。
2. **中等。** 为 LSTM 分类器实现最大池化、平均池化和最后状态池化。在一个小数据集上比较；记录哪种池化胜出并假设原因。
3. **困难。** 构建一个 BiLSTM-CRF NER 标注器（结合第 06 课与本课）。在 CoNLL-2003 上训练。与第 06 课的纯 CRF 基线以及 BERT 微调比较。报告训练时间、内存和 F1。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| TextCNN | 文本 CNN | 在词嵌入上做 1D 卷积堆叠，再接全局最大池化。Kim (2014)。 |
| RNN | 循环网络 | 每个时间步更新隐藏状态：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | 门控 RNN | 增加输入 / 遗忘 / 输出门和细胞状态。能在长序列上稳定训练。 |
| GRU | 简化版 LSTM | 两个门而非三个。准确率相近，参数更少。 |
| Bidirectional | 双向 | 前向 + 后向 RNN 拼接。每个 token 都能看到上下文两侧。 |
| Vanishing gradient | 训练信号消失 | 普通 RNN 中反复乘以小于 1 的权重，使早期时间步的梯度几乎为零。 |

## 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 论文。八页。可读性强。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 论文。出乎意料地清晰。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 让 LSTM 变得人人可懂的图解。
