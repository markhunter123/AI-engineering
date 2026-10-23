# 序列到序列模型

> 两个 RNN 假装自己是翻译器。它们遇到的瓶颈正是注意力存在的原因。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 08 (CNNs + RNNs for Text), Phase 3 · 11 (PyTorch Intro)
**Time:** ~75 minutes

## 问题

分类把变长序列映射到单个标签；翻译把变长序列映射到另一个变长序列。输入和输出使用不同的词汇表，可能是不同的语言，长度也未必对应。

seq2seq 架构（Sutskever、Vinyals、Le，2014）用一个刻意简单的配方破解了它：两个 RNN。一个读取源句子并生成一个固定大小的上下文向量；另一个读取该向量并逐个生成目标句子 token。代码和第 08 课写的一样，只是以不同方式粘合在一起。

学习它有两个原因。首先，上下文向量瓶颈是 NLP 中最具教学价值的失败案例，它解释了注意力和 Transformer 擅长的一切。其次，训练配方（teacher forcing、scheduled sampling、推理时的 beam search）仍然适用于包括 LLM 在内的每个现代生成系统。

## 概念

**编码器（Encoder）。** 读取源句子的 RNN。它的最终隐藏状态就是**上下文向量（context vector）**—— supposedly 对整个输入的固定大小摘要。

**解码器（Decoder）。** 另一个从上下文向量初始化的 RNN。每一步它把上一个生成的 token 作为输入，输出目标词汇上的分布。采样或取 argmax 得到下一个 token，再把它喂回去。重复直到生成 `<EOS>` 或达到最大长度。

**训练。** 每个解码器步骤都计算交叉熵损失，并对序列求和。通过两个网络做标准的反向传播 through time。

**Teacher forcing。** 训练时，解码器在步骤 `t` 的输入是位置 `t-1` 的*真实* token，而不是解码器自己上一步的预测。这稳定了训练；没有它，早期错误会级联，模型永远学不会。推理时则必须使用模型自己的预测，因此训练/推理之间永远存在分布差距。这个差距称为**曝光偏差（exposure bias）**。

**瓶颈。** 编码器学到的关于源端的所有信息都必须挤进那一个上下文向量。长句丢失细节，罕见词被模糊，语序调整（chat noir vs. black cat）必须靠记忆而非计算。

注意力（第 10 课）通过让解码器查看*每一个*编码器隐藏状态来修复这个问题，而不仅是最后一个。这就是全部卖点。

```figure
lstm-gates
```

## 构建它

### 第 1 步：编码器

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 的形状是 `[batch, seq_len, hidden_dim]`——每个输入位置一个隐藏状态。`hidden` 的形状是 `[1, batch, hidden_dim]`——最后一步。第 08 课说“对 outputs 做池化用于分类”。这里我们把最后隐藏状态当作上下文向量，并忽略每步输出。

### 第 2 步：解码器

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器每次调用一步。输入：一个 batch 的单 token 和当前隐藏状态。输出：下一个 token 的词汇 logits 和更新后的隐藏状态。

### 第 3 步：带 teacher forcing 的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

两个值得指出的细节。`ignore_index=0` 跳过 padding token 的损失。`teacher_forcing_ratio` 是每一步使用真实 token 而非模型预测的概率。从 1.0（完全 teacher forcing）开始，训练过程中逐渐降到约 0.5，以缩小曝光偏差差距。

### 第 4 步：推理循环（贪心）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪心解码每一步都选概率最高的 token。它可能跑偏：一旦选定某个 token 就无法收回。**Beam search** 会保留 top-`k` 个部分序列，最后选得分最高的完整序列。beam width 3–5 是标准配置。

### 第 5 步：瓶颈演示

在玩具复制任务上训练模型：源端 `[a, b, c, d, e]`，目标 `[a, b, c, d, e]`。不断增加序列长度，观察准确率。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个 GRU 隐藏状态无法无损记住 40 个 token 的输入。信息在每个编码器步骤都存在，但解码器只能看到最后状态。注意力直接解决了这一点。

## 使用它

PyTorch 提供 `nn.Transformer` 和基于 `nn.LSTM` 的 seq2seq 模板。Hugging Face 的 `transformers` 库提供完整的编码器-解码器模型（BART、T5、mBART、NLLB），它们在数十亿 token 上训练过。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代编码器-解码器用 Transformer 替换了 RNN。高层结构（编码器、解码器、逐个生成 token）与 2014 年的 seq2seq 论文相同，只是每个块内部的机制不同。

### 什么时候仍会考虑基于 RNN 的 seq2seq

对新项目几乎从不。具体例外：

- 流式翻译：你一次消费一个 token，内存受限。
- 端侧文本生成：Transformer 的内存成本过高。
- 教学：理解编码器-解码器瓶颈是理解 Transformer 为何获胜的最快路径。

### 曝光偏差及其缓解

- **Scheduled sampling。** 训练过程中降低 teacher forcing 比例，让模型学会从自己的错误中恢复。
- **Minimum risk training。** 用句子级 BLEU 分数而非 token 级交叉熵训练。更接近真实目标。
- **强化学习微调。** 用某个指标奖励序列生成器。现代 LLM 的 RLHF 也使用它。

三者同样适用于基于 Transformer 的生成。

## 交付它

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 练习

1. **简单。** 实现玩具复制任务。在输入-输出对中目标等于源的 GRU seq2seq 上训练。测量长度 5、10、20 的准确率。复现瓶颈。
2. **中等。** 加入 beam width 3 的 beam search 解码。在一个小型平行语料库上测量与贪心解码相比的 BLEU。记录 beam search 在哪些地方胜出（通常是末尾 token）以及哪些地方没有区别。
3. **困难。** 在一个 1 万对的复述数据集上微调 `facebook/bart-base`。比较微调后模型与基础模型在保留输入上的 beam-4 输出。报告 BLEU 并挑选 10 个定性例子。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Encoder | 输入 RNN | 读取源端。输出每步隐藏状态和最终上下文向量。 |
| Decoder | 输出 RNN | 从上下文向量初始化。逐个生成目标 token。 |
| Context vector | 摘要 | 编码器最终隐藏状态。固定大小。注意力要解决的瓶颈。 |
| Teacher forcing | 使用真实 token | 训练时把真实的前一个 token 喂给解码器。稳定学习。 |
| Exposure bias | 训练/测试差距 | 模型在真实 token 上训练，从未练习过从自己的错误中恢复。 |
| Beam search | 更好的解码 | 每一步保留 top-k 个部分序列，而不是贪心提交。 |

## 延伸阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 原始 seq2seq 论文。四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — 引入 GRU 和编码器-解码器框架。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 注意力论文。本课后立即阅读。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 可上手的 seq2seq + 注意力代码。
