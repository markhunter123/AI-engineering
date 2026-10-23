# GloVe、FastText 与子词嵌入

> Word2Vec 为每个词训练一个嵌入。GloVe 分解了共现矩阵。FastText 嵌入了片段。BPE 架起了通往 Transformer 的桥梁。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec from Scratch)
**Time:** ~45 minutes

## 问题

Word2Vec 留下了两个开放问题。

第一，当时另有一条研究线直接分解共现矩阵（LSA、HAL），而不是做在线 skip-gram 更新。Word2Vec 的迭代方法本质更好，还是差异只是两者处理计数方式不同造成的？**GloVe** 回答了这个问题：用精心选择的损失函数做矩阵分解，可以匹配或超越 Word2Vec，且训练成本更低。

第二，这两种方法都对从未见过的词束手无策。`Zoomer-approved`、`dogecoin`、上周新造的任何专有名词、罕见词根的每种屈折变化。**FastText** 通过嵌入字符 n-gram 解决了这个问题：一个词是其片段（包括词素）之和，因此即使词表外词也能得到合理的向量。

第三，Transformer 到来后问题再次转移。词级词汇表上限约一百万，而真实语言比这更开放。**字节对编码（Byte-Pair Encoding，BPE）** 及其变体通过学习频繁的子词单元解决了这个问题，覆盖所有文本。每个现代 LLM 的 tokenizer 都是子词 tokenizer。

本节课依次走过这三种方法，然后解释何时该选哪一种。

## 概念

**GloVe（Global Vectors）。** 构建词-词共现矩阵 `X`，其中 `X[i][j]` 表示词 `j` 在词 `i` 的上下文中出现多少次。训练向量使得 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。给损失加权，避免高频词对主导。搞定。

**FastText。** 一个词是其字符 n-gram 加上词本身的和。`where` 变成 `<wh, whe, her, ere, re>, <where>`。词向量是这些组成向量的和。像 Word2Vec 一样训练。好处是未见过的词（`whereupon`）可以由已知的 n-gram 组合出来。

**BPE（Byte-Pair Encoding）。** 从单个字节（或字符）的词汇表开始。统计语料中每对相邻符号，把最频繁的一对合并成一个新 token。重复 `k` 次。结果是一个 `k + 256` 大小的词汇表，其中频繁序列（`ing`、`tion`、`the`）是单个 token，罕见词被拆成熟悉的片段。任何句子都能被 tokenize。

```figure
n5-subword-merge
```

## 构建它

### GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

有两个值得指出的细节。加权函数 `f(x) = (x/x_max)^alpha` 压低极频繁词对（如 `(the, and)`）的损失，防止它们主导总损失。最终嵌入是中心表 `W` 和上下文表 `W_tilde` 之和。把两个表相加是文献中的技巧，通常比只用其中一个更好。

### FastText：子词感知嵌入

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词由其 n-gram 集合（通常 3 到 6 个字符）表示。词嵌入是其 n-gram 嵌入之和。对 skip-gram 训练，把原来 Word2Vec 用单个向量的地方换成这个。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对未见词，只要它的一些 n-gram 已知，仍然可以得到向量。`whereupon` 与 `where` 共享 `<wh`、`her`、`ere`、`<where`，因此两者在空间中靠近。

### BPE：学习子词词汇表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一次迭代合并最常见的相邻对。经过足够多次迭代，频繁子串（`low`、`est`、`tion`）变成单个 token，罕见词也会被干净地拆分。

真实的 GPT / BERT / T5 tokenizer 会学习 3 万-10 万次合并。结果：任何文本都能 tokenize 成已知 ID 的有限长度序列，永远没有 OOV。

## 使用它

实践中，你很少自己训练这些，而是加载预训练 checkpoint。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

Transformer 时代的 BPE 式子词 tokenization：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 前缀标记词边界（GPT-2 的约定）。每个现代 tokenizer 都是 BPE 变体、WordPiece（BERT）或 SentencePiece（T5、LLaMA）。

### 何时选哪个

| 场景 | 选择 |
|-----------|------|
| 预训练通用词向量，不需要容忍 OOV | GloVe 300d |
| 预训练通用词向量，必须处理拼写错误 / 新造词 / 形态丰富语言 | FastText |
| 任何进入 Transformer 的数据（训练或推理） | 用模型附带的 tokenizer。绝不能换。 |
| 从头训练自己的语言模型 | 先在语料上训练 BPE 或 SentencePiece tokenizer |
| 生产文本分类用线性模型 | 仍然用 TF-IDF。第 02 课。 |

## 交付它

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## 练习

1. **简单。** 运行 `char_ngrams("playing")` 和 `char_ngrams("played")`。计算两个 n-gram 集合的 Jaccard 重叠。你会看到大量共享片段（`pla`、`lay`、`play`），这正是 FastText 能跨形态变体迁移的原因。
2. **中等。** 扩展 `learn_bpe`，追踪词汇表增长。绘制「每个语料字符对应的 token 数」随合并次数变化的曲线。开始会快速压缩，最终渐近于每个 token 约 2-3 个字符。
3. **困难。** 在莎士比亚全集上训练一个 1k 次合并的 BPE。比较常见词与罕见专有名词的 tokenization。测量合并前后的平均每词 token 数。写下让你惊讶的地方。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| Co-occurrence matrix | 词-词频率表 | `X[i][j]` = 词 `j` 在词 `i` 窗口内出现的次数。 |
| Subword | 词的片段 | 字符 n-gram（FastText）或学习到的 token（BPE/WordPiece/SentencePiece）。 |
| BPE | 字节对编码 | 反复合并最频繁的相邻对，直到词汇表达到目标大小。 |
| OOV | 词表外 | 模型从未见过的词。Word2Vec/GloVe 无法处理；FastText 和 BPE 可以。 |
| Byte-level BPE | 原始字节上的 BPE | GPT-2 的方案。词汇表从 256 个字节开始，因此永远没有 OOV。 |

## 延伸阅读

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) —— GloVe 论文，七页，仍是损失函数推导的最佳来源。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) —— FastText。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) —— 把 BPE 引入现代 NLP 的论文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) —— BPE、WordPiece 和 SentencePiece 在实践中究竟有何不同。
