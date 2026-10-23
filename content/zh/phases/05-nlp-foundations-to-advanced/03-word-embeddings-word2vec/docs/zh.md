# 词嵌入 —— 从零开始实现 Word2Vec

> 观其友，知其人。在这个想法上训练一个浅层网络，几何结构就会自然涌现。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 minutes

## 问题

TF-IDF 知道 `dog` 和 `puppy` 是两个不同的词，却不知道它们意思几乎相同。在 `dog` 上训练的分类器无法泛化到关于 `puppy` 的评论。你可以靠罗列同义词来掩盖，但这在罕见词、领域术语以及你没想到的每种语言上都会失败。

你想要一种表示：让 `dog` 和 `puppy` 在空间中靠近；让 `king - man + woman` 落在 `queen` 附近；让在 `dog` 上训练的模型把一部分信号免费迁移到 `puppy`。

Word2Vec 给了我们这个空间。两层神经网络、万亿 token 训练规模、2013 年发表。架构简单到近乎尴尬，结果却重塑了此后十年的 NLP。

## 概念

**分布假设（Distributional hypothesis）**（Firth，1957）："You shall know a word by the company it keeps." 如果两个词出现在相似的上下文中，它们可能意思相近。

Word2Vec 有两种实现，都利用了这一思想：

- **Skip-gram。** 给定中心词，预测周围词。`cat -> (the, sat, on)`，窗口大小为 2。
- **CBOW（continuous bag of words）。** 给定周围词，预测中心词。`(the, sat, on) -> cat`。

Skip-gram 训练更慢，但对罕见词效果更好，因此成为默认选择。

网络只有一个隐藏层，没有非线性。输入是词汇表上的 one-hot 向量，输出是词汇表上的 softmax。训练完成后扔掉输出层，隐藏层权重就是 embedding。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

技巧在于：对 10 万词做 softmax 慢得无法接受。Word2Vec 用 **negative sampling（负采样）** 把它变成二分类任务：预测"这个上下文词是否出现在该中心词附近"。每次训练对只采样少量负例（未共现的词），而不是在整个词汇表上算 softmax。

```figure
word-vector-arithmetic
```

## 构建它

### 第 1 步：从语料中生成训练对

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口内的每个（中心词，上下文词）对都是一个正训练样本。

### 第 2 步：embedding 表

两个矩阵。`W` 是中心词 embedding 表（要保留的那个）。`W'` 是上下文词表（通常丢弃，有时也会与 `W` 平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小的随机初始化。 vocab 10k、dim 100 是现实规模；教学用 50 vocab × 16 dim 就足以观察几何。

### 第 3 步：负采样目标

对每个正样本对 `(center, context)`，从词汇表中采样 `k` 个随机词作为负例。训练模型使得正例的点积 `W[center] · W'[context]` 高，负例的点积低。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

神奇公式：正例的逻辑损失（希望 sigmoid 接近 1）加上负例的逻辑损失（希望 sigmoid 接近 0）。梯度同时流向两个表。完整推导见原始论文；如果你想真正记住，建议用纸笔推一遍。

### 第 4 步：在玩具语料上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在大语料上经过足够多 epoch，共享上下文的词会有相似的中心 embedding。在玩具语料上效果很弱，但在数十亿 token 上效果显著。

### 第 5 步：类比技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的 300d Google News 向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。不是因为模型知道什么是皇室，而是因为向量 `(king - man)` 捕捉到了类似"皇室的"东西，把它加到 `woman` 上就会落在"皇室女性"区域附近。

## 使用它

从零写 Word2Vec 是为了教学。生产 NLP 用 `gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

真实工作中，你几乎不会自己训练 Word2Vec，而是下载预训练向量：

- **GloVe** —— Stanford 的共现矩阵分解方法。有 50d、100d、200d、300d checkpoint。通用覆盖好。第 04 课专门讲 GloVe。
- **fastText** —— Facebook 对 Word2Vec 的扩展，嵌入字符 n-gram。通过组合子词来处理词表外词。第 04 课。
- **Google News 上的预训练 Word2Vec** —— 300d、300 万词词汇表，2013 年发布，至今每天仍被下载。

### Word2Vec 在 2026 年仍然占优的场景

- 轻量级领域特定检索。在笔记本电脑上花一小时内用医学摘要训练，得到通用模型捕捉不到的专业向量。
- 类比式特征工程。`gender_vector = mean(man - woman pairs)`，把它从其他词上减去，得到性别中立轴。公平性研究中仍在使用。
- 可解释性。100 维小到可以通过 PCA 或 t-SNE 画图，真正看到聚类形成。
- 任何必须在无 GPU 设备上本地推理的地方。Word2Vec 查表只是一次行读取。

### Word2Vec 失败的地方

多义词墙。`bank` 只有一个向量，`river bank` 和 `financial bank` 共享它；`table`（电子表格 vs 家具）也共享。下游分类器无法从向量中区分不同语义。

 contextual embedding（ELMo、BERT 及之后的每个 Transformer）通过根据周围上下文为同一个词的每次出现生成不同向量解决了这个问题。这就是从 Word2Vec 到 BERT 的跃迁：从静态到上下文相关。Transformer 部分在第 7 阶段深入讲解。

另一个失败是词表外（OOV）问题。`Zoomer-approved` 如果不在训练数据里，Word2Vec 就从未见过它，没有回退机制。fastText 通过子词组合修复这一点（第 04 课）。

## 交付它

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## 练习

1. **简单。** 在一个极小的语料（20 句关于猫和狗的句子）上跑训练循环。200 个 epoch 后，验证 `nearest(vocab, W, W[vocab["cat"]])` 的 top 3 中是否包含 `dog`。如果没有，增加 epoch 或扩充词汇表。
2. **中等。** 加入高频词子采样。频率高于 `10^-5` 的词按与其频率成正比的概率从训练对中丢弃。测量对罕见词相似度的影响。
3. **困难。** 在 20 Newsgroups 语料上训练一个模型。计算两条偏见轴：`he - she` 和 `doctor - nurse`。把职业词投影到两条轴上，报告哪些职业的偏见差距最大。这是公平性研究者常用的探测方法。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| Word embedding | 词作为向量 | 从上下文中学到的密集低维（通常 100-300 维）表示。 |
| Skip-gram | Word2Vec 技巧 | 从中心词预测上下文词。比 CBOW 慢，但对罕见词更好。 |
| Negative sampling | 训练捷径 | 用与 `k` 个随机词的二分类替代全词汇表 softmax。 |
| Static embedding | 一词一向量 | 不论上下文如何都相同。多义词上失败。 |
| Contextual embedding | 上下文敏感向量 | 根据周围词为每次出现生成不同向量。Transformer 的输出。 |
| OOV | 词表外 | 训练时未出现过的词。Word2Vec 无法为它们生成向量。 |

## 延伸阅读

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) —— 负采样论文，简短易读。
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) —— 如果原始论文的数学太密，这是最清晰的梯度推导。
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) —— 真正有效的生产训练设置。
