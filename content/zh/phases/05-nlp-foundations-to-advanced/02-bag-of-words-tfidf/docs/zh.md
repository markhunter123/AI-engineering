# 词袋、TF-IDF 与文本表示

> 先计数，再思考。在 2026 年，定义明确的任务上 TF-IDF 仍然能打败 embedding。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 02 (Linear Regression from Scratch)
**Time:** ~75 minutes

## 问题

模型需要数字。你手里只有字符串。

每个 NLP 管道都要回答同一个问题：如何把变长的 token 流变成分类器能消费的定长向量。学界最早想到的答案是最蠢但有效的那个：数词，造向量。

这个向量支撑过的生产 NLP 比任何 embedding 模型都多。垃圾邮件过滤器、主题分类器、日志异常检测、搜索排序（BM25 之前）、第一波情感分析、前二十年的学术 NLP 基准。2026 年的从业者在狭窄分类任务上仍然会先想到它。它快、可解释，而且在「词是否出现」就是关键信号的任务上，往往与 4 亿参数的 embedding 模型难分高下。

本节课从零构建词袋，再构建 TF-IDF，然后展示 scikit-learn 用三行代码做同样的事，最后指出那个会让你转向 embedding 的失效模式。

## 概念

**词袋（Bag of Words，BoW）** 丢弃顺序。对每篇文档，统计每个词汇词出现的次数。向量长度等于词汇表大小，位置 `i` 是词 `i` 的出现次数。

**TF-IDF** 对 BoW 重新加权。每个文档都出现的词信息量低，要压低；在语料中罕见、但在某篇文档中频繁的词是信号，要抬高。

```math
\begin{aligned}
\text{TF-IDF}(w, d) &= \text{TF}(w, d) \cdot \text{IDF}(w) \\
&= \frac{\mathop{\mathrm{count}}(w \text{ in } d)}{|d|} \cdot \log\frac{N}{\mathop{\mathrm{df}}(w)}
\end{aligned}
```

其中 `TF` 是词在文档中的频率，`df` 是文档频率（包含该词的文档数），`N` 是总文档数。`log` 让 ubiquitous 词的权重有界。

关键性质：两者都产生稀疏向量，且每个轴都可解释。你可以查看训练好的分类器权重，读出哪些词把文档推向哪个类别。768 维的 BERT embedding 做不到这一点。

```figure
bow-tfidf
```

## 构建它

### 第 1 步：构建词汇表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：分词后的文档列表（任何词级分词器都可以；本课 `code/main.py` 使用简化的小写版本）。输出：`{word: index}` 字典。稳定的插入顺序意味着索引 0 是第一篇文档里遇到的第一个词。约定各有不同；scikit-learn 会按字母排序。

### 第 2 步：词袋

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行是文档，列是词汇索引。`[i][j]` 表示「词 `j` 在文档 `i` 中出现多少次」。文档 1 有 `cat` 两次，因为它确实出现了两次；文档 0 没有 `ran`，因为它确实没出现。

### 第 3 步：词频与文档频

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

有两个平滑技巧值得一提。`(n+1)/(d+1)` 避免 `log(x/0)`。末尾 `+1` 保证每个文档都出现的词 IDF 为 1 而不是 0，与 scikit-learn 的默认行为一致。其他实现用原始 `log(N/df)`，两者都能工作，平滑版更友好。

### 第 4 步：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三篇文档，五个词汇词（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 在三篇里都出现，所以 IDF 低；`dog` 只在一篇里出现，所以 IDF 高。向量稀疏（大多数项很小），有区分力的词被放大。

### 第 5 步：L2 归一化行

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

不做归一化，长文档会得到更大的向量，在相似度计算中占主导。L2 归一化把每篇文档放到单位超球面上，行之间的余弦相似度就退化成了点积。

## 使用它

scikit-learn 提供生产级实现。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 一次调用完成分词、构建词汇表和 BoW。`TfidfVectorizer` 再加上 IDF 加权和 L2 归一化。两者都返回稀疏矩阵。10 万篇文档时稠密版放不进内存；在分类器要求稠密之前一直保持稀疏。

几个能改变一切的参数：

| 参数 | 效果 |
|-----|--------|
| `ngram_range=(1, 2)` | 包含二元组。通常能提升分类效果。 |
| `min_df=2` | 去掉出现在少于 2 篇文档中的词。噪声数据上可裁剪词汇表。 |
| `max_df=0.95` | 去掉出现在超过 95% 文档中的词。近似停用词移除，却不用硬编码列表。 |
| `stop_words="english"` | scikit-learn 内置停用词表。是否使用取决于任务 —— 情感分析*不应该*去掉否定词。 |
| `sublinear_tf=True` | 用 `1 + log(tf)` 替代原始 `tf`。某个词在一篇文档中反复出现时很有帮助。 |

### TF-IDF 仍然占优的场景（截至 2026 年）

- 垃圾邮件检测、主题标注、日志异常标记。词是否出现才是关键；语义细微差别不重要。
- 低数据场景（只有几百条标注样本）。TF-IDF 加 logistic 回归没有预训练成本。
- 任何延迟敏感的地方。TF-IDF 加线性模型以微秒级响应；用 Transformer 把文档 embedding 一遍要 10-100 ms。
- 需要解释预测的系统。检查分类器系数，排名靠前的正向词就是理由。

### TF-IDF 失败的地方

语义失明。看这两句：

- "The movie was not good at all."
- "The movie was excellent."

一句负面，一句正面。它们的 TF-IDF 重叠恰好是 `{the, movie, was}`。词袋分类器必须记住 `not` 出现在 `good` 附近会翻转标签。数据够多时它能学会，但永远不会像理解句法的模型那样优雅。

另一个失败：推理时的词表外词。在 IMDb 评论上训练的 BoW 模型不知道该怎么处理 `Zoomer-approved`，如果训练里没见过这个词。子词 embedding（第 04 课）能处理；TF-IDF 不能。

### 混合：TF-IDF 加权 embedding

2026 年中数据量分类的务实默认：用 TF-IDF 权重作为对词 embedding 的注意力。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你从 embedding 获得语义能力，从 TF-IDF 获得罕见词强调。分类器在池化后的向量上训练。在情感、主题、意图分类等任务上，当标注样本低于约 5 万时，混合模型往往优于两者各自单独使用。

## 交付它

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## 练习

1. **简单。** 在 L2 归一化后的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档得分 1.0，词汇完全不交的文档得分 0.0。
2. **中等。** 给 `bag_of_words` 增加 n-gram 支持。参数 `n` 表示统计 `n`-gram 的出现次数。验证在 `["the", "cat", "sat"]` 上 `n=2` 会产出 `["the cat", "cat sat"]` 的二元组计数。
3. **困难。** 用 GloVe 100d 向量（下载一次并缓存）构建上面的 TF-IDF 加权 embedding 混合模型。在 20 Newsgroups 数据集上比较它与纯 TF-IDF 和纯平均池化 embedding 的分类准确率。报告各自在什么情况下获胜。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| BoW | 词频向量 | 一篇文档中词汇词的出现次数。丢弃顺序。 |
| TF | 词频 | 一个词在文档中的出现次数，可选按文档长度归一化。 |
| DF | 文档频率 | 至少包含该词一次的文档数。 |
| IDF | 逆文档频率 | 平滑后的 `log(N / df)`。压低到处都出现的词。 |
| 稀疏向量 | 大部分为零 | 词汇表通常 1 万-10 万词；任何一篇文档只包含其中一小部分。 |
| 余弦相似度 | 向量夹角 | L2 归一化后的点积。1 表示相同，0 表示正交。 |

## 延伸阅读

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) —— 权威 API 参考，附带每个参数的说明。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) —— 让 TF-IDF 在十年间成为默认方法的论文。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) —— 2026 年关于旧方法何时获胜以及为何获胜的文章。
