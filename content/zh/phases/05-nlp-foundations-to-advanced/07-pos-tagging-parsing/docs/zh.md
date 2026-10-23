# 词性标注与句法分析

> 语法曾一度不受欢迎。后来每个 LLM pipeline 都需要验证结构化抽取，它又回来了。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## 问题

第 01 课承诺，lemmatization 需要一个词性标签（part-of-speech tag）。如果不知道 `running` 是动词，lemmatizer 就无法把它还原成 `run`；如果不知道 `better` 是形容词，也无法还原成 `good`。

这个承诺背后隐藏着一整个子领域。词性标注（POS tagging）为每个 token 分配语法类别。句法分析则还原句子的树状结构：哪个词修饰哪个词，哪个动词支配哪些论元。传统 NLP 花了二十年打磨这两者。后来深度学习把它们压缩成一个基于预训练 Transformer 的 token 分类任务，研究社区便转向了别处。

但应用社区没有。每个结构化抽取 pipeline 仍在底层使用 POS 和依存树。LLM 生成的 JSON 要对照语法约束来验证。问答系统用依存分析来分解查询。机器翻译质量评估器会检查分析树的对齐。

值得了解。本课介绍标签集、基线模型，以及那个你会停止手写实现、转而调用 spaCy 的临界点。

## 概念

**POS 标注**给每个 token 标注一个语法类别。英语默认使用 **Penn Treebank（PTB）** 标签集：36 个标签，包含普通读者觉得过于细分的区别，例如 `NN` 单数名词、`NNS` 复数名词、`NNP` 专有名词单数、`VBD` 动词过去式、`VBZ` 动词第三人称单数现在时，等等。**Universal Dependencies（UD）** 标签集更粗（17 个标签）且语言无关，已成为跨语言工作的默认选择。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法分析**生成一棵树。两大流派：

- **成分句法分析（Constituency parsing）。** 名词短语、动词短语、介词短语相互嵌套，输出是一棵非终结符类别（NP、VP、PP）组成的树，词作为叶子。
- **依存句法分析（Dependency parsing）。** 每个词都有一个它所依赖的中心词，并用语法关系标注。输出是一棵每个边都是（中心词，依赖词，关系）三元组的树。

依存分析在 2010 年代胜出，因为它跨语言泛化得更干净，尤其适用于自由语序语言。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

```figure
pos-tagger
```

```figure
dependency-arcs
```

## 构建它

### 第 1 步：最高频标签基线

最笨拙但有效的 POS 标注器：对每个词，预测它在训练集中出现次数最多的标签。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown 语料库上，这个基线达到约 85% 的准确率。不算好，但任何严肃模型都不应低于这个下限。

### 第 2 步：二元 HMM 标注器

建模序列的联合概率：

```math
P(\text{tags}, \text{words}) = \prod_i P(\text{tag}_i \mid \text{tag}_{i-1}) \cdot P(\text{word}_i \mid \text{tag}_i)
```

两个表：转移概率（给定前一个标签的标签概率）和发射概率（给定标签的词概率）。用拉普拉斯平滑从计数中估计两者，再用 Viterbi（在标签格子上做动态规划）解码。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Brown 语料库上的二元 HMM 达到约 93% 准确率。从 85% 到 93% 的提升主要来自转移概率——模型学会了 `DET NOUN` 很常见，而 `NOUN DET` 很罕见。

### 第 3 步：为什么现代标注器能做得更好

转移概率和发射概率都是局部的，无法捕捉 `saw` 在 "I bought a saw" 中是名词、在 "I saw the movie" 中是动词的区别。引入任意特征（后缀、词形、前后词、词本身）的 CRF 可达约 97%；BiLSTM-CRF 或 Transformer 可达 98% 以上。

这项任务的天花板由标注者分歧决定。人类标注者在 Penn Treebank 上的一致性约为 97%；超过 98% 的模型很可能在过拟合测试集。

### 第 4 步：依存分析概览

完整的依存分析从零实现超出本课范围；标准教材处理见 Jurafsky and Martin。需要了解的两个经典流派：

- **基于转移（Transition-based）** 分析器（arc-eager、arc-standard）像移进-归约分析器一样工作：读入 token，把它们移入栈，再应用创建弧的归约动作。贪心解码很快。经典实现是 MaltParser。现代神经版本：Chen and Manning 的基于转移分析器。
- **基于图（Graph-based）** 分析器（Eisner 算法、Dozat-Manning biaffine）为每个可能的中心-依赖边打分，然后选择最大生成树。更慢但更准确。

对大多数应用工作，直接调用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下往上读 `dep` 列，句子的语法结构就显而易见了。

## 使用它

每个生产级 NLP 库都把 POS 和依存分析作为标准 pipeline 的一部分。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）：快速、准确，与 tokenization + NER + lemmatization 集成。`token.tag_`（Penn）、`token.pos_`（UD）、`token.dep_`（依存关系）。
- **Stanford NLP（stanza）**：Stanford CoreNLP 的继任者，在 60 多种语言上达到最先进水平。
- **trankit**：基于 Transformer，UD 准确率很好。
- **NLTK**：`pos_tag`。可用、较慢、较老。适合教学。

### 2026 年这仍然重要的场景

- **Lemmatization。** 第 01 课需要 POS 才能正确还原词形。永远如此。
- **从 LLM 输出中做结构化抽取。** 验证生成的句子是否满足语法约束（例如主谓一致、必需的修饰语）。
- **基于方面的情感分析。** 依存分析能告诉你哪个形容词修饰哪个名词。
- **查询理解。** "movies directed by Wes Anderson starring Bill Murray" 可以通过分析分解成结构化约束。
- **跨语言迁移。** UD 标签和依存关系是语言无关的，可对新语言做零样本结构化分析。
- **低算力 pipeline。** 如果无法部署 Transformer，POS + 依存分析 + 地名辞典能让你走得很远。

## 交付它

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## 练习

1. **简单。** 使用最高频标签基线，在一个小型标注语料库（例如 NLTK 的 Brown 子集）上测试，在留出句子上测量准确率。验证约 85% 的结果。
2. **中等。** 训练上面的二元 HMM，报告每个标签的精确率 / 召回率。HMM 最容易混淆哪些标签？
3. **困难。** 用 spaCy 的依存分析从 1000 句样本中提取主谓宾三元组，在 50 条人工标注三元组上评估。记录提取失败的场景（常为被动语态、并列结构和省略主语）。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| POS tag | 词的类型 | 语法类别。PTB 有 36 个；UD 有 17 个。 |
| Penn Treebank | 标准标签集 | 英语专用。细分动词时态和名词数。 |
| Universal Dependencies | 多语言标签集 | 比 PTB 更粗；语言中立；跨语言工作的默认选择。 |
| Dependency parse | 句子树 | 每个词有一个中心，每条边有一个语法关系。 |
| Viterbi | 动态规划 | 给定发射和转移概率，找出概率最高的标签序列。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — 词性标注与句法分析的标准教材处理。
- [Universal Dependencies project](https://universaldependencies.org/) — 跨语言标签集和树库集合，每个多语言分析器都在使用。
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — `Token` 上每个属性的实用参考。
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 把神经分析器带入主流的论文。
