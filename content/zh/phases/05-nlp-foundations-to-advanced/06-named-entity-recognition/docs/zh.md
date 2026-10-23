# 命名实体识别

> 把名字抽出来。听起来很简单，直到你要处理模糊的边界、嵌套实体和领域术语。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word Embeddings)
**Time:** ~75 minutes

## 问题

「Apple 因 iPhone 搜索协议在美国起诉 Google。」这里有五个实体：Apple（ORG）、Google（ORG）、iPhone（PRODUCT）、search deal（协议，可能算一个）、US（GPE）。好的 NER 系统能把它们全部抽出并标对类型。差的系统会漏掉 iPhone，把水果 Apple 和公司 Apple 搞混，还把「US」标成 PERSON。

NER 是每个结构化抽取 pipeline 底下的苦力。简历解析、合规日志扫描、病历脱敏、搜索查询理解、聊天机器人回答的 grounding、法律合同抽取。你几乎从来看不见它，却始终依赖它。

本课沿着经典路线（规则、HMM、CRF）走到现代路线（BiLSTM-CRF，再到 Transformer）。每一步都在解决前一步的具体局限。这个演进模式本身就是本课的重点。

## 概念

**BIO 标注**（或 BILOU）把实体抽取变成序列标注问题。每个 token 标上 `B-TYPE`（实体开头）、`I-TYPE`（实体内部）或 `O`（不属于任何实体）。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多 token 实体按链式标注：`New B-GPE`、`York I-GPE`、`City I-GPE`。理解 BIO 的模型可以抽取任意长度的 span。

架构的演进：

- **基于规则。** 正则 + 地名词典（gazetteer）查表。已知实体上精度高，新实体上零覆盖。
- **HMM。** 隐马尔可夫模型（Hidden Markov Model）。给定标签时 token 的发射概率，以及标签到标签的转移概率。用 Viterbi 解码。在标注数据上训练。
- **CRF。** 条件随机场（Conditional Random Field）。和 HMM 类似，但它是判别式的，因此可以混用任意特征（词形、大小写、相邻词）。2026 年在低资源部署中仍是经典的生产主力。
- **BiLSTM-CRF。** 用神经特征取代手工特征。LSTM 双向读句子，顶部的 CRF 层保证标签序列一致。
- **基于 Transformer。** 用 token 分类头微调 BERT。精度最高，算力消耗也最大。

```figure
ner-bio-tagging
```

## 构建它

### 步骤 1：BIO 标注辅助函数

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 步骤 2：手工特征

对经典（非神经网络）NER 来说，特征就是一切。有用的特征包括：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 返回 `xXxxxx`。`word_shape("USA-2024")` 返回 `XXX-dddd`。大小写模式对专有名词来说是强信号。

### 步骤 3：简单的规则 + 词典基线

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产环境的地名词典有数百万条目，从 Wikipedia 和 DBpedia 抓取而来。覆盖率很好。消歧（公司 `Apple` 还是水果）却很糟。这就是统计模型胜出的原因。

### 步骤 4：CRF 这一步（草图，非完整实现）

没有概率论基础时，用 50 行从零实现完整 CRF 并不能带来启发。改用 `sklearn-crfsuite`：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 分别是 L1 和 L2 正则化。`all_possible_transitions=True` 让模型学到非法序列（例如 `O` 之后的 `I-ORG`）不太可能出现；CRF 就是这样在你不写约束的情况下保证 BIO 一致性的。

### 步骤 5：BiLSTM-CRF 增加了什么

特征变成了学出来的。输入是 token embedding（GloVe 或 fastText）。LSTM 从左到右、从右到左各读一遍。拼接后的隐藏状态经过 CRF 输出层。CRF 仍然保证标签序列一致；LSTM 则用学到的特征取代手工特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

CRF 层可以用 `torchcrf.CRF`（pip install pytorch-crf）。相比手工特征 CRF，提升是可测量的，但没有你想象的那么大——除非你有数万条标注句子。

## 使用它

spaCy 开箱即用，自带生产级 NER。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意 `iPhone` 被标成了 `ORG` 而不是 `PRODUCT`——spaCy 的小模型对产品类实体覆盖较弱。大模型（`en_core_web_lg`）表现更好，Transformer 模型（`en_core_web_trf`）还要更好。

用 Hugging Face 做基于 BERT 的 NER：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 会把连续的 B-X、I-X token 合并成一个 span。不加这个参数，你拿到的是 token 级标签，需要自己合并。

### 基于 LLM 的 NER（2026 年的选项）

在许多领域，零样本和少样本 LLM NER 已经能与微调模型竞争；标注数据稀缺时，优势尤其明显。

- **零样本提示。** 给 LLM 一份实体类型列表和一个示例 schema，要求输出 JSON。开箱可用；在新领域上精度中等。
- **ZeroTuneBio 式提示。** 把任务拆成候选抽取 → 含义解释 → 判断 → 复查。多阶段 prompt（而非一次性）能显著提升生物医学 NER 的精度。同样的模式也适用于法律、金融和科学领域。
- **带 RAG 的动态提示。** 每次推理时，从一个小的标注种子集中检索最相似的标注样本，现场拼出少样本 prompt。在 2026 年的基准中，这比静态提示把 GPT-4 的生物医学 NER F1 提升了 11-12%。
- **按实体类型拆解。** 对长文档来说，一次调用抽取所有实体类型会随长度增长而损失召回。为每种实体类型单独跑一遍抽取。推理成本更高，精度也显著更高。这是临床记录和法律合同的标准做法。

2026 年的生产建议：在收集训练数据之前，先用 LLM 零样本基线打底。通常 F1 已经够好，根本不需要微调。

### 经典 NER 仍然占优的场景

即便有 LLM 可用，经典 NER 在以下情况仍然占优：

- 延迟预算低于 50ms。
- 你有数千条标注样本，且需要 98%+ 的 F1。
- 领域本体稳定，预训练 CRF 或 BiLSTM 迁移效果好。
- 监管要求使用本地部署、非生成式模型。

### 它会在哪里失灵

- **领域偏移。** 在 CoNLL 上训练的 NER 用在法律合同上，表现还不如地名词典。要在你的领域上微调。
- **嵌套实体。**「Bank of America Tower」同时是 ORG 和 FACILITY。标准 BIO 无法表示重叠的 span。你需要嵌套 NER（多遍抽取或基于 span 的模型）。
- **长实体。**「United States Federal Deposit Insurance Corporation」会被 token 级模型拆开。用 `aggregation_strategy` 或后处理。
- **稀疏类型。** 医学 NER 标签如 DRUG_BRAND、ADVERSE_EVENT、DOSE。通用模型毫无概念。那里的起点是 Scispacy 和 BioBERT。

## 交付它

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## 练习

1. **简单。** 实现 `bio_to_spans`（`spans_to_bio` 的逆操作），在 10 个句子上验证往返一致性。
2. **中等。** 在 CoNLL-2003 英文 NER 数据集上训练上面的 sklearn-crfsuite CRF。用 `seqeval` 报告各实体类型的 F1。典型结果约为 84 F1。
3. **困难。** 在领域特定的 NER 数据集（医学、法律或金融）上微调 `distilbert-base-cased`。与 spaCy 小模型对比。记录数据泄漏检查，并写下让你意外的地方。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| NER | 抽名字 | 给 token span 打上类型标签（PERSON、ORG、GPE、DATE……）。 |
| BIO | 标注方案 | `B-X` 表示开始，`I-X` 表示延续，`O` 表示外部。 |
| BILOU | 更好的 BIO | 增加 `L-X`（最后）、`U-X`（单个）以划清边界。 |
| CRF | 结构化分类器 | 建模标签之间的转移，而不只是发射概率。保证序列合法。 |
| 嵌套 NER | 重叠实体 | 一个 span 与它的子 span 是不同实体。BIO 无法表达。 |
| 实体级 F1 | 正确的 NER 指标 | 预测 span 必须与真实 span 完全匹配。token 级 F1 会高估精度。 |

## 延伸阅读

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)——BiLSTM-CRF 论文。经典之作。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)——引入了后来成为标准的 token 分类模式。
- [spaCy 语言特性——命名实体](https://spacy.io/usage/linguistic-features#named-entities)——关于 `Doc.ents` 和 `Span` 每个属性的实用参考。
- [seqeval](https://github.com/chakki-works/seqeval)——正确的指标库。永远用它。
