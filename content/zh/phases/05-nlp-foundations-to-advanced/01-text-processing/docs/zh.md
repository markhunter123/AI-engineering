# 文本处理 —— 分词、词干提取、词形还原

> 语言是连续的，模型是离散的，预处理是连接二者的桥梁。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## 问题

模型读不懂 "The cats were running."。它只读整数。

每个 NLP 系统一开始都要回答三个同样的问题：单词从哪里开始；单词的词根是什么；什么时候该把 "run"、"running"、"ran" 当成同一件事，什么时候又该把它们区分开。

分词错了，模型就会从垃圾里学习。如果你的 tokenizer 把 `don't` 当成一个 token，却把 `do n't` 当成两个，训练分布就会分裂。如果你的 stemmer 把 `organization` 和 `organ` 压缩成同一个词干，主题建模就会垮掉。如果你的 lemmatizer 需要词性上下文而你没传，动词就会被当成名词处理。

本节课从零构建三个预处理步骤，然后展示 NLTK 和 spaCy 如何做同样的工作，让你看清其中的取舍。

## 概念

三项操作，各有职责，各有失效模式。

**分词（Tokenization）** 把字符串拆成 token。"Token" 故意保持模糊，因为合适的粒度取决于任务：经典 NLP 用词级别，Transformer 用子词级别，没有空格的语言用字符级别。

**词干提取（Stemming）** 按规则砍掉后缀。快、激进、但傻。`running -> run`。`organization -> organ`。后者就是它的失效模式。

**词形还原（Lemmatization）** 用语法知识把单词还原成词典形式。更慢、更准，需要查表或形态分析器。`ran -> run`（需要知道 "ran" 是 "run" 的过去式）。`better -> good`（需要知道比较级形式）。

经验法则：速度优先且能容忍噪声时用词干提取（搜索索引、粗略分类）；语义优先时用词形还原（问答、语义搜索、任何用户会读到的文本）。

```figure
edit-distance
```

## 构建它

### 第 1 步：正则表达式分词器

最简单实用的分词器按非字母数字字符拆分，同时把标点符号单独保留成 token。不完美，也不终极，但一行就能跑。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三个模式按优先级排列：可带内部撇号的单词（`don't`、`it's`）、纯数字、任何非空白非字母数字的单个字符作为独立 token（标点）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

值得注意的失效模式。`3pm` 被拆成 `['3', 'pm']`，因为我们在字母串和数字串之间做了交替。对大多数任务来说够用了。URL、邮箱、话题标签都会崩。生产环境要在通用模式前补充专门模式。

### 第 2 步：Porter 词干提取器（仅 step 1a）

完整 Porter 算法有五阶段规则。仅 step 1a 就覆盖了最高频的英语后缀，足以展示其模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

规则自上而下读取。`ies -> i` 这条规则让 `ponies -> poni` 而不是 `pony`。真正的 Porter 有 step 1b 会修正它。规则之间存在竞争，前面的规则优先。顺序比任何单条规则都重要。

### 第 3 步：基于查表的词形还原器

真正的词形还原需要形态学知识。一个可用于教学的简化版使用小型 lemma 表加回退。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一种是关键教学点。`watched` 不在我们的表中，而且回退只处理 `ing`。真正的词形还原要覆盖 `ed`、不规则动词、比较级形容词、发音变化的复数（`children -> child`）。这就是为什么生产系统会用 WordNet、spaCy 的 morphologizer 或完整形态分析器。

### 第 4 步：把它们串成管道

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的环节是一个词性标注器（POS tagger）。Phase 5 · 07（词性标注）会构建一个。目前默认把所有词当成 `NOUN`，并承认这个限制。

## 使用它

NLTK 和 spaCy 都自带生产级实现。各自只需几行。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 能处理缩写、Unicode、以及你正则遗漏的边界情况。`PorterStemmer` 跑完五个阶段。`WordNetLemmatizer` 需要把 NLTK 的 Penn Treebank 标注 scheme 翻译成 WordNet 的缩写集。上面的翻译连接是大多数教程都会跳过但你必须自己接上的部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 把整个管道藏在 `nlp(text)` 后面。分词、词性标注、词形还原一次性跑完。大规模比 NLTK 快，开箱更准。代价是你不容易单独替换某个组件。

### 何时选哪个

| 场景 | 选择 |
|-----------|------|
| 教学、研究、需要替换组件 | NLTK |
| 生产、多语言、速度优先 | spaCy |
| Transformer 管道（反正你会用模型自己的 tokenizer） | 用 `tokenizers` / `transformers`，跳过经典预处理 |

### 没人提醒你的两个失效模式

大多数教程只教算法就结束。真实预处理管道里有两件事会咬人，而且几乎没人讲。

**可复现性漂移。** NLTK 和 spaCy 在不同版本之间的分词和词形还原行为会变。spaCy 2.x 可能产出 `['do', "n't"]`，3.x 可能产出 `["don't"]`。你的模型是在一种分布上训练的，推理时却跑在另一种分布上。准确率悄无声息地下降，没人知道为什么。要在 `requirements.txt` 里固定库版本。写一份预处理回归测试，冻结 20 个样本句子的预期分词结果。每次升级都跑一遍。

**训练 / 推理不匹配。** 训练时用了激进预处理（小写、去停用词、词干提取），部署时却直接喂原始用户输入，性能会暴跌。这是生产 NLP 中最常见的单一失败。如果你在训练时做了预处理，推理时必须运行完全相同的函数。把预处理作为函数打包进模型包，而不是让服务团队重写成 notebook 单元格。

## 交付它

一个可复用的 prompt，帮助工程师在不读三本教科书的情况下选出预处理策略。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## 练习

1. **简单。** 扩展 `tokenize`，让 URL 保持为单个 token。测试：`tokenize("Visit https://example.com today.")` 应该产出一个 URL token。
2. **中等。** 实现 Porter step 1b。如果一个单词包含元音且以 `ed` 或 `ing` 结尾，则去掉它。处理双辅音规则（`hopping -> hop`，而不是 `hopp`）。
3. **困难。** 构建一个词形还原器，用 WordNet 作为查表，WordNet 没有条目时回退到你的 Porter stemmer。在带标注的语料上测量准确率，与纯 WordNet 和纯 Porter 对比。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| Token | 一个词 | 模型消费的任何单位。可以是词、子词、字符或字节。 |
| Stem | 词根 | 基于规则去后缀后的结果。不一定是真实存在的词。 |
| Lemma | 词典形式 | 你会去查的那种形式。需要语法上下文才能正确计算。 |
| POS tag | 词性 | 如 NOUN、VERB、ADJ 之类的类别。准确词形还原需要它。 |
| Morphology | 词形规则 | 单词如何根据时态、数、格改变形式。词形还原依赖它。 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) —— 原始论文，五页，至今仍是最清晰的解释。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) —— 真实管道是如何连线的。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) —— 你还想不到的分词边界情况。
