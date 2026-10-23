# 多语言 NLP

> 一个模型，100+ 种语言，其中大多数语言零训练数据。跨语言迁移是 2020 年代的实用奇迹。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 04 (GloVe, FastText, Subword), Phase 5 · 11 (Machine Translation)
**Time:** ~45 minutes

## 问题

英语有数十亿标注样本。乌尔都语有数千。迈蒂利语几乎为零。任何服务全球受众的实用 NLP 系统，都必须在缺乏任务特定训练数据的长尾语言上工作。

多语言模型通过同时在多种语言上训练一个模型来解决这个问题。共享表示让模型把在高资源语言中学到的能力迁移到低资源语言。用英语情感分析微调模型，它开箱就能在乌尔都语上给出出奇好的情感预测。这就是零样本跨语言迁移，它重塑了 NLP 向世界交付的方式。

本课点明权衡、标准模型，以及让初涉多语言工作的团队栽跟头的那一个决策：为迁移选择源语言。

## 概念

![通过共享多语言 embedding 空间的跨语言迁移](../assets/multilingual.svg)

**共享词表。** 多语言模型使用在所有目标语言文本上训练的 SentencePiece 或 WordPiece 分词器。词表是共享的：同一个子词单元在亲属语言中表示同一个语素。英语和意大利语里的 `anti-` 得到同一个 token。

**共享表示。** 在许多语言上以掩码语言建模预训练的 Transformer 会学到：不同语言中语义相似的句子产生相似的隐藏状态。mBERT、XLM-R 和 NLLB 都体现了这一点。英语中 "cat" 的 embedding 靠近法语 "chat" 和西班牙语 "gato"，整句 embedding 也是如此。

**零样本迁移。** 在一种语言（通常是英语）的标注数据上微调模型。推理时，在模型支持的任何其他语言上运行。不需要目标语言标签。对类型学上相近的语言效果好，对距离远的语言较弱。

**少样本微调。** 加入 100-500 个目标语言标注样本。在分类任务上，准确率跃升至英语基线的 95-98%。这是多语言 NLP 中性价比最高的杠杆。

## 模型

| 模型 | 年份 | 覆盖 | 备注 |
|-------|------|----------|-------|
| mBERT | 2018 | 104 种语言 | 在 Wikipedia 上训练。第一个实用的多语言 LM。低资源语言上较弱。 |
| XLM-R | 2019 | 100 种语言 | 在 CommonCrawl 上训练（比 Wikipedia 大得多）。确立了跨语言基线。Base 270M，Large 550M。 |
| XLM-V | 2023 | 100 种语言 | XLM-R 搭配 1M token 词表（对比 250k）。低资源语言上更好。 |
| mT5 | 2020 | 101 种语言 | 用于多语言生成的 T5 架构。 |
| NLLB-200 | 2022 | 200 种语言 | Meta 的翻译模型；包含 55 种低资源语言。 |
| BLOOM | 2022 | 46 种语言 + 13 种编程语言 | 多语言训练的开放 176B LLM。 |
| Aya-23 | 2024 | 23 种语言 | Cohere 的多语言 LLM。阿拉伯语、印地语、斯瓦希里语上很强。 |

按用例选择。分类任务用 XLM-R-base 作为稳妥默认。生成任务视翻译还是开放生成选择 mT5 或 NLLB。LLM 风格的工作搭配 Aya-23 或 Claude，使用显式的多语言提示。

## 源语言决策（2026 年研究）

大多数团队默认用英语作为微调源语言。近期研究（2026）表明这往往是错的。

语言相似度比原始语料规模更能预测迁移质量。对斯拉夫语目标，德语或俄语常常优于英语。对印度语系目标，印地语常常优于英语。**qWALS** 相似度指标（2026，基于 World Atlas of Language Structures 特征）对此进行量化。**LANGRANK**（Lin et al., ACL 2019）是另一种更早的方法，综合语言相似度、语料规模和发生学亲缘关系对候选源语言排序。

实用规则：如果你的目标语言有一个类型学上接近的高资源亲属，先在那个语言上微调试试，再与英语微调的结果比较。

```figure
n5-crosslingual-bridge
```

## 构建它

### 第 1 步：零样本跨语言分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，同一个 API。在 NLI 数据上训练的 XLM-R 通过蕴含技巧很好地迁移到分类。

### 第 2 步：多语言 embedding 空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译句在 embedding 空间中靠得很近。一句不同的英语则离得更远。这正是跨语言检索、聚类和相似度能够工作的原因。

### 第 3 步：少样本微调策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对 100-500 个目标语言样本，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全默认值。更高的学习率会导致多语言对齐崩塌，你得到的是一个只会英语的模型。

## 真正有效的评估

- **留出集上的逐语言准确率。** 不要只看聚合值。聚合会掩盖长尾。
- **对照单语基线做基准。** 对数据充足的语言，从头训练的单语模型有时能击败多语言模型。要测。
- **实体级测试。** 目标语言中的命名实体。多语言模型对远离拉丁字母的文字常常分词能力弱。
- **跨语言一致性。** 同一含义在两种语言中应产生相同预测。测量差距。

## 使用它

2026 年的技术栈：

| 任务 | 推荐 |
|-----|-------------|
| 分类，100 种语言 | 微调的 XLM-R-base（约 270M） |
| 零样本文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句向量 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻译，200 种语言 | `facebook/nllb-200-distilled-600M`（见第 11 课） |
| 生成式多语言 | Claude、GPT-4、Aya-23、mT5-XXL |
| 低资源语言 NLP | XLM-V，或在相关高资源语言上做领域特定微调 |

如果性能重要，始终为目标语言微调预留预算。零样本是起点，不是最终答案。

### Tokenization 税（低资源语言会出什么问题）

多语言模型在所有语言之间共享一个分词器。该词表在一个由英语、法语、西班牙语、中文、德语主导的语料上训练。对任何不在主导集合中的语言，三种税会悄悄叠加：

- **生育率税（Fertility tax）。** 低资源语言文本每个词被切成远多于英语的 token。一个印地语句子可能需要等价英语句子 3-5 倍的 token。这 3-5 倍吃掉你的上下文窗口、训练效率和延迟。
- **变体恢复税。** 每个拼写错误、变音符号变体、Unicode 归一化不匹配或大小写变化，在 embedding 空间中都会变成一个冷启动的无关序列。模型无法学到母语者视为显然的正字法对应关系。
- **容量溢出税。** 税 1 和税 2 消耗上下文位置、层深和 embedding 维度。留给真正推理的部分，系统性地少于同模型上的高资源语言。

实际症状：模型在印地语上正常训练，损失曲线看起来没问题，评估困惑度也合理，而生产输出却有微妙错误。词法在句子中途崩坏。稀有屈折变化无法恢复。**你无法靠堆数据走出一个破损的分词器。**

缓解措施：选择对目标语言覆盖良好的分词器（XLM-V 的 1M token 词表是直接修复）；训练前在留出的目标语言文本上验证分词生育率；对真正长尾的文字使用字节级回退（SentencePiece `byte_fallback=True`、GPT-2 风格的字节级 BPE），确保永不出现 OOV。

## 交付它

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 练习

1. **简单。** 在英语、法语、印地语和阿拉伯语各 10 个句子上运行零样本分类 pipeline。报告每种语言的准确率。你应该看到法语强、印地语尚可、阿拉伯语不稳定。
2. **中等。** 使用 `paraphrase-multilingual-MiniLM-L12-v2` 在一个小型混合语言语料库上构建跨语言检索器。用英语查询，检索任意语言的文档。测量 recall@5。
3. **困难。** 为一个印地语分类任务比较英语源和印地语源的微调。在两种方案下都用 500 个目标语言样本做少样本微调。报告哪种源得到更好的印地语准确率，以及差距多大。这是 LANGRANK 论点的缩影。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Multilingual model | 一个模型，多种语言 | 跨语言共享词表和参数。 |
| Cross-lingual transfer | 在一种语言上训练，在另一种上运行 | 在源语言上微调，在没有目标语言标签的目标语言上评估。 |
| Zero-shot | 无目标语言标签 | 不在目标语言上微调的直接迁移。 |
| Few-shot | 少量目标语言标签 | 用 100-500 个目标语言样本做微调。 |
| mBERT | 第一个多语言 LM | 在 Wikipedia 上预训练的 104 语言 BERT。 |
| XLM-R | 标准跨语言基线 | 在 CommonCrawl 上预训练的 100 语言 RoBERTa。 |
| NLLB | Meta 的 200 语言 MT | No Language Left Behind。包含 55 种低资源语言。 |

## 延伸阅读

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) —— XLM-R 论文。
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) —— 开启跨语言迁移研究线的分析论文。
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) —— NLLB-200 论文。
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) —— Aya，Cohere 的多语言 LLM。
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) —— qWALS / LANGRANK 源语言论文。
