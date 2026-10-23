# 实体链接与消歧

> NER 找到了「Paris」。实体链接要决定：是法国巴黎？Paris Hilton？得州巴黎？还是（特洛伊王子）帕里斯？没有链接，你的知识图谱就会一直含糊不清。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 24 (Coreference Resolution)
**Time:** ~60 minutes

## 问题

一句英文写道：「Jordan beat the press.」你的 NER 把「Jordan」标为 PERSON。很好。但*哪一个* Jordan？

- Michael Jordan（篮球运动员）？
- Michael B. Jordan（演员）？
- Michael I. Jordan（伯克利机器学习教授——没错，这种混淆在 ML 论文里真实存在）？
- Jordan（约旦这个国家）？
- Jordan（希伯来语名字）？

实体链接（entity linking，EL）把每个提及（mention）解析到知识库中的一个唯一条目：Wikidata、Wikipedia、DBpedia 或你自己的领域知识库。它包含两个子任务：

1. **候选生成。** 给定「Jordan」，知识库中哪些条目是合理的候选？
2. **消歧。** 结合上下文，哪个候选才是正确的？

两步都可以学习，也都有基准测试。整个 pipeline 十年来结构稳定——变化的是消歧器的质量。

## 概念

![实体链接流程：提及 → 候选 → 消歧后的实体](../assets/entity-linking.svg)

**候选生成。** 给定提及的表面形式（surface form，如「Jordan」），在别名索引中查找候选。Wikipedia 的别名词典覆盖了大多数命名实体：「JFK」→ John F. Kennedy、Jacqueline Kennedy、JFK 机场、电影《JFK》。典型索引对每个提及返回 10-30 个候选。

**消歧：三种方法。**

1. **先验 + 上下文（Milne & Witten，2008）。** 计算 `P(entity | mention) × context-similarity(entity, text)`。效果好、速度快、无需训练。
2. **基于 embedding（ESS / REL / Blink）。** 编码提及 + 上下文，再编码每个候选的描述，取余弦相似度最大者。这是 2020-2024 年的默认做法。
3. **生成式（GENRE，2021；基于 LLM，2023 之后）。** 逐 token 解码实体的规范名称，并用合法实体名构成的 trie 做约束解码，保证输出一定是合法的知识库 id。

**端到端 vs. pipeline。** 现代模型（ELQ、BLINK、ExtEnD、GENRE）在一次前向中完成 NER + 候选生成 + 消歧。生产中仍以 pipeline 系统为主，因为组件可以替换。

### 两个测量指标

- **提及召回率（候选生成阶段）。** 在标注正确的提及中，正确的知识库条目出现在候选列表里的比例。这是整个 pipeline 的下限。
- **消歧准确率 / F1。** 在候选正确的前提下，top-1 选对的频率。

两个指标必须同时报告。一个在 80% 候选召回上做到 99% 消歧的系统，整体 pipeline 只有 80%。

```figure
gx-entity-linking
```

## 构建它

### 步骤 1：从 Wikipedia 重定向构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia 的别名数据约有 1800 万对（alias, entity）。可从 Wikidata dumps 下载，用倒排索引存储。

### 步骤 2：基于上下文的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

这里的 Jaccard 重叠只是玩具版本。实际应替换为 embedding 上的余弦相似度（transformer 版本见 `code/main.py` 的 step-2）。

### 步骤 3：基于 embedding（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

建索引时，把每个知识库实体各编码一次。查询时，把提及 + 上下文编码一次，与候选池做点积，取最大值。

### 步骤 4：生成式实体链接（概念）

GENRE 逐字符解码实体的 Wikipedia 标题。约束解码（见第 20 课）保证只能输出合法标题，并与知识库支撑的 trie 紧密集成。现代的继承者是 REL-GEN，以及带结构化输出的 LLM 提示式实体链接。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

配合白名单（Outlines 的 `choice`），这是 2026 年最容易交付的实体链接 pipeline。

### 步骤 5：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是实体链接的标准基准：1,393 篇 Reuters 文章、3.4 万个提及、Wikipedia 实体。报告知识库内准确率（`P@1`）和知识库外 NIL 检测率。

## 常见陷阱

- **NIL 处理。** 有些提及不在知识库中（新出现的实体、冷门人物）。系统必须预测 NIL，而不是猜一个错误实体。这一项单独测量。
- **提及边界错误。** 上游 NER 漏掉部分跨度（把「Bank of America」只标成「Bank」），实体链接的召回率就会下降。
- **热门度偏差。** 训练过的系统会过度预测高频实体。ML 论文里的「Michael I. Jordan」常常被链接到篮球运动员 Jordan。
- **跨语言实体链接。** 把中文文本中的提及映射到英文 Wikipedia 实体，需要多语言编码器或一步翻译。
- **知识库过期。** 新公司、新事件、新人物不在去年的 Wikipedia dump 里。生产 pipeline 需要定期刷新循环。

## 使用它

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 通用英文 + Wikipedia | BLINK 或 REL |
| 跨语言，知识库为 Wikipedia | mGENRE |
| 适合 LLM，每天提及量少 | 用候选列表 + 约束 JSON 提示 Claude/GPT-4 |
| 领域知识库（医疗、法律） | 带知识库感知检索的自定义 BERT，并在领域 AIDA 风格数据集上微调 |
| 极低延迟 | 仅用精确匹配先验（Milne-Witten 基线） |
| 研究 SOTA | GENRE / ExtEnD / 生成式 LLM-EL |

2026 年落地的生产模式：NER → 共指消解 → 对每个提及做实体链接 → 把同一簇折叠成一个规范实体。输出是文档中每个实体一个知识库 id，而不是每个提及一个。

## 交付它

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 练习

1. **简单。** 在 `code/main.py` 中实现先验 + 上下文消歧器，跑 10 个有歧义的提及（Paris、Jordan、Apple），手工标注正确实体，测量准确率。
2. **中等。** 用 sentence transformer 编码 50 个有歧义的提及，并嵌入每个候选的描述。比较基于 embedding 的消歧与 Jaccard 上下文重叠。
3. **困难。** 构建一个有 1000 个实体的领域知识库（例如公司的员工 + 产品），端到端实现 NER + 实体链接，在 100 个留出句子上测量精确率和召回率。

## 关键术语

| 术语 | 常见的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 实体链接（EL） | 链接到 Wikipedia | 把提及映射到唯一的知识库条目。 |
| 候选生成 | 可能是谁？ | 为一个提及返回合理的知识库条目候选列表。 |
| 消歧 | 选对的那个 | 用上下文给候选打分，选出胜者。 |
| 别名索引 | 查找表 | 从表面形式 → 候选实体的映射。 |
| NIL | 不在知识库中 | 显式预测没有知识库条目匹配。 |
| KB | 知识库 | Wikidata、Wikipedia、DBpedia 或你的领域知识库。 |
| AIDA-CoNLL | 基准数据集 | 1,393 篇带有人工实体链接标注的 Reuters 文章。 |

## 延伸阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://researchcommons.waikato.ac.nz/entities/publication/b9a0b520-abc5-47c5-a86a-da6c579893ab) —— 奠定基础的先验 + 上下文方法。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) —— 基于 embedding 的主力方法。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) —— 带约束解码的生成式实体链接。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) —— 基准数据集论文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) —— 开源生产级技术栈。
