# BERT —— 掩码语言建模

> GPT 预测下一个词，BERT 预测缺失的词。一句话之差——以及此后五年所有 embedding 形态的东西。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 5 · 02 (Text Representation)
**Time:** ~45 minutes

## 问题

2018 年，每个 NLP 任务——情感、NER、QA、蕴含——都要在自己的标注数据上从零训练自己的模型。没有一个可以拿来微调的预训练「理解英语」checkpoint。ELMo（2018）展示了可以用双向 LSTM 预训练上下文 embedding；有帮助，但没有泛化。

BERT（Devlin 等，2018）问：如果我们拿一个 Transformer 编码器，在互联网上的每个句子上训练它，强迫它从两侧的上下文预测缺失的词，会怎样？然后你在下游任务上微调一个头。参数效率令人耳目一新。

结果：18 个月内，BERT 及其变体（RoBERTa、ALBERT、ELECTRA）统治了当时存在的所有 NLP 排行榜。到 2020 年，地球上每个搜索引擎、内容审核 pipeline 和语义搜索系统里都有一个 BERT。

2026 年，仅编码器模型仍是分类、检索和结构化抽取的正确工具——它们每个 token 的运行速度比解码器快 5–10 倍，其 embedding 是每个现代检索栈的骨干。ModernBERT（2024 年 12 月）用 Flash Attention + RoPE + GeGLU 把架构推到了 8K 上下文。

## 概念

![掩码语言建模：挑选 token、加掩码、预测原文](../assets/bert-mlm.svg)

### 训练信号

取一个句子：`the quick brown fox jumps over the lazy dog`。

随机掩码 15% 的 token：

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

训练模型预测被掩码位置上的原始 token。因为编码器是双向的，预测位置 1 的 `[MASK]` 时可以利用位置 2 及之后的 `brown fox jumps`。这正是 GPT 做不到的事情。

### BERT 的掩码规则

在被选中用于预测的 15% token 中：

- 80% 替换为 `[MASK]`。
- 10% 替换为随机 token。
- 10% 保持不变。

为什么不总是用 `[MASK]`？因为 `[MASK]` 在推理时从不出现。如果训练模型期望 100% 的掩码位置都是 `[MASK]`，就会在预训练和微调之间造成分布偏移。10% 随机 + 10% 不变让模型保持诚实。

### 下一句预测（NSP）——以及为什么被放弃

原始 BERT 还在 NSP 上训练：给定句子 A 和 B，预测 B 是否跟随 A。RoBERTa（2019）做了消融实验，表明 NSP 有害无益。现代编码器都跳过它。

### 2026 年有什么变化：ModernBERT

2024 年的 ModernBERT 论文用 2026 年的基础件重建了模块：

| 组件 | 原始 BERT（2018） | ModernBERT（2024） |
|-----------|----------------------|-------------------|
| 位置编码 | 可学习绝对位置 | RoPE |
| 激活函数 | GELU | GeGLU |
| 归一化 | LayerNorm | Pre-norm RMSNorm |
| 注意力 | 全稠密 | 交替局部（128）+ 全局 |
| 上下文长度 | 512 | 8192 |
| Tokenizer | WordPiece | BPE |

与 2018 年的技术栈不同，它原生支持 Flash Attention。在序列长度 8K 时，推理比 DeBERTa-v3 快 2–3 倍，GLUE 分数也更好。

### 2026 年仍选编码器的使用场景

| 任务 | 为什么编码器胜过解码器 |
|------|---------------------------|
| 检索 / 语义搜索 embedding | 双向上下文 = 每个 token 的 embedding 质量更好 |
| 分类（情感、意图、毒性） | 一次前向传播；没有生成开销 |
| NER / token 标注 | 逐位置输出，原生双向 |
| 零样本蕴含（NLI） | 在编码器之上加分类头 |
| RAG 的 reranker | 交叉编码器打分，比 LLM reranker 快 10 倍 |

```figure
transformer-residual
```

## 构建它

### 第 1 步：掩码逻辑

见 `code/main.py`。函数 `create_mlm_batch` 接收 token ID 列表、词表大小和掩码概率。返回 input ID（已应用掩码）和 label（只在被掩码的位置有值，其余为 -100——PyTorch 的 ignore index 约定）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### 第 2 步：在微型语料上运行 MLM 预测

在 20 个词、200 个句子的词表上训练一个 2 层编码器 + MLM 头。没有梯度——我们只做前向传播的健全性检查。完整训练需要 PyTorch。

### 第 3 步：比较掩码类型

展示三路规则如何让模型在没有 `[MASK]` 的情况下也能使用。在未掩码的句子上和掩码过的句子上做预测。两者都应产生合理的 token 分布，因为模型在训练中见过两种模式。

### 第 4 步：微调头

在一个玩具情感数据集上，把 MLM 头换成分类头。只有头参与训练；编码器冻结。这是每个 BERT 应用都遵循的模式。

## 使用它

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Embedding 模型就是微调过的 BERT。** `sentence-transformers` 的模型如 `all-MiniLM-L6-v2` 是用对比损失训练的 BERT。编码器相同，损失变了。

**交叉编码器 reranker 也是微调过的 BERT。** 在 `[CLS] query [SEP] doc [SEP]` 上做配对分类。query 与 doc 之间的双向注意力，正是交叉编码器质量胜过双编码器的原因。

**2026 年什么时候不该选 BERT。** 任何生成式任务。编码器没有合理的方式自回归地产生 token。还有：参数量低于 1B、小型解码器能以更大灵活性匹配质量的场景（Phi-3-Mini、Qwen2-1.5B）。

## 交付它

见 `outputs/skill-bert-finetuner.md`。这个 skill 为新的分类或抽取任务规划 BERT 微调（骨干选择、头规范、数据、评估、停止条件）。

## 练习

1. **简单。** 运行 `code/main.py`，在 10,000 个 token 上打印掩码分布。确认约 15% 被选中，其中约 80% 变成 `[MASK]`。
2. **中等。** 实现整词掩码：如果一个词被分词成多个子词，要么全部一起掩码，要么都不掩码。在 500 句的语料上测量这是否提高 MLM 准确率。
3. **困难。** 在公共数据集的 10,000 个句子上训练一个微型 BERT（2 层，d=64）。为 SST-2 情感任务微调 `[CLS]` token。与参数量匹配的仅解码器基线比较——谁赢？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| MLM | 「掩码语言建模」 | 训练信号：随机把 15% 的 token 替换为 `[MASK]`，预测原文。 |
| Bidirectional | 「双向看」 | 编码器注意力没有因果掩码——每个位置都能看到所有其他位置。 |
| `[CLS]` | 「池化 token」 | 每个序列前置的特殊 token；其最终 embedding 用作句子级表示。 |
| `[SEP]` | 「片段分隔符」 | 分隔成对的序列（例如 query/doc、句子 A/B）。 |
| NSP | 「下一句预测」 | BERT 的第二个预训练任务；RoBERTa 证明其无用，2019 年后被放弃。 |
| Fine-tuning | 「适配任务」 | 编码器基本冻结；在顶部训练一个小头以完成下游任务。 |
| Cross-encoder | 「一种 reranker」 | 同时接收 query 和 doc 作为输入、输出相关性分数的 BERT。 |
| ModernBERT | 「2024 年的刷新」 | 用 RoPE、RMSNorm、GeGLU、交替局部/全局注意力重建的编码器，8K 上下文。 |

## 延伸阅读

- [Devlin 等（2018）. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) —— 原始论文。
- [Liu 等（2019）. RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) —— 如何正确地训练 BERT；终结了 NSP。
- [Clark 等（2020）. ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) —— 替换 token 检测在同等计算量下胜过 MLM。
- [Warner 等（2024）. Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) —— ModernBERT 论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) —— 规范的编码器参考实现。
