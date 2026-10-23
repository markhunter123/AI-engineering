# 预训练的数据流水线

> 模型是一面镜子。你喂它什么数据，它就反射什么。喂它垃圾，它就会以完美的流利度反射垃圾。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lessons 01-02 (Tokenizers, Building a Tokenizer)
**Time:** ~90 minutes

## 学习目标

- 构建流式数据流水线，对 TB 级文本执行分词、分块、打乱和批处理，而无需把全部数据载入内存
- 实现真实预训练流水线中使用的数据质量过滤器（去重、语言检测、内容过滤）
- 创建定长训练序列，配备正确的注意力掩码与文档边界处理
- 剖析流水线吞吐量，确保 dataloader 跟得上 GPU 训练速度

## 问题

你有分词器了。现在你需要数据。

不是一份数据集。不是一份 CSV 文件。是 TB 级的文本——清洗、去重、按质量过滤、分词成定长序列，并以随机批次的形式、以足够快的速度供给，让你的 8 卡 GPU 集群永远不用等下一批。

大多数人以为训练 LLM 的关键是模型架构。并非如此。Llama 3 用了 15.6 万亿 token。GPT-3 用了 3000 亿。DeepSeek-V2 用了 8.1 万亿。三者的架构大致相同：堆叠的 Transformer 块，带注意力层和前馈层。输出质量的差异绝大多数来自数据。

DeepMind 的 Chinchilla 论文把这件事说得很精确。在给定算力预算下，模型参数量与训练 token 数之间存在一个最优比例。Chinchilla 表明 2022 年的大多数模型都严重欠训练——相对于它们见过的数据量，参数太多了。一个 700 亿参数、在 1.4 万亿 token 上训练的模型（Chinchilla 最优），胜过了在 3000 亿 token 上训练的 2800 亿参数模型（Gopher）。

你的数据流水线决定了模型学到的到底是语言，还是噪声。

## 概念

### 数据从哪来

每个大语言模型都在混合数据源上训练。确切的配比对多数实验室来说是严格保守的秘密，但我们知道的足以理解这些类别。

| 来源 | 规模 | 质量 | 使用者 |
|--------|------|---------|---------|
| Common Crawl | 原始约 250 TB | 低（需要重度过滤） | GPT-3、Llama、多数开源模型 |
| Wikipedia | 约 20 GB | 高 | 每个主流 LLM |
| GitHub 代码 | 约 1 TB+ | 中（大量重复、死代码） | StarCoder、CodeLlama、DeepSeek-Coder |
| 书籍（BookCorpus、Pile） | 约 100 GB | 高 | GPT-2、GPT-3、早期模型 |
| 学术论文（arXiv、S2ORC） | 约 100 GB | STEM 领域质量高 | Llama、Galactica |
| StackOverflow、Reddit | 约 100 GB | 中 | Llama、Falcon |
| 精选网页（C4、RefinedWeb） | 约 5 TB | 中高（已预过滤） | T5、Falcon |

Llama 3 公开了它的数据配比：约 50% 网页数据、25% 代码、13% 书籍与学术论文、8% 数学数据、4% 多语言网页数据。总量是 15.6 万亿 token，源数据超过 5 TB 原始文本。

配比与总量同样重要。网页数据太多，模型就变成 Reddit 鹦鹉。代码太少，它就不会编程。数学太少，它推理就会失败。把配比调对是训练 LLM 最难的部分之一，而且没有公式——它需要实验和评估。

### 数据清洗

原始网页数据很脏。一份典型的 Common Crawl 转储包含：

- HTML 标签和 JavaScript
- 模板化的页眉、页脚、导航菜单
- 重复页面（完全重复和近似重复）
- 机器生成的垃圾信息
- 个人可识别信息（PII）
- 低质量文本（关键词列表、SEO 垃圾）
- 以文本形式编码的非文本内容

清洗不是可选项。它决定了一个模型是生成连贯的段落，还是把 HTML 标签和商品列表混在一起吐出来。

```mermaid
graph TD
    A[Raw Text] --> B[HTML Strip]
    B --> C[Language Detection]
    C --> D[Quality Filter]
    D --> E[Deduplication]
    E --> F[PII Removal]
    F --> G[Clean Text]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#e94560,color:#fff
```

每一步消除一类噪声：

**HTML 剥离：** 移除所有标记，只保留可见文本内容。`trafilatura` 或 `readability` 这类库能提取文章正文，同时丢弃导航、广告和模板内容。

**语言检测：** 用 fastText 的语言识别模型（lid.176.bin）为每篇文档分类。过滤到你目标的语言。一篇被判为英文但置信度低于 0.8 的文档，很可能不是干净的英文。

**质量过滤：** 这里开始有意思了。RefinedWeb（Falcon 背后的数据集）使用基于困惑度的过滤器：在 Wikipedia 上训练一个小语言模型，然后给每篇文档打分。困惑度高意味着文档不像 Wikipedia——很可能是垃圾、关键词列表或机器生成内容。困惑度超过阈值的文档会被移除。

**去重：** 影响最大的单一清洗步骤。Common Crawl 包含数量惊人的重复页面——法律免责声明、cookie 提示、服务条款。在重复数据上训练会浪费算力，还可能让模型记住并逐字复述特定段落。

**PII 移除：** 姓名、电子邮件地址、电话号码、社会安全号码。结构化 PII 用基于正则的检测，上下文中的姓名用 NER 模型。

### 用 MinHash 去重

精确去重很简单：对每篇文档做哈希，删掉重复项。但近似重复才是真正的问题。同一篇新闻文章的两份拷贝，周边广告略有不同，就是近似重复。内容 95% 相同，但逐字节比较则有差异。

MinHash + 局部敏感哈希（LSH）能高效解决这个问题。

```mermaid
graph LR
    A[Document] --> B[Shingling]
    B --> C[MinHash Signature]
    C --> D[LSH Buckets]
    D --> E[Candidate Pairs]
    E --> F[Jaccard Similarity]
    F --> G[Deduplicated Set]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#e94560,color:#fff
```

思路：

1. **Shingling（瓦片化）：** 把每篇文档转换成一组 n-gram（例如由单词或字符构成的 5-gram）。「the quick brown fox」用 3 词瓦片会变成 {"the quick brown", "quick brown fox"}。

2. **MinHash：** 对每篇文档的瓦片集合，计算 k 个哈希值。每个哈希值是所有瓦片在某个哈希函数下的最小值。这样得到一个固定大小的「签名」，近似任意两篇文档之间的 Jaccard 相似度。

3. **LSH：** 根据 MinHash 签名的分段（band）把文档分到桶中。同一桶中的文档是候选近似重复项。这避免了两两比较——你只需比较候选对。

4. **验证：** 对每个候选对计算精确的 Jaccard 相似度。相似度超过阈值（通常 0.8）就删除其中一份。

Llama 团队报告称，通过去重移除了约 38% 的网页数据。这不是个小数字。Common Crawl 中有超过三分之一是重复或近似重复的内容。

### 序列打包

模型期望定长输入序列。你的文档长度不一。有些 50 个 token，有些 50,000 个。

朴素做法：把每篇文档填充到最大序列长度。这在填充 token 上浪费大量算力，而它们对学习毫无贡献。

更好的做法：把多篇文档打包进一条序列，用序列结束 token 分隔。一条 2048 token 的序列可能包含三篇短文档，之间用 [EOS] token 连接。

```mermaid
graph TD
    subgraph Naive Packing
        A1["Doc A (200 tokens)"] --> P1["[PAD] x 1848"]
        A2["Doc B (500 tokens)"] --> P2["[PAD] x 1548"]
        A3["Doc C (100 tokens)"] --> P3["[PAD] x 1948"]
    end

    subgraph Efficient Packing
        B1["Doc A (200) | Doc B (500) | Doc C (100) | Doc D (400) | Doc E (848)"]
    end

    style A1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style A2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style A3 fill:#1a1a2e,stroke:#e94560,color:#fff
    style P1 fill:#333,stroke:#666,color:#999
    style P2 fill:#333,stroke:#666,color:#999
    style P3 fill:#333,stroke:#666,color:#999
    style B1 fill:#1a1a2e,stroke:#16c784,color:#fff
```

注意力掩码必须设置正确。同一条打包序列中，文档 A 的 token 不应注意到文档 B 的 token。这需要块对角注意力掩码。

长文档会被截断，或在序列边界处切块。切分点很重要：从句子中间切开会迫使模型看到不完整的想法。有些流水线在可能时会按段落或句子边界对齐切分点。

### Chinchilla 扩展定律

对于固定的算力预算 $C$（以 FLOPs 计），最优模型规模 $N$ 与数据集规模 $D$ 满足：

```math
\begin{aligned}
N_{\text{opt}} &\sim C^{0.5} \\
D_{\text{opt}} &\sim C^{0.5}
\end{aligned}
```

在实践中，这意味着模型规模与数据集规模应当大致等比扩展。参数多 10 倍的模型，大约需要多 10 倍的训练 token 才能达到相同的损失。

| 模型 | 参数量 | 训练 token 数 | 是否符合 Chinchilla 最优？ |
|-------|-----------|----------------|-------------------|
| GPT-3 | 175B | 300B | 否（欠训练 3-4 倍） |
| Chinchilla | 70B | 1.4T | 是（按设计如此） |
| Llama 2 | 70B | 2T | 过训练（有意为之） |
| Llama 3 | 70B | 15T | 严重过训练 |

Llama 3 故意违反 Chinchilla 定律。Meta 发现，在更多数据上过训练——远超算力最优比例——能为推理产出更好的模型。额外的训练成本只需支付一次，但更小的模型永远更便宜地提供服务。这有时被称为「推理最优」扩展方法，自 2024 年起已成为行业标准。

```figure
l5-data-pipeline
```

## 构建它

### 第 1 步：文本清洗

剥离 HTML、归一化空白、移除非文本内容。我们用一段公版文本（Project Gutenberg）作为小语料。

```python
import re

def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def quality_filter(text, min_words=50, max_ratio_caps=0.3, max_ratio_special=0.1):
    words = text.split()
    if len(words) < min_words:
        return False
    caps_ratio = sum(1 for w in words if w.isupper()) / len(words)
    if caps_ratio > max_ratio_caps:
        return False
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special_chars / max(len(text), 1) > max_ratio_special:
        return False
    return True
```

质量过滤器能抓住 SEO 垃圾（全大写）、机器生成的噪声（特殊字符比例高）和占位页（太短）。仅这三项检查就能从网页抓取数据中清除数量惊人的垃圾。

### 第 2 步：MinHash 去重

从零实现 MinHash。不需要外部库——只用 `hashlib`。

```python
import hashlib
from collections import defaultdict

def get_shingles(text, k=5):
    words = text.lower().split()
    if len(words) < k:
        return set()
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}

def minhash_signature(shingles, num_hashes=128):
    signature = []
    for i in range(num_hashes):
        min_hash = float("inf")
        for shingle in shingles:
            h = int(hashlib.sha256(f"{i}:{shingle}".encode()).hexdigest(), 16)
            min_hash = min(min_hash, h)
        signature.append(min_hash)
    return signature

def lsh_buckets(signature, bands=16):
    rows_per_band = len(signature) // bands
    buckets = []
    for b in range(bands):
        start = b * rows_per_band
        band_data = tuple(signature[start:start + rows_per_band])
        bucket_hash = hashlib.md5(str(band_data).encode()).hexdigest()
        buckets.append((b, bucket_hash))
    return buckets

def deduplicate(documents, threshold=0.8, num_hashes=128, bands=16):
    signatures = []
    shingle_sets = []
    for doc in documents:
        shingles = get_shingles(doc)
        shingle_sets.append(shingles)
        signatures.append(minhash_signature(shingles, num_hashes))

    bucket_map = defaultdict(list)
    for doc_idx, sig in enumerate(signatures):
        for band_id, bucket_hash in lsh_buckets(sig, bands):
            bucket_map[(band_id, bucket_hash)].append(doc_idx)

    duplicate_pairs = set()
    for bucket_docs in bucket_map.values():
        if len(bucket_docs) < 2:
            continue
        for i in range(len(bucket_docs)):
            for j in range(i + 1, len(bucket_docs)):
                duplicate_pairs.add((bucket_docs[i], bucket_docs[j]))

    removed = set()
    for i, j in duplicate_pairs:
        if i in removed or j in removed:
            continue
        s1, s2 = shingle_sets[i], shingle_sets[j]
        if not s1 or not s2:
            continue
        jaccard = len(s1 & s2) / len(s1 | s2)
        if jaccard >= threshold:
            removed.add(j)

    return [doc for idx, doc in enumerate(documents) if idx not in removed], len(removed)
```

`num_hashes=128` 和 `bands=16` 两个参数控制精确率与召回率的权衡。哈希越多，相似度估计越准确。分段越多，召回率越高（抓到更多重复项），但误报也更多。这些取值对典型网页文本效果不错。

### 第 3 步：分词并打包序列

把清洗、去重后的文本分词，打包成训练用的定长序列。

```python
def tokenize_corpus(documents, tokenizer):
    all_tokens = []
    for doc in documents:
        tokens = tokenizer.encode(doc)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_id)
    return all_tokens

def pack_sequences(token_ids, seq_length, pad_id=0):
    sequences = []
    attention_masks = []
    for i in range(0, len(token_ids), seq_length):
        seq = token_ids[i:i + seq_length]
        mask = [1] * len(seq)
        if len(seq) < seq_length:
            pad_count = seq_length - len(seq)
            seq = seq + [pad_id] * pad_count
            mask = mask + [0] * pad_count
        sequences.append(seq)
        attention_masks.append(mask)
    return sequences, attention_masks
```

### 第 4 步：训练用 DataLoader

产出随机化的打包序列批次。这就是训练循环消费的东西。

```python
import random

class PreTrainingDataLoader:
    def __init__(self, sequences, attention_masks, batch_size, shuffle=True):
        self.sequences = sequences
        self.attention_masks = attention_masks
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.sequences) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.sequences)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            batch_seqs = [self.sequences[i] for i in batch_idx]
            batch_masks = [self.attention_masks[i] for i in batch_idx]
            yield batch_seqs, batch_masks
```

### 第 5 步：数据集统计

计算真正重要的数字：总 token 数、唯一 token 数、压缩比、文档长度分布。

```python
from collections import Counter

def compute_statistics(documents, token_ids, sequences, tokenizer_vocab_size):
    total_chars = sum(len(d) for d in documents)
    total_tokens = len(token_ids)
    unique_tokens = len(set(token_ids))
    compression_ratio = total_chars / total_tokens

    doc_lengths = [len(d.split()) for d in documents]
    avg_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)
    max_doc_length = max(doc_lengths) if doc_lengths else 0
    min_doc_length = min(doc_lengths) if doc_lengths else 0

    token_counts = Counter(token_ids)
    top_tokens = token_counts.most_common(10)

    non_pad_tokens = sum(sum(1 for t in seq if t != 0) for seq in sequences)
    total_positions = sum(len(seq) for seq in sequences)
    utilization = non_pad_tokens / max(total_positions, 1)

    stats = {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "vocab_utilization": unique_tokens / tokenizer_vocab_size,
        "compression_ratio": compression_ratio,
        "avg_doc_length_words": avg_doc_length,
        "max_doc_length_words": max_doc_length,
        "min_doc_length_words": min_doc_length,
        "num_sequences": len(sequences),
        "sequence_utilization": utilization,
        "top_10_tokens": top_tokens,
    }
    return stats
```

压缩比告诉你分词器在这个语料上的效率。英文文本通常压缩到每个 token 约 3-4 个字符。如果你看到每个 token 只有 1.5 个字符，说明分词器切得太碎。如果看到 8 个以上，说明它学到了非常领域特定的合并。

序列利用率告诉你打包序列中有多少是真实数据、多少是填充。低于 90% 意味着打包效率低——你在填充 token 上浪费算力。

## 使用它

### 与 HuggingFace Datasets 对比

通过 HuggingFace 的 datasets 库加载同一语料，并比较流水线速度。

```python
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

import time

start = time.time()
tokenized = ds.map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=2048),
    batched=True,
    num_proc=4,
)
hf_time = time.time() - start
total_tokens = sum(len(t) for t in tokenized["input_ids"])
print(f"HuggingFace: {total_tokens:,} tokens in {hf_time:.2f}s ({total_tokens/hf_time:,.0f} tokens/sec)")
```

HuggingFace 流水线底层使用 Rust 分词器，并在 4 个核心上并行处理。你的纯 Python 流水线会慢 10-50 倍。这个差距就是生产团队使用编译型分词器的原因。算法相同，差别在实现语言。

## 交付它

本课产出用于验证和调试 LLM 训练流水线数据质量的 prompt。见 `outputs/prompt-data-quality-checker.md`。

## 练习

1. **简单：** 用简单启发式（字符集分析）给清洗流水线添加语言检测。只保留英文文档，并测量有多少文档被移除。
2. **中等：** 在 MinHash 近似去重之外，用 SHA-256 哈希实现精确去重。在网页抓取语料上比较两种方法各自抓到的重复项数量。
3. **困难：** 构建基于困惑度的质量过滤器。在 Wikipedia 文本上训练一个小型 bigram 语言模型，按困惑度为每篇文档打分，移除底部 20%。比较在过滤后与未过滤数据上训练的模型输出质量。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| Common Crawl | 「互联网」 | 一个每月抓取网页的非营利组织——原始数据约 250TB，是大多数 LLM 训练数据的起点 |
| MinHash | 「某种哈希技巧」 | 用固定大小的签名估计集合间 Jaccard 相似度的技术——使近似重复检测能规模化进行 |
| LSH | 「Locality-Sensitive Hashing」 | 把相似项分到同一桶的方法——把两两比较从 $O(n^2)$ 降到接近线性 |
| 序列打包 | 「拼接文档」 | 把多篇文档装入定长序列并配以正确的注意力掩码——消除填充浪费 |
| Chinchilla 扩展 | 「用更多数据训练」 | 在固定算力预算下，最优性能要求模型规模与训练 token 数大致等比扩展 |
| Fertility | 「每词 token 数」 | 每个单词的平均 token 数——GPT-4 处理英文为 1.3，非拉丁文字更高 |
| 数据混合 | 「选择训练数据」 | 代码、文本、数学、多语言数据之间的配比——没有公式，需要实验 |
| 困惑度过滤器 | 「质量打分」 | 用一个小语言模型给文档打分——困惑度高说明文本不像干净的参考数据 |
| 去重 | 「删除拷贝」 | 消除完全重复和近似重复的文档——通常能移除原始网页数据的 30-40% |
| 注意力掩码 | 「该看哪些 token」 | 一个二值掩码，阻止打包序列中的注意力跨越文档边界 |

## 延伸阅读

- [Hoffmann et al., 2022 -- Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)——改变了我们对数据规模看法的论文
- [Penedo et al., 2023 -- The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116)——如何把 Common Crawl 过滤到高质量
- [Touvron et al., 2023 -- Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)——Llama 2 的数据流水线细节
- [Lee et al., 2022 -- Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)——为什么去重比你想象的更重要
- [Broder, 1997 -- On the Resemblance and Containment of Documents](https://ieeexplore.ieee.org/document/666900)——MinHash 的原始论文
- [Meta, 2024 -- Llama 3 Technical Report](https://arxiv.org/abs/2407.21783)——15.6T token、数据混合比例、过滤流水线
