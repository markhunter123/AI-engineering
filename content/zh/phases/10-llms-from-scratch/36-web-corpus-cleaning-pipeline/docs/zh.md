# Web Corpus Cleaning: Filtering, Deduplication, and Mixing

> 原始 Common Crawl 是一座垃圾填埋场：其中大部分是样板文本、垃圾信息和重复内容。预训练工程师的工作，就是把它变成可重复、可度量的训练配比。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10 · 03 (Data Pipelines)
**Time:** ~70 minutes

## Learning Objectives

- 解释为什么原始网页抓取数据中只有一小部分可用文本，以及质量过滤器如何将其回收。
- 实现一套 Gopher 风格的规则过滤器，阈值可调，并统计每条规则的淘汰数量。
- 推导 MinHash 签名与 LSH 分带技术，并用它们以近似线性成本检测近似重复文档。
- 构建基于 temperature 的配比器，按 $p_i \propto s_i^{1/T}$ 上采样稀缺的高质量来源。
- 把三个阶段组装成一个接收 `List[str]` 的清洗 API，可直接替换到真实 Common Crawl WET 分片上。
- 解读存活率漏斗，并用它调试哪条过滤器过度移除了有效信号。

## The Problem

你已经有了数据管道，手里有数 TB 的 Common Crawl WET 文件，但还没有训练语料。

原始 Common Crawl 出了名的脏。对网页级抓取的独立审计反复发现，大多数文档都不适合语言模型训练：HTML 样板、导航菜单、Cookie 横幅、SEO 关键词农场、机器生成的垃圾、乱码、成人内容、个人信息，以及数量惊人的精确与近似重复文本。OpenAI 的 GPT-3 论文提到，Common Crawl 质量太低，以至于需要一个额外的基于分类器的过滤器；Gopher 论文（Rae et al. 2021）表明，一套设计良好的规则过滤器可以移除约一半的网页抓取数据，同时降低下游困惑度；FineWeb 报告（Penedo et al. 2024）则证明，单是去重就足以改变有效 token 分布并影响基准测试。

语料质量对小模型而言不是可选项，而是下限。一个用干净数据训练的 1B 参数模型，通常能在大多数下游任务上超过用原始抓取数据训练的 3B 模型。原因很简单：小模型没有多余参数去记忆导航页脚和重复的隐私政策。清洗是预训练里杠杆最高的工程步骤，而且它大部分是确定性代码，可以写测试、版本化和审计。

本课构建一个三阶段管道，把一批原始网页文档变成去重后的、按来源加权的训练配比：

```mermaid
graph LR
    A[原始文档] --> B[规则过滤]
    B --> C[MinHash + LSH]
    C --> D[Temperature 配比]
    D --> E[训练 mix]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#16c784,color:#fff
```

## The Concept

### 为什么 Common Crawl 大部分不可用

每月的 Common Crawl 快照包含数百 TB 的 WET 文本，其中可用比例很小。失败模式可以归为几类：

| 失败模式 | 看起来什么样 | 对训练的伤害 |
|----------|-------------|-------------|
| 样板文本 | "Accept cookies"、"Privacy policy"、"Copyright 2024" | 模型会记忆重复的导航文本，而不是学习语言。 |
| SEO 垃圾 | "Cheap flights cheap flights cheap hotels" | 关键词堆砌没有句法结构。 |
| 编码乱码 | `Ã©Â¢ÂÃ¥Â¤Â´` 而不是 `额头` | 让模型以为 token 以随机顺序出现。 |
| 过短 | 只有一句话的 stub 页面 | 上下文不足以学习长程依赖。 |
| 机器文本 | Lorem ipsum、占位文本 | 没有真实语义。 |
| 近似重复 | 同一篇文章配不同广告 | 浪费算力，并导致逐字复述。 |
| 停用词比例过低 | 长串技术标识符 | 不是自然语言。 |

Gopher 和 C4 的处理方式不同。Gopher（Rae et al. 2021）发布了一套明确的启发式清单。C4（Raffel et al. 2019）使用更小但更激进的过滤器，专为英语 T5 预训练调优。CCNet（Wenzek et al. 2019）加入了语言识别和困惑度过滤。FineWeb（Penedo et al. 2024）则把这些思路与现代去重技术结合，应用到 Common Crawl 规模。

### Gopher 的规则清单

Gopher 的过滤器刻意保持简单。它们不需要学习模型，因此运行快、可审计、容易复现。最重要的规则包括：

1. **最小文档长度。** 低于词数阈值的文档几乎总是 stub、错误页或导航。
2. **符号占比。** 非字母数字、非空白字符过多，通常意味着编码乱码、标记泄漏或源代码被当成文本。
3. **大写比例。** 大写单词比例过高是强烈的垃圾信号。
4. **重复行占比。** 大量重复同一行的页面通常是错误信息、服务条款或生成的表格。
5. **重复段落占比。** 页脚和页头的样板会造成段落重复。
6. **停用词比例。** 自然语言包含冠词、介词、代词。停用词几乎为零的文档通常是关键词列表或标识符堆。
7. **省略号结尾。** 以 `...` 结尾的文档往往是截断或占位文本。
8. **平均词长。** 平均词长过长意味着标识符拼接或混淆文本。

每条规则都有一个阈值，每个被移除的文档都会按规则计数。这个计数就是存活率漏斗：如果某条规则移除了 40% 的数据，你需要先看它在移除什么，再决定是否信任它。

### C4 启发式

C4 使用更小但部分重叠的过滤器。以下情况会被移除：

- 少于三句话；
- 任意一行少于三个词且以标点结尾（可能是菜单项）；
- 词数低于阈值；
- 某一行包含单词 `javascript`；
- 包含 `lorem ipsum`；
- 包含花括号 `{`（可能是源码或 JSON）；
- 文档内某短语重复出现。

C4 针对英语调优。多语言管道通常把其中一些规则替换为语言专用启发式，或采用 CCNet 的语言识别加困惑度方案。

### 从精确去重到近似去重

精确去重很简单：用 SHA-256 给每个文档哈希，每个哈希保留一份。真正的问题是近似重复。两篇只在广告不同的同一新闻文章，字节级不同，但内容 95% 相同，精确哈希会漏掉它们。

标准解法是 **MinHash** 加 **局部敏感哈希（LSH）**。

1. **Shingle。** 把每个文档变成 $k$-gram 集合。"the quick brown fox jumps" 的 5 词 shingle 是 `the quick brown fox jumps`。
2. **MinHash 签名。** 对 $m$ 个哈希函数，分别记录所有 shingle 中的最小哈希值。Jaccard 相似度高的两篇文档会在很多最小值上达成一致。
3. **LSH 分带。** 把 $m$ 维签名分成 $b$ 个 band，每个 band 有 $r$ 行。只要两个文档在任意 band 上碰撞，它们就成为候选重复对。
4. **验证。** 只对候选对计算精确 Jaccard 相似度。

两个 shingle 集合 $A$、$B$ 的 Jaccard 相似度为：

```math
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
```

MinHash 有一个漂亮的性质：两篇文档在签名任意一行上一致的概率等于它们的 Jaccard 相似度 $s$。把签名分成 $b$ 个 band、每个 band $r$ 行后，相似度为 $s$ 的两篇文档成为候选对的概率是：

```math
P(\text{candidate}) = 1 - (1 - s^r)^b
```

当 $b=16$、$r=4$ 时，$s=0.8$ 的候选概率为：

```math
1 - (1 - 0.8^4)^{16} \approx 1 - 0.0002 = 0.9998
```

而 $s=0.3$ 的候选概率只有：

```math
1 - (1 - 0.3^4)^{16} \approx 0.05
```

这条类 S 型曲线正是 LSH 的价值所在。没有 LSH，比较 $N$ 篇文档需要 $O(N^2)$；有了 LSH，每篇文档只需被哈希到少量 bucket，再扫描 bucket 找候选，成本近似线性于 $N$。

### 用 temperature 配比来源

过滤和去重之后，你会得到来自多个来源的干净文档：网页、书籍、代码、论文、论坛。接下来需要训练配比。

最简单的方法是按来源大小比例采样：80% 网页 token、20% 代码 token 就抽 80% 网页。但这把每个来源当成同质量。实际上，代码和书籍的每 token 质量通常高于原始网页，因此实验室会故意上采样它们。

常见做法是 **temperature scaling**。设 $s_i$ 是来源 $i$ 的可用大小，则期望权重为：

```math
p_i \propto s_i^{1/T}
```

- $T=1$ 是比例采样。
- $T < 1$ 会让分布更尖锐，偏向最大来源。
- $T > 1$ 会让分布更平坦，上采样较小、通常更高质量的来源。

例如网页大小是代码的 10 倍：

- $T=0.5$ 时指数为 2，权重为 $100:1$，差距放大到 100 倍；
- $T=1.0$ 时权重为 $10:1$；
- $T=2.0$ 时指数为 0.5，权重为 $\sqrt{10}:1 \approx 3.16:1$。

所以低温会放大主导来源，高温会让稀有来源更接近均势。实验室通过消融而不是拍脑袋来选择 temperature：用多个配比训练小模型，选在留出评估上困惑度最低的配比。

## Build It

### 阶段 1：规则过滤器

第一阶段是可配置的 `RuleFilter`。每条规则返回一个淘汰原因，管道统计每条规则移除多少文档，形成存活率漏斗。

```python
from main import RuleFilter

config = {
    "min_words": 30,
    "max_symbol_ratio": 0.20,
    "max_caps_ratio": 0.20,
    "max_duplicate_line_ratio": 0.50,
    "max_duplicate_paragraph_ratio": 0.60,
    "min_stopword_ratio": 0.03,
    "ban_ellipsis_ending": True,
    "max_mean_word_length": 12,
    "max_lines_with_ellipsis_ratio": 0.20,
}

flt = RuleFilter(config)
reasons = flt.reasons("BUY NOW SALE DISCOUNT BUY NOW SALE DISCOUNT...")
print(reasons)  # ['max_caps_ratio', 'min_stopword_ratio']
```

`code/main.py` 中的实现直接把 Gopher 规则翻译成可审计的 Python。每个阈值都是具名常量，因此你可以从配置中去掉某条规则来关闭它，或者在训练源代码等符号比例天然较高的领域放宽它。

## Verify It

`code/tests/test_main.py` 锁定了过滤器阶段：

- 每个合成垃圾文档都被至少一条规则拒绝；
- 每个合成正常文档都通过所有规则；
- 每条规则的淘汰数加上保留数与输入规模一致。

运行方式：

```console
python3 -m unittest discover tests -v
```

## Build It

### 阶段 2：MinHash + LSH 去重

`MinHashDeduplicator` 构建 64 行的 MinHash 签名，并把它哈希到 16 个 band、每个 band 4 行。随后用精确 Jaccard 相似度验证候选对，并返回重复簇。

```python
from main import MinHashDeduplicator

dedup = MinHashDeduplicator(
    num_hashes=64,
    bands=16,
    rows_per_band=4,
    shingle_size=5,
)

kept, indices, clusters = dedup.deduplicate(docs, threshold=0.8)
```

一个 cluster 是彼此近似重复的原始索引列表。去重器只保留每个 cluster 的第一篇文档，其余移除。

### 阶段 3：Temperature 配比器

配比器接收一个从来源名到文档列表的字典，按 temperature 缩放后的权重采样目标 mix。

```python
from main import temperature_mix

mix, counts = temperature_mix(
    {"web": web_docs, "code": code_docs, "discussion": discussion_docs},
    total=10_000,
    temperature=0.7,
)
```

$T=0.7$ 比比例采样稍尖锐：它偏向最大来源，同时仍保留足够的小来源样本，让模型学到它们的风格。

### 面向 batch-of-strings 的 API

三个阶段被包装进 `CorpusCleaner`，它接收字符串列表和可选的来源标签，返回一个 `CleaningResult`。

```python
from main import CorpusCleaner

cleaner = CorpusCleaner(mix_total=10_000, mix_temperature=0.7)
result = cleaner.clean(raw_docs, source_tags=tags)
```

这就是本课的 Shipped artifact。输入可以是 `main.py` 内置的合成语料，也可以是从磁盘读入的一批 Common Crawl WET 文档。API 完全相同。

## Verify It

端到端测试验证了让这个管道真正有用的行为：

- 精确重复文档通过过滤后进入同一个重复簇；
- 高 Jaccard 的近似重复对与其原文至少共享一个 LSH band bucket；
- MinHash 签名一致率近似于精确 Jaccard 相似度；
- 升高或降低 temperature 会改变最终 mix 的来源计数；
- 配比器从不会从某个来源采样超过其可用数量；
- 去重对每个 cluster 只保留一个代表。

运行完整 demo：

```console
python3 main.py
```

你会看到一个存活率漏斗：有多少文档通过规则过滤、移除了多少重复簇、最终 mix 如何在来源间分配。

## Why This Matters for Engineers

- **语料质量是小模型的下限。** 架构上新闻很多，但用干净数据训练的 1B 模型在大多数下游任务上都能超过用原始抓取数据训练的 3B 模型。清洗是提升模型质量最便宜的方式。
- **清洗大部分是确定性工程。** 你可以写测试、版本化阈值、审计每篇文档被移除的原因。在以随机训练运行为主的领域里，这很少见。
- **去重防止记忆和逐字复述。** 近重复训练数据是模型引用版权文本或重复隐私政策的主要原因。LSH 让这种预防变得便宜。
- **配比设计是消融问题，不是拍脑袋。** Temperature scaling 给你一个旋钮。正确设置是在你的评估上获胜的那个，而不是感觉对的那个。
- **任何 RAG 语料库都使用同一套管道。** 当索引文档是垃圾、重复或样板时，检索增强生成系统会失败。同样的过滤、去重和配比逻辑也适用于更小规模。
- **存活率漏斗是你的调试工具。** 如果某条规则移除了 40% 的数据，先检查被移除的样本再信任它。计数会告诉你管道在哪里过于激进。

## Hints

- 如果规则过滤器移除了大部分代码数据，可以提高 `max_symbol_ratio` 或增加代码专用规则，而不是完全禁用过滤器。
- 如果一个重复簇只有两篇文档且长度差异很大，通常是不同正文外包裹着相同样板。在移除前检查 Jaccard 阈值。
- 如果 `temperature_mix` 对每个 temperature 返回相同的计数，说明 `total` 大于各来源可用文档数。配比器会按来源大小封顶，从而把所有权重压平。
- 当替换真实 Common Crawl 时，先在小样本上运行规则过滤器并阅读淘汰原因。本课的阈值针对英语网页调优；其他语言或领域需要不同的默认值。

## Key Terms

| 术语 | 人们常说 | 实际含义 |
|------|---------|---------|
| Common Crawl | "互联网" | 一家每月抓取网页的非营利机构；大多数 LLM 预训练的原始起点。 |
| WET | "提取出的文本" | Common Crawl 的纯文本格式，已移除标记，每条记录一篇文档。 |
| Gopher 规则 | "质量启发式" | 一套公开的长度、符号、大写、停用词和重复检查规则。 |
| C4 | "清洗后的 Common Crawl" | 用于训练 T5 的英语语料，通过激进的去重和启发式过滤构建。 |
| MinHash | "相似度哈希技巧" | 把集合压缩成固定签名，使签名一致率近似 Jaccard 相似度。 |
| LSH | "局部敏感哈希" | 把签名分带存入 bucket，让相似文档碰撞，只对候选对做精确验证。 |
| Candidate pair | "可能的重复" | 至少共享一个 LSH band bucket 并被检查精确相似度的两篇文档。 |
| Temperature mixing | "来源加权" | 按 $s_i^{1/T}$ 缩放可用来源大小，以锐化或平坦化训练配比。 |
| Survival funnel | "通过率计数" | 每个清洗阶段存活的文档数，用于调试阈值。 |
| Near-duplicate | "基本一样" | 仅在广告、页头或格式上不同但内容 shingle 高度重合的文档。 |
