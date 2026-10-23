# 分块策略对比

> 分块决定了检索器究竟能召回到什么。边界一旦切错，任何 embedding 模型、重排序器或 LLM 都无法在下游修复损伤。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 lessons 04 (embeddings), 06 (RAG), 07 (advanced RAG); Phase 19 Track B foundations (lessons 20-29)
**Time:** ~90 minutes

## 学习目标
- 从零实现五种分块策略：固定窗口、句子、递归切分、语义聚类、结构化 markdown 标题。
- 在带金标答案区间的 fixture 语料上测量 recall@k，并解释为什么一种策略在散文上胜出，另一种在技术文档上胜出。
- 读懂分块长度分布，识别每种策略引入的失败模式：孤立句子、符号中间切断、纯标题块、语义漂移。
- 不运行基准，仅通过检查三个属性为新语料选择默认策略：文档类型、平均段落长度、格式是否带显式结构。

## 问题

每个 RAG pipeline 都从把源文档切成片段开始：小到 embedding 模型能容纳，大到每个片段承载一个自包含的想法。切在哪里的选择不是超参数，它决定了检索器能返回内容的上限。

一个问「预算中止阈值长什么样」的查询，只有当持有该中止阈值的分块可达时才可能成功。如果固定窗口切分器把阈值数值与周围上下文切开了，embedding 就会漂移到另一个簇，BM25 分数下降，重排序器看到噪声，LLM 生成的答案就错了。2024 年的论文 "LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs" 测得，仅分块选择一项就带来 35 个百分点的绝对召回率波动。2025 年关于上下文分块标题的后续工作缩小了差距，但并未消除。

本节课并排构建五种策略，在带金标答案区间的 fixture 语料上运行，让你自己读召回数字。

## 概念

```mermaid
flowchart LR
  Doc[Source Document] --> S1[Fixed Window]
  Doc --> S2[Sentence]
  Doc --> S3[Recursive Split]
  Doc --> S4[Semantic Cluster]
  Doc --> S5[Structural Markdown]
  S1 --> Chunks1[Chunks]
  S2 --> Chunks2[Chunks]
  S3 --> Chunks3[Chunks]
  S4 --> Chunks4[Chunks]
  S5 --> Chunks5[Chunks]
  Chunks1 --> Index[Embedding Index]
  Chunks2 --> Index
  Chunks3 --> Index
  Chunks4 --> Index
  Chunks5 --> Index
  Index --> Eval[Recall@k vs Gold Spans]
```

### 固定窗口

暴力基线。每 N 个字符切一次。可选重叠，使在位置 N 被切断的句子完整出现在从 N - overlap 开始的块中。快、确定、边界处理糟糕。把它当对照，不要当默认。

### 句子

用正则或简单状态机在句子边界切分。把一句或多句打包进一个块，直到目标字符预算。不会切断单词。但仍会切断段落和章节。这是许多早期 RAG pipeline 的默认选择，对没有其他结构的散文是合理选择。

### 递归切分

2023 年前后库推广的层级策略。先尝试最强的分隔符（双换行、段落），失败则退到下一个（单换行），再到句子，最后到字符。当块符合预算时递归终止。对结构不一致的文档很强，因为它按区域自适应。

### 语义聚类

对每个句子做 embedding。把主题质心相近的连续句子聚成一簇。当运行中的相似度相对质心降到阈值以下时就切。边界反映含义而非字符。构建更慢且依赖 embedding 模型，但对段落内部切换主题的文档有韧性。

### 结构化 markdown 标题

对于带显式结构的文档（markdown、reStructuredText、RFC 风格编号章节），在标题边界切分。每个块由标题加上其下直到下一个同级或更高级标题之间的全部内容构成。每主题的块最小，但只在语料格式良好时可用。

### recall@k 如何度量边界选择

一条带金标的查询携带答案区间在源文档中的精确字符偏移。分块之后，你问：检索器返回的 top-k 分块中是否有任何一个与金标区间重叠？有，则该查询的 recall@k 为 1；没有则为 0。在查询集上取平均。对每种策略运行同样的评估，差距会告诉你哪种边界策略能在你手头的语料上存活。

```figure
ci-chunk-boundaries
```

## 构建它

`code/main.py` 实现：

- `fixed_window(text, size, overlap)` —— 基线。
- `sentence_chunks(text, target)` —— 简单句子打包器。
- `recursive_split(text, separators, target)` —— 层级递归。
- `semantic_chunks(text, similarity_threshold)` —— 在确定性模拟 embedding 之上做质心聚类。
- `structural_markdown(text)` —— 标题感知切分器。
- `mock_embed(text, dim)` —— 基于哈希的 embedding，让循环离线运行。
- `DenseIndex` —— 与 Phase 19 Track B 混合检索课相同的形状。
- `eval_recall(strategy, corpus, queries, k)` —— 对比循环。
- 一个 `main()`：在 fixture 语料上运行每种策略并打印 recall@k 表。

运行它：

```bash
python3 code/main.py
```

输出是一张小表，每个策略一行，每个 k 一列。句子策略在结构化 fixture 上表现最差。结构化 markdown 在 markdown fixture 上胜出。递归切分在混合 fixture 上表现稳健，因为递归会自适应。语义聚类在没有有用结构线索的散文 fixture 上胜出。

## 表格掩盖不了的失败模式

**孤立句子。** 句子打包会产生错过主题句的块。embedding 随后指向错误的簇。

**符号中间切断。** 在代码或 YAML 内部使用固定窗口会把标识符切成两半。两半都 embed 成噪声。

**纯标题块。** 结构化 markdown 会产生只有 `## Title` 的块。把它们过滤掉，或附上下一个块的首段。

**语义漂移。** 当语料主题高度一致时，语义聚类会切得过少。一个 5000 字符的块把许多具体答案塞进一个弥散的 embedding。把语义聚类与硬性字符上限结合。

**过期的 embedding。** 语义聚类使用 embedding 模型。更换模型也会改变分块。把分块模型与检索模型分开固定，或者一起重建索引。

## 不运行基准也能选默认策略

三个属性决定新语料的默认分块器。

| 属性 | 取值 | 默认 |
|------|------|------|
| 文档类型 | 无结构散文 | 递归切分，目标 800 |
| 文档类型 | Markdown / RFC / API 文档 | 结构化 markdown |
| 文档类型 | 代码 | AST 感知（超出范围；见 Phase 19 第 02 课） |
| 段落长度 | 长，单一主题 | 句子，目标 500 |
| 段落长度 | 短，主题混杂 | 语义，阈值 0.6 |

拿不准时，选递归切分。它是最强的单策略基线。

## 使用它

生产模式：

- 上线新 pipeline 前先跑评估；不要相信你所用库的默认策略。
- 每当更换 embedding 模型或语料构成时重跑评估；胜出者取决于语料。
- 把策略名持久化到每个分块的 metadata 中，以便日后归因回归。

## 交付它

第 69 课的 Track F 端到端 RAG 系统把这里选出的分块器作为第一阶段。第 68 课的评估工具读取的 recall@k 与本课 `eval_recall` 返回的形状相同。选出在你的语料上胜出的策略，向前传递。

## 练习

1. 增加第六种策略：用 `tiktoken` 按 token 窗口而非字符数切分。在相同 fixture 上与固定窗口对比。
2. 向散文 fixture 注入 30% 的代码块。重跑表格。解释为什么除结构化 markdown 外每种策略的召回都下降。
3. 把确定性 embedding 换为你项目真实提供商的 embedding。测量语义聚类召回的变化量。报告策略之间的差距是扩大还是缩小。
4. 为每个块增加 `summary` 字段：一句质心描述。把 summary 附加到块正文后重跑评估。测量召回提升。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Recall@k | 「我们拿到正确的块了吗？」 | top-k 中任一块与金标答案区间重叠的查询比例 |
| 分块重叠（Chunk overlap） | 「滑动窗口」 | 把上一块最后 N 个字符重新包含进下一块 |
| 结构切分器（Structural splitter） | 「标题感知分块」 | 在 H1/H2/H3 边界切分；标题文本属于块的一部分 |
| 语义分块器（Semantic chunker） | 「主题感知分块」 | 对句子做 embedding，按质心相似度聚类，在漂移处切分 |
| 质心漂移（Centroid drift） | 「主题切换」 | 运行均值与下一句的余弦相似度跌破阈值 |

## 延伸阅读

- [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs（arXiv 2406.15319）](https://arxiv.org/abs/2406.15319)
- [Anthropic：Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [LlamaIndex：面向生产 RAG 的分块策略](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
- Phase 11 第 06 课 —— RAG 基础
- Phase 11 第 07 课 —— 高级 RAG
- Phase 19 第 65 课 —— 对这里产出的分块排序的混合检索
- Phase 19 第 68 课 —— 在生产中为策略选择打分的评估工具
