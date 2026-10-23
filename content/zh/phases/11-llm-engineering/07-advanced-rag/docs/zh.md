# 高级 RAG（分块、重排序与混合搜索）

> 基础 RAG 检索最相似的前 k 个块。这对简单问题有效，但面对多跳推理、模糊查询和大型语料库就会崩溃。高级 RAG 是"能在 10 份文档上跑通的 demo"与"能在 1000 万份文档上运行的系统"之间的差别。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11, Lesson 06 (RAG)
**Time:** ~90 minutes
**Related:** Phase 5 · 23（Chunking Strategies for RAG）涵盖全部六种分块算法——递归、语义、句子、父文档、延迟分块、上下文检索——并附带 Vectara/Anthropic 基准。本课在其上继续深入：混合搜索、重排序、查询变换。

## 学习目标

- 实现保留文档结构与上下文的高级分块策略（语义、递归、父子）
- 构建混合搜索流水线，结合 BM25 关键词匹配、语义向量搜索和交叉编码器重排序器
- 应用查询变换技术（HyDE、多查询、步退查询）以改善模糊或复杂问题的检索
- 诊断并修复常见的 RAG 故障：检索到错误的块、答案不在上下文中、多跳推理中断

## 问题

你在第 06 课构建了一条基础 RAG 流水线。它能在小型语料库上回答直截了当的问题。现在试试这些：

**模糊查询**："上个季度的收入是多少？"语义搜索返回的是关于收入战略、收入预测以及 CFO 对收入增长看法的块。它们都与"收入"这个词语义相似，但没有一个包含实际数字。正确的块写着"2025 年 Q3 为 4720 万美元"，但用的是"收益"（earnings）而不是"收入"（revenue）。embedding 模型认为"收入战略"比"Q3 收益为 4720 万美元"更接近查询。

**多跳问题**："哪个团队的客户满意度得分提升最大？"这需要找出每个团队的满意度得分、逐一比较并确定最大值。没有任何单个块包含答案。信息散落在各团队的报告中。

**大型语料库问题**：你有 200 万个块。正确答案在第 1,847,293 块。你的 top-5 检索拉回的是第 14、第 89,201、第 1,200,000、第 44 和第 901,333 块。它们在 embedding 空间中很接近，但没有一个包含答案。在这个规模上，近似最近邻搜索引入的误差足以把相关结果挤出 top-k。

基础 RAG 失败，是因为向量相似度并不等于相关性。一个块可以在语义上与查询相似，却无助于回答它。高级 RAG 用四项技术解决这个问题：混合搜索（加入关键词匹配）、重排序（更仔细地对候选打分）、查询变换（在搜索前修正查询），以及更好的分块（在正确的粒度上检索）。

## 概念

### 混合搜索：语义 + 关键词

语义搜索（向量相似度）擅长理解含义。"How do I cancel my subscription?" 能匹配 "Steps to terminate your plan"，尽管两者没有共用的词。但它会漏掉精确匹配。如果 embedding 模型把 "E-4021" 当成噪声，查询 "Error code E-4021" 可能匹配不到包含 "E-4021" 的块。

关键词搜索（BM25）恰恰相反。它擅长精确匹配。"E-4021" 完美匹配。但如果文档写的是 "terminate your plan"，"cancel my subscription" 会返回零结果。

混合搜索同时运行两者，然后合并结果。

**BM25**（Best Matching 25）是标准的关键词搜索算法。自 1990 年代以来，它一直是搜索引擎的支柱。公式如下：

```
BM25(q, d) = sum over terms t in q:
    IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
```

其中 tf(t,d) 是词 t 在文档 d 中的词频，IDF(t) 是逆文档频率，|d| 是文档长度，avgdl 是平均文档长度，k1 控制词频饱和（默认 1.2），b 控制长度归一化（默认 0.75）。

通俗地说：当文档包含查询词（尤其是罕见词）时，BM25 给更高的分，但对重复词有递减的回报。一个出现 50 次 "revenue" 的文档，并不会比只出现一次的相关 50 倍。

### 倒数排名融合（RRF）

你有两个排序列表：一个来自向量搜索，一个来自 BM25。如何合并？倒数排名融合（Reciprocal Rank Fusion）是标准做法。

```
RRF_score(d) = sum over rankings R:
    1 / (k + rank_R(d))
```

其中 k 是常数（通常为 60），用于防止排名第一的结果一家独大。

一个在向量搜索中排第 1、在 BM25 中排第 5 的文档得分：1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318

一个在向量搜索中排第 3、在 BM25 中排第 2 的文档得分：1/(60+3) + 1/(60+2) = 0.0159 + 0.0161 = 0.0320

RRF 自然地平衡了两种信号。在两个列表中排名都高的文档得到最佳分数。在一个列表中排第 1、却缺席另一个列表的文档得到一个中等分数。它之所以稳健，是因为使用排名而非原始分数，所以两个系统之间的分数分布差异无关紧要。

### 重排序

检索（无论是向量、关键词还是混合）快但不精确。它使用双编码器（bi-encoder）：查询和每份文档分别独立嵌入，然后比较。embedding 计算一次并缓存，因此可以扩展到数百万份文档。

重排序使用交叉编码器（cross-encoder）：查询和候选文档一起送入模型，由模型输出相关性分数。模型同时看到两段文本，能捕捉它们之间细粒度的交互。交叉编码器能理解 "What were Q3 earnings?" 与包含 "$47.2M in Q3" 的块高度相关，即使双编码器漏掉了这层联系。

权衡在于：交叉编码器比双编码器慢 100-1000 倍，因为它们要联合处理查询-文档对。你无法为一百万份文档预先计算交叉编码器分数。解决方案是：先检索一个更大的候选集（混合搜索取 top-50），再用交叉编码器重排，得到最终的 top-5。

```mermaid
graph LR
    Q["Query"] --> H["Hybrid Search"]
    H --> C50["Top 50 candidates"]
    C50 --> RR["Cross-Encoder Reranker"]
    RR --> C5["Top 5 final results"]
    C5 --> P["Build prompt"]
    P --> LLM["Generate answer"]
```

常见的重排序模型（2026 年阵容）：
- Cohere Rerank 3.5：托管 API，多语言，在混合语料上召回提升最佳
- Voyage rerank-2.5：托管 API，托管选项中延迟最低
- Jina-Reranker-v2 Multilingual：开放权重，支持 100 多种语言
- bge-reranker-v2-m3：开放权重，强大的基线
- cross-encoder/ms-marco-MiniLM-L-6-v2：开放权重，可在 CPU 上运行，适合原型开发
- ColBERTv2 / Jina-ColBERT-v2：晚交互多向量重排序器——打分时是 $O(\text{tokens})$ 而不是 $O(\text{docs})$

### 查询变换

有时问题不在检索，而在查询本身。"What was that thing about the new policy change?" 是一个糟糕的搜索查询。它不包含任何具体的词，embedding 很模糊。任何检索系统都无法从中找到正确的文档。

**查询重写**：把用户的查询改写成更好的搜索查询。LLM 可以做到这一点：

```
User: "What was that thing about the new policy change?"
Rewritten: "Recent policy changes and updates"
```

**HyDE（Hypothetical Document Embeddings，假设性文档嵌入）**：不用查询去搜索，而是生成一个假设答案，对它做 embedding，再去搜索相似的真实文档。

```
Query: "What is the refund policy for enterprise?"
Hypothetical answer: "Enterprise customers are eligible for a full refund
within 60 days of purchase. Refunds are pro-rated based on the remaining
subscription period and processed within 5-7 business days."
```

对假设答案做 embedding，并搜索与它相似的真实文档。直觉是：在 embedding 空间中，假设答案比原始问题更接近真实答案。问题和答案具有不同的语言结构。通过生成假设答案，你在 embedding 中弥合了"问题空间"与"答案空间"之间的鸿沟。

HyDE 在检索之前增加一次 LLM 调用，使延迟增加 500-2000ms。当原始查询的检索质量很差时，这是值得的。

### 父子分块

标准分块迫使你权衡：小块用于精确检索，大块用于提供足够的上下文。父子分块消除了这种权衡。

对小块（128 token）建立索引用于检索。当一个小块被检索到，返回它的父块（512 token）用于提示。小块精确匹配查询，父块为 LLM 提供足够的上下文来生成好的答案。

```mermaid
graph TD
    P["Parent chunk (512 tokens)<br/>Full section about refund policy"]
    C1["Child chunk (128 tokens)<br/>Standard plan: 30-day refund"]
    C2["Child chunk (128 tokens)<br/>Enterprise: 60-day pro-rated"]
    C3["Child chunk (128 tokens)<br/>Processing time: 5-7 days"]
    C4["Child chunk (128 tokens)<br/>How to submit a request"]

    P --> C1
    P --> C2
    P --> C3
    P --> C4

    Q["Query: enterprise refund?"] -.->|"matches child"| C2
    C2 -.->|"return parent"| P
```

查询 "enterprise refund?" 精确匹配子块 C2。但提示收到的是完整的父块 P，其中包含关于处理时间和提交流程的周边上下文。

### 元数据过滤

在运行向量搜索之前，按元数据过滤语料库：日期、来源、类别、作者、语言。这缩小了搜索空间，避免无关结果。

"What changed in the security policy last month?" 应该只搜索安全类别中最近 30 天的文档。没有元数据过滤，你会搜索整个语料库，可能检索到一份两年前的安全文档，而它恰好语义相似。

生产 RAG 系统会把元数据与每个块一起存储：来源文档、创建日期、类别、作者、版本。向量数据库支持在相似度搜索之前按元数据预过滤，这对大规模场景下的性能至关重要。

### 评估

你构建了一个 RAG 系统。怎么知道它是否有效？三个指标：

**检索相关性（Recall@k）**：对于一组已知相关文档的测试问题，相关文档出现在 top-k 结果中的比例是多少？如果某个问题的答案在第 47 块，第 47 块会出现在 top-5 中吗？

**忠实度**：生成的答案是否以检索到的文档为依据？如果检索到的块写的是"60 天退款窗口"，而模型说"90 天退款窗口"，那就是忠实度失败。模型明明有正确的上下文，却产生了幻觉。

**答案正确性**：生成的答案是否与预期答案一致？这是端到端的指标，它综合了检索质量和生成质量。

一个简单的忠实度检查：取生成答案中的每个断言，验证它（实质上）出现在检索到的块中。如果答案包含任何检索块中都没有的事实，那很可能是幻觉。

```mermaid
graph TD
    subgraph "Evaluation Framework"
        Q["Test questions<br/>+ expected answers<br/>+ relevant doc IDs"]
        Q --> Ret["Retrieval evaluation<br/>Recall@k: are right<br/>docs retrieved?"]
        Q --> Faith["Faithfulness evaluation<br/>Is answer grounded<br/>in retrieved docs?"]
        Q --> Correct["Correctness evaluation<br/>Does answer match<br/>expected answer?"]
    end
```

```figure
agentic-rag-loop
```

## 构建它

### 第 1 步：BM25 实现

```python
import math
from collections import Counter

class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_lengths = []
        self.avg_dl = 0
        self.doc_freqs = {}
        self.n_docs = 0

    def index(self, documents):
        self.docs = documents
        self.n_docs = len(documents)
        self.doc_lengths = []
        self.doc_freqs = {}

        for doc in documents:
            words = doc.lower().split()
            self.doc_lengths.append(len(words))
            unique_words = set(words)
            for word in unique_words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        self.avg_dl = sum(self.doc_lengths) / self.n_docs if self.n_docs else 1

    def score(self, query, doc_idx):
        query_words = query.lower().split()
        doc_words = self.docs[doc_idx].lower().split()
        doc_len = self.doc_lengths[doc_idx]
        word_counts = Counter(doc_words)
        score = 0.0

        for term in query_words:
            if term not in word_counts:
                continue
            tf = word_counts[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * numerator / denominator

        return score

    def search(self, query, top_k=10):
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

### 第 2 步：倒数排名融合

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused
```

### 第 3 步：混合搜索流水线

```python
def hybrid_search(query, chunks, vector_embeddings, vocab, idf, bm25_index, top_k=5, fusion_k=60):
    query_emb = tfidf_embed(query, vocab, idf)
    vector_results = search(query_emb, vector_embeddings, top_k=top_k * 3)
    bm25_results = bm25_index.search(query, top_k=top_k * 3)
    fused = reciprocal_rank_fusion([vector_results, bm25_results], k=fusion_k)
    return fused[:top_k]
```

### 第 4 步：简单重排序器

生产环境中你会使用交叉编码器模型。这里我们构建一个重排序器，利用词重叠、词重要性和短语匹配来对查询-文档相关性打分。

```python
def rerank(query, candidates, chunks):
    query_words = set(query.lower().split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                  "why", "when", "where", "do", "does", "for", "of", "in", "to",
                  "and", "or", "on", "at", "by", "it", "its", "this", "that",
                  "with", "from", "be", "has", "have", "had", "not", "but"}
    query_terms = query_words - stop_words

    scored = []
    for doc_id, initial_score in candidates:
        chunk = chunks[doc_id].lower()
        chunk_words = set(chunk.split())

        term_overlap = len(query_terms & chunk_words)

        query_bigrams = set()
        q_list = [w for w in query.lower().split() if w not in stop_words]
        for i in range(len(q_list) - 1):
            query_bigrams.add(q_list[i] + " " + q_list[i + 1])
        bigram_matches = sum(1 for bg in query_bigrams if bg in chunk)

        position_boost = 0
        for term in query_terms:
            pos = chunk.find(term)
            if pos != -1 and pos < len(chunk) // 3:
                position_boost += 0.5

        rerank_score = (
            term_overlap * 1.0
            + bigram_matches * 2.0
            + position_boost
            + initial_score * 5.0
        )
        scored.append((doc_id, rerank_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
```

### 第 5 步：HyDE（假设性文档嵌入）

```python
def hyde_generate_hypothesis(query):
    templates = {
        "what": "The answer to '{query}' is as follows: Based on our documentation, {topic} involves specific policies and procedures that define how the process works.",
        "how": "To address '{query}': The process involves several steps. First, you need to initiate the request. Then, the system processes it according to the defined rules.",
        "default": "Regarding '{query}': Our records indicate specific details and policies related to this topic that provide a comprehensive answer."
    }
    query_lower = query.lower()
    if query_lower.startswith("what"):
        template = templates["what"]
    elif query_lower.startswith("how"):
        template = templates["how"]
    else:
        template = templates["default"]

    topic_words = [w for w in query.lower().split()
                   if w not in {"what", "is", "the", "how", "do", "does", "a", "an",
                                "for", "of", "to", "in", "on", "at", "by", "and", "or"}]
    topic = " ".join(topic_words) if topic_words else "this topic"

    return template.format(query=query, topic=topic)


def hyde_search(query, chunks, vector_embeddings, vocab, idf, top_k=5):
    hypothesis = hyde_generate_hypothesis(query)
    hypothesis_emb = tfidf_embed(hypothesis, vocab, idf)
    results = search(hypothesis_emb, vector_embeddings, top_k)
    return results, hypothesis
```

### 第 6 步：父子分块

```python
def create_parent_child_chunks(text, parent_size=200, child_size=50):
    words = text.split()
    parents = []
    children = []
    child_to_parent = {}

    parent_idx = 0
    start = 0
    while start < len(words):
        parent_end = min(start + parent_size, len(words))
        parent_text = " ".join(words[start:parent_end])
        parents.append(parent_text)

        child_start = start
        while child_start < parent_end:
            child_end = min(child_start + child_size, parent_end)
            child_text = " ".join(words[child_start:child_end])
            child_idx = len(children)
            children.append(child_text)
            child_to_parent[child_idx] = parent_idx
            child_start += child_size

        parent_idx += 1
        start += parent_size

    return parents, children, child_to_parent
```

### 第 7 步：忠实度评估

```python
def evaluate_faithfulness(answer, retrieved_chunks):
    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not answer_sentences:
        return 1.0, []

    grounded = 0
    ungrounded = []
    context = " ".join(retrieved_chunks).lower()

    for sentence in answer_sentences:
        words = set(sentence.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                      "to", "of", "in", "for", "on", "at", "by", "it", "this", "that"}
        content_words = words - stop_words
        if not content_words:
            grounded += 1
            continue

        matched = sum(1 for w in content_words if w in context)
        ratio = matched / len(content_words) if content_words else 0

        if ratio >= 0.5:
            grounded += 1
        else:
            ungrounded.append(sentence)

    score = grounded / len(answer_sentences) if answer_sentences else 1.0
    return score, ungrounded


def evaluate_retrieval_recall(queries_with_relevant, retrieval_fn, k=5):
    total_recall = 0.0
    results = []

    for query, relevant_indices in queries_with_relevant:
        retrieved = retrieval_fn(query, k)
        retrieved_indices = set(idx for idx, _ in retrieved)
        relevant_set = set(relevant_indices)
        hits = len(retrieved_indices & relevant_set)
        recall = hits / len(relevant_set) if relevant_set else 1.0
        total_recall += recall
        results.append({
            "query": query,
            "recall": recall,
            "hits": hits,
            "total_relevant": len(relevant_set)
        })

    avg_recall = total_recall / len(queries_with_relevant) if queries_with_relevant else 0
    return avg_recall, results
```

## 使用它

用真实的交叉编码器做重排序：

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_with_cross_encoder(query, candidates, chunks, top_k=5):
    pairs = [(query, chunks[doc_id]) for doc_id, _ in candidates]
    scores = reranker.predict(pairs)
    scored = list(zip([doc_id for doc_id, _ in candidates], scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
```

使用 Cohere 的托管重排序器：

```python
import cohere

co = cohere.Client()

def rerank_with_cohere(query, candidates, chunks, top_k=5):
    docs = [chunks[doc_id] for doc_id, _ in candidates]
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=top_k
    )
    return [(candidates[r.index][0], r.relevance_score) for r in response.results]
```

用真实 LLM 实现 HyDE：

```python
import anthropic

client = anthropic.Anthropic()

def hyde_with_llm(query):
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"Write a short paragraph that would be a good answer to this question. Do not say you don't know. Just write what the answer would look like.\n\nQuestion: {query}"
        }]
    )
    return response.content[0].text
```

用 Weaviate 做生产级混合搜索：

```python
import weaviate

client = weaviate.connect_to_local()

collection = client.collections.get("Documents")
response = collection.query.hybrid(
    query="enterprise refund policy",
    alpha=0.5,
    limit=10
)
```

alpha 参数控制平衡：0.0 = 纯关键词（BM25），1.0 = 纯向量，0.5 = 等权。大多数生产系统使用 0.3 到 0.7 之间的 alpha。

## 交付它

本课产出：
- `outputs/prompt-advanced-rag-debugger.md`——一个用于诊断和修复 RAG 质量问题的提示
- `outputs/skill-advanced-rag.md`——一个用于构建带混合搜索和重排序的生产级 RAG 的技能

## 练习

1. 在示例文档上比较 BM25、向量搜索和混合搜索。对 5 个测试查询中的每一个，记录哪种方法把最相关的块放在第 1 位。混合搜索应至少在其中 3 个上胜出。

2. 实现元数据过滤。为每份文档添加 "category" 字段（security、billing、api、product）。在运行向量搜索之前，把块过滤到只保留相关类别。用 "What encryption is used?" 测试，验证它只搜索 security 类别的块。

3. 使用第 06 课的简单生成函数构建完整的 HyDE 流水线。在全部 5 个测试查询上比较直接查询搜索与 HyDE 搜索的检索质量（top-3 相关性）。HyDE 应能改善模糊查询的结果。

4. 在示例文档上实现父子分块策略。使用 child_size=30、parent_size=100。用子块搜索，但在提示中返回父块。把生成的答案与 chunk_size=50 的标准分块相比较。

5. 创建一个评估数据集：10 个已知答案块的问题。为以下方案测量 Recall@3、Recall@5 和 Recall@10：（a）仅向量搜索，（b）仅 BM25，（c）混合搜索，（d）混合 + 重排序。绘制结果，找出重排序帮助最大的地方。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| BM25 | "关键词搜索" | 一种概率排序算法，按词频、逆文档频率和文档长度归一化对文档打分 |
| 混合搜索 | "两全其美" | 并行运行语义（向量）搜索和关键词（BM25）搜索，然后用排名融合合并结果 |
| 倒数排名融合 | "合并排序列表" | 对每份文档在所有列表中的 1/(k + rank) 求和，从而合并多个排序列表 |
| 重排序 | "第二遍打分" | 用更昂贵的交叉编码器模型对初始检索得到的候选集重新打分 |
| 交叉编码器 | "联合的查询-文档模型" | 把查询和文档作为单个输入、输出相关性分数的模型；比双编码器更准确，但对全语料搜索来说太慢 |
| 双编码器 | "独立 embedding 模型" | 分别对查询和文档做 embedding 的模型；因为 embedding 可预计算而速度快，但准确率不如交叉编码器 |
| HyDE | "用假答案来搜索" | 为查询生成一个假设答案，对它做 embedding，再搜索与它相似的真实文档 |
| 父子分块 | "小块搜索，大块上下文" | 索引小块用于精确检索，但返回更大的父块以提供足够的上下文 |
| 元数据过滤 | "先缩小范围再搜索" | 在运行向量搜索之前按属性（日期、来源、类别）过滤文档，以缩小搜索空间 |
| 忠实度 | "它是否守住了依据" | 生成的答案是否有检索到的文档支持，而不是模型从训练数据中幻觉出来的 |

## 延伸阅读

- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond"（2009）——BM25 的权威参考，解释了公式背后的概率论基础
- Cormack 等人，"Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods"（2009）——RRF 的原始论文，表明它优于更复杂的融合方法
- Gao 等人，"Precise Zero-Shot Dense Retrieval without Relevance Labels"（2022）——HyDE 论文，证明假设性文档 embedding 无需任何训练数据就能改善检索
- Nogueira & Cho, "Passage Re-ranking with BERT"（2019）——表明在 BM25 之上做交叉编码器重排序能显著提升检索质量
- [Khattab 等人，"DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"（2023）](https://arxiv.org/abs/2310.03714)——把提示构造和权重选择视为检索流水线上的优化问题；读这篇了解如何"编程 LLM"而不是"提示 LLM"。
- [Edge 等人，"From Local to Global: A Graph RAG Approach to Query-Focused Summarization"（Microsoft Research 2024）](https://arxiv.org/abs/2404.16130)——GraphRAG 论文：实体-关系抽取 + Leiden 社区检测用于查询聚焦摘要；全局检索与局部检索的区分。
- [Asai 等人，"Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"（ICLR 2024）](https://arxiv.org/abs/2310.11511)——带反思 token 的自我评估 RAG；超越静态"先检索再生成"的智能体式前沿。
- [LangChain Query Construction 博客](https://blog.langchain.dev/query-construction/)——如何把自然语言查询转换为结构化数据库查询（Text-to-SQL、Cypher），作为检索前的步骤。
