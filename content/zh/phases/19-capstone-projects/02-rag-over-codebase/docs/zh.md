# 项目 02 — 代码库上的 RAG（跨仓库语义搜索）

> 2026 年，每一家严肃的工程组织都在运行理解语义而非只匹配字符串的内部代码搜索。Sourcegraph Amp、Cursor 的 codebase answers、Augment 的企业图谱、Aider 的 repomap、Pinterest 的内部 MCP —— 形态都一样。摄入多个仓库，用 tree-sitter 解析，对函数级和类级分块做 embedding，混合搜索，重排序，带引用作答。这个项目要求你构建一个能处理 10 个仓库、200 万行代码的系统，并且能在每次 git push 时完成增量重建索引。

**Type:** Capstone
**Languages:** Python (ingestion), TypeScript (API + UI)
**Prerequisites:** Phase 5 (NLP foundations), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure)
**Phases exercised:** P5 · P7 · P11 · P13 · P17
**Time:** 30 hours

## 问题

到 2026 年，每个前沿编码智能体都自带代码库检索层，因为仅靠上下文窗口解决不了跨仓库问题。Claude 的 100 万 token 上下文有帮助，但并不能消除排序检索的必要性。对原始分块做朴素的余弦搜索，会在生成代码、monorepo 重复以及很少被导入的长尾符号上污染结果。生产环境的答案是：对 AST 感知的分块做混合（dense + BM25）搜索，配合重排序器，背后再支撑一张符号引用图。

你要通过索引一个真实的仓库集群 —— 而不是一个教程仓库 —— 并测量 MRR@10、引用忠实度和增量新鲜度来学会它。失败模式是基础设施层面的：10 万文件的 monorepo、一次改动半个文件的 push、一个需要跨四个仓库才能正确回答的查询。

## 概念

AST 感知的摄入 pipeline 用 tree-sitter 解析每个文件，抽取函数节点和类节点，并在节点边界处切分块，而不是按固定 token 窗口切。每个分块有三种表示：稠密 embedding（Voyage-code-3 或 nomic-embed-code）、稀疏 BM25 词项，以及一段简短的自然语言摘要。摘要增加了第三种可检索模态 —— 用户问「X 是怎么授权的」，摘要里提到「authz」，即使代码里只有 `check_permission`。

检索是混合式的。一个查询同时触发稠密搜索和 BM25 搜索，合并 top-k，把并集交给 cross-encoder 重排序器（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。重排后的列表送入长上下文合成器（带 prompt caching 的 Claude Sonnet 4.7，或自托管的 Llama 3.3 70B），并要求每条论断都按文件和行范围给出引用。没有引用的回答会被后置过滤器拒掉。

增量新鲜度是基础设施层面的难题。git push 会触发一次 diff：哪些文件变了，哪些符号变了。只有受影响的分块重新做 embedding。受影响的跨文件符号边（导入、方法调用）重新计算。这样索引保持一致，而不必在每次提交时重新处理 200 万行。

## 架构

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 技术栈

- 解析：tree-sitter，带 17 种语言语法（Python、TS、Rust、Go、Java、C++ 等）
- 稠密 embedding：Voyage-code-3（托管）或 nomic-embed-code-v1.5（自托管），备选 bge-code-v1
- 稀疏索引：Tantivy（Rust），使用 BM25F，在符号名与符号体之间做字段加权
- 向量数据库：带混合搜索的 Qdrant 1.12；向量规模低于 5000 万的团队可用 pgvector + pgvectorscale
- 分块摘要模型：Claude Haiku 4.5 或 Gemini 2.5 Flash，启用 prompt caching
- 重排序器：Cohere rerank-3 或自托管的 bge-reranker-v2-gemma-2b
- 编排：摄入用 LlamaIndex Workflows，查询智能体用 LangGraph
- 合成器：Claude Sonnet 4.7（100 万上下文），启用 prompt caching
- 符号图：Neo4j（托管）或 kuzu（嵌入式），存储导入和调用边
- 可观测性：每个检索 + 合成步骤都有 Langfuse span

```figure
ce-hybrid-retrieval
```

## 构建它

1. **摄入遍历器。** 在每个 push hook 上遍历 git 历史。收集变更文件。对每个文件用 tree-sitter 解析，抽取函数节点和类节点及其完整源码跨度。产出分块记录 `{repo, path, start_line, end_line, symbol, body}`。

2. **分块摘要器。** 把分块批量送入 Haiku 4.5 调用，对系统前导语启用 prompt caching。Prompt：「用一句话总结这个函数，说明其公开契约和副作用。」把摘要与分块一起存储。

3. **Embedding 池。** 两条并行队列：稠密（Voyage-code-3，batch 128）和摘要（同一模型，但输入是摘要字符串）。把向量写入 Qdrant，payload 为 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 索引。** 字段加权的 Tantivy 索引：符号名权重 4，符号体权重 1，摘要权重 2。这样既能支持「找到名为 X 的函数」，也能支持「找到做 X 的函数」。

5. **符号图。** 为每个分块记录边：导入（本文件使用来自仓库 Z 的符号 Y）、调用（本函数调用类 C 上的方法 M）、继承。存入 kuzu。查询时用于跨仓库边界扩展检索。

6. **查询智能体。** 一个含三个节点的 LangGraph。`retrieve` 并行触发 dense + BM25，按 (repo, path, symbol) 去重。`rerank` 对 top-50 运行 cross-encoder，保留 top-10。`synth` 把重排后的分块放进上下文调用 Claude Sonnet 4.7，缓存系统 prompt，要求提供 file:line 引用。

7. **引用强制。** 解析模型输出；任何没有 `(repo/path:start-end)` 锚点的论断都会被标记为重新询问或丢弃。只把带引用的回答返回给用户。

8. **增量重建索引。** 每次 webhook 时计算符号级 diff。只对文本发生变化的分块重新做 embedding。对导入发生变化的分块重新计算符号边。衡量指标：在 200 万行代码的仓库集群中，50 个文件的 push 在 60 秒内完成重建索引。

9. **评测。** 为 100 个跨仓库问题标注 gold file:line 答案。测量 MRR@10、nDCG@10、引用忠实度（带可验证锚点的论断比例）以及 p50/p99 延迟。

## 使用它

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 交付它

交付物 skill 为 `outputs/skill-codebase-rag.md`。给定一个仓库语料库，它会搭起摄入 pipeline、混合索引和查询智能体，并为任何跨仓库问题返回带引用的答案。评分标准：

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 检索质量 | 在 100 个问题的留出集上的 MRR@10 和 nDCG@10 |
| 20 | 引用忠实度 | 回答论断中带可验证 file:line 锚点的比例 |
| 20 | 延迟与规模 | 在已索引语料规模下、10k QPS 时的 p95 查询延迟 |
| 20 | 增量索引正确性 | 50 文件提交时，从 git push 到可搜索的耗时 |
| 15 | 用户体验与答案格式 | 引用可点击性、片段预览、追问入口 |
| **100** | | |

## 练习

1. 把 Voyage-code-3 换成自托管的 nomic-embed-code。测量 MRR@10 的差值。报告开启重排序后差距是否缩小。

2. 向语料库注入 20% 的生成代码（LLM 产出的样板代码）并重新评测。观察检索被污染的现象。在 payload 中加一个「generated」标记，并降低这些命中的权重。

3. 在你的语料规模下，对 Qdrant 混合搜索与 pgvector + pgvectorscale 做基准测试。报告 batch size 为 1 时的 p99。

4. 加入基于采样的漂移检查：每周重跑 100 题评测。MRR@10 下降超过 5% 时告警。

5. 扩展到跨语言符号解析：一个 Python 函数通过 gRPC 调用 Go 服务。用符号图把它们关联起来。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| AST 感知分块 | 「函数级切分」 | 在 tree-sitter 节点边界切分代码，而不是按固定 token 窗口 |
| 混合搜索 | 「稠密 + 稀疏」 | 并行运行 BM25 和向量搜索，合并 top-k，再重排序 |
| Cross-encoder 重排序 | 「第二阶段排序」 | 把每个 (查询, 候选) 对放在一起打分的模型，比余弦更准确 |
| Prompt caching | 「缓存系统 prompt」 | 2026 年 Claude / OpenAI 的功能，对重复的前缀 token 最多打一折 |
| 符号图 | 「代码图」 | 跨文件和跨仓库的导入、调用、继承边 |
| 引用忠实度 | 「有据可查的答案比例」 | 用户能通过点击锚点并阅读所指跨度来验证的论断比例 |
| 增量重建索引 | 「push 到可搜索的耗时」 | 从 git push 到变更符号可被查询的实际耗时 |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) —— 生产级跨仓库代码智能
- [Sourcegraph Cody RAG 架构](https://sourcegraph.com/blog/how-cody-understands-your-codebase) —— 本项目的参考深度剖析
- [Aider repo-map](https://aider.chat/docs/repomap.html) —— 基于 tree-sitter 的仓库排序视图
- [Augment Code 企业图谱](https://www.augmentcode.com) —— 商业化的符号图 RAG
- [Qdrant 混合搜索文档](https://qdrant.tech/documentation/concepts/hybrid-queries/) —— 参考实现
- [Voyage AI 代码 embedding](https://docs.voyageai.com/docs/embeddings) —— Voyage-code-3 细节
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) —— cross-encoder 参考
- [Pinterest MCP 内部搜索](https://medium.com/pinterest-engineering) —— 内部平台参考
