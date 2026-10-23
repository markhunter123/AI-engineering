# 项目 08 —— 面向受监管垂直行业的生产级 RAG 聊天机器人

> Harvey、Glean、Mendable 和 LlamaCloud 在 2026 年跑的都是同一套生产形态。用 docling 或 Unstructured 做摄取，用 ColPali 处理视觉内容。混合检索。用 bge-reranker-v2-gemma 重排序。用 Claude Sonnet 4.7 做合成，靠 prompt 缓存把命中率做到 60-80%。用 Llama Guard 4 和 NeMo Guardrails 做防护。用 Langfuse 和 Phoenix 做观测。用 RAGAS 在 200 题的黄金集上打分。在一个受监管领域（法律、临床、保险）里做一个，而项目的标准是通过黄金集、红队测试和漂移仪表盘。

**Type:** Capstone
**Languages:** Python (pipeline + API), TypeScript (chat UI)
**Prerequisites:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P5 · P7 · P11 · P12 · P17 · P18
**Time:** 30 hours

## 问题

受监管领域的 RAG（法律合同、临床试验方案、保险保单）是 2026 年交付量最大的生产形态，因为 ROI 显而易见，风险又切实具体。Harvey（Allen & Overy）为法律行业打造了它。Mendable 交付的是开发者文档那一款。Glean 覆盖企业搜索。模式是这样的：高保真摄取、带重排序的混合检索、带引用强制和 prompt 缓存的合成、多层安全防护，以及持续的漂移监控。

难的地方不在模型。难的是感知辖区的合规（HIPAA、GDPR、SOC2）、引用级可审计性、成本控制（命中率高时 prompt 缓存能带来 60-90% 的折扣）、通过 RAGAS 忠实度做幻觉检测，以及源文档更新而索引没跟上时的漂移检测。这个项目要求你在 200 题的黄金集上把这些全部交付，并附上一套红队测试。

## 概念

流水线分两侧。**摄取**：docling 或 Unstructured 解析结构化文档；ColPali 处理视觉内容丰富的文档；分块会带上摘要、标签和基于角色的访问标签。向量进入 pgvector + pgvectorscale（5000 万向量以下）或 Qdrant Cloud；稀疏 BM25 并行运行。**对话**：LangGraph 负责记忆和多轮；每个查询走混合检索，用 bge-reranker-v2-gemma-2b 重排序，用 Claude Sonnet 4.7 合成（prompt 已缓存），输出经 Llama Guard 4 和 NeMo Guardrails 过滤，最后返回锚定引用的回答。

评估栈有四层。**黄金集**（200 条带引用的标注问答）衡量正确性。**红队**（越狱、PII 提取尝试、域外问题）衡量安全性。**RAGAS** 逐轮自动计算忠实度 / 答案相关性 / 上下文精确率。**漂移仪表盘**（Arize Phoenix）每周监视检索质量和幻觉分数。

prompt 缓存是成本杠杆。Claude 4.5+ 和 GPT-5+ 支持缓存系统提示 + 检索到的上下文。命中率 60-80% 时，每次查询的成本会下降 3-5 倍。流水线必须围绕稳定前缀来设计（系统提示 + 重排序后的上下文放在最前），才能获得高缓存命中率。

## 架构

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## 技术栈

- 摄取：结构化文档用 Unstructured.io 或 docling；视觉内容丰富的 PDF 用 ColPali
- 向量数据库：5000 万向量以下用 pgvector + pgvectorscale；否则用 Qdrant Cloud
- 稀疏检索：Tantivy BM25，带字段权重
- 编排：LlamaIndex Workflows（摄取）+ LangGraph（对话）
- 重排序器：自托管 bge-reranker-v2-gemma-2b，或托管的 Voyage rerank-2
- LLM：Claude Sonnet 4.7，启用 prompt 缓存；回退到自托管的 Llama 3.3 70B
- 评估：在线用 RAGAS 0.2，幻觉和越狱套件用 DeepEval
- 可观测性：自托管 Langfuse 带标注队列；Arize Phoenix 做漂移
- 防护栏：Llama Guard 4 输入/输出分类器、NeMo Guardrails v0.12 策略、Presidio PII 清洗
- 合规：分块上的基于角色的访问标签；GDPR/HIPAA 辖区标签

```figure
canary-rollout
```

## 构建它

1. **摄取。** 用 Unstructured 或 docling 解析语料（认真做的话是 1000-10000 份文档）。扫描件或视觉内容密集的页面走 ColPali。产出带摘要、角色标签、辖区标签的分块。

2. **索引。** 稠密 embedding（Voyage-3 或 Nomic-embed-v2）进入 pgvector + pgvectorscale。BM25 侧索引走 Tantivy。角色和辖区过滤作为 payload。

3. **混合检索。** 先按角色+辖区过滤；然后并行跑稠密和 BM25；用倒数排名融合合并；前 20 条交给重排序器；前 5 条交给合成。

4. **带 prompt 缓存的合成。** 系统提示 + 静态策略放在缓存头；重排序后的上下文作为缓存延伸；用户问题作为不缓存的尾部。稳定状态下目标缓存命中率 60-80%。

5. **防护栏。** 输入侧用 Llama Guard 4；NeMo Guardrails 的规则拦截域外问题或策略禁止的话题；Presidio 清洗输出中意外出现的 PII；引用强制作为后置过滤。

6. **黄金集。** 200 对由领域专家标注的问答（答案、引用）。按精确引用匹配、答案正确性、忠实度（RAGAS）给智能体打分。

7. **红队。** 50 条对抗提示：越狱（PAIR、TAP）、PII 外泄尝试、域外问题、跨辖区泄漏。按通过/失败和严重程度打分。

8. **漂移仪表盘。** Arize Phoenix 每周跟踪检索质量（nDCG、引用忠实度）。下降 5% 时告警。

9. **成本报告。** Langfuse：prompt 缓存命中率、每次查询的 token 数、按阶段拆分的 $/查询。

## 使用它

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 交付它

`outputs/skill-production-rag.md` 描述了交付物。一个部署上线、带合规标签、通过评分标准的受监管领域聊天机器人，并有实时漂移监控。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | RAGAS 忠实度 + 答案相关性 | 在黄金集（200 条问答）上的在线分数 |
| 20 | 引用正确性 | 带可验证来源锚点的回答比例 |
| 20 | 防护栏覆盖 | Llama Guard 4 通过率 + 越狱套件结果 |
| 20 | 成本 / 延迟工程 | prompt 缓存命中率、p95 延迟、$/查询 |
| 15 | 漂移监控仪表盘 | Phoenix 实时仪表盘，带每周检索质量趋势 |
| **100** | | |

## 练习

1. 在另一个辖区下再建一个语料切片（例如在 GDPR 之外加上 HIPAA）。用 20 个跨辖区探针证明角色+辖区过滤能阻止交叉泄漏。

2. 测量一周生产流量中的 prompt 缓存命中率。找出哪些查询破坏了缓存前缀。重构。

3. 加多轮记忆，使用 10k token 的摘要缓冲。测量对话变长时忠实度是否下降。

4. 把 Claude Sonnet 4.7 换成自托管的 Llama 3.3 70B。测量 $/查询和忠实度的差值。

5. 加一个「不确定」模式：如果重排序后的最高分低于阈值，智能体就说「我没有可信的引用」，而不是强行回答。测量误置信的下降幅度。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Prompt caching | 「缓存系统提示 + 上下文」 | Claude/OpenAI 的特性：命中的缓存前缀 token 打 60-90% 折扣 |
| RAGAS | 「RAG 评估器」 | 自动打分忠实度、答案相关性、上下文精确率 |
| Golden set | 「标注评估集」 | 200+ 条专家标注的带引用问答；即 ground truth |
| Jurisdiction tag | 「合规标签」 | 挂在分块上的 GDPR/HIPAA/SOC2 范围标记；由检索过滤强制执行 |
| Citation faithfulness | 「有依据的回答率」 | 有可检索来源片段支撑的论断比例 |
| Drift | 「检索质量衰减」 | nDCG 或引用分数的周度变化；告警阈值 5% |
| Red team | 「对抗评估」 | 发布前的越狱、PII 提取、域外探测 |

## 延伸阅读

- [Harvey AI](https://www.harvey.ai) —— 法律行业生产栈参考
- [Glean 企业搜索](https://www.glean.com) —— 企业规模 RAG 的参考
- [Mendable 文档](https://mendable.ai) —— 开发者文档 RAG 参考
- [LlamaCloud Parse + Index](https://docs.cloud.llamaindex.ai/llamaparse/getting_started) —— 托管摄取
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) —— 成本杠杆的参考
- [RAGAS 0.2 文档](https://docs.ragas.io/) —— 标准 RAG 评估框架
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) —— 漂移可观测性的参考
- [Llama Guard 4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) —— 2026 年安全分类器
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) —— 策略规则框架
