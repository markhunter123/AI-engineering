# 混合记忆：向量 + 图 + KV

> 混合记忆并行运行三个存储——向量用于语义相似度，KV 用于快速事实查找，图用于实体关系推理——再用一个打分层在检索时将它们融合。这是外部记忆广泛使用的生产模式；Mem0（Chhikara 等，2025）是一个参考实现。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 minutes

## 学习目标

- 解释为什么单一存储（只有向量、只有图、只有 KV）不足以支撑 Agent 记忆。
- 说出 Mem0 的三个并行存储，以及每个存储优化什么。
- 描述 Mem0 的融合打分——相关性、重要性、时效性——并解释为什么它是加权和，而不是层级关系。
- 用标准库实现一个玩具级三存储记忆：`add()` 写入全部三个存储，`search()` 融合结果。

## 问题

单一存储对三类查询中的某一类总是错的：

- **语义相似度** —— 「我们上周讨论过什么关于 Agent 漂移的内容？」向量胜出；KV 和图都会漏掉。
- **事实查找** —— 「用户的电话号码是多少？」KV 胜出；向量浪费，图是杀鸡用牛刀。
- **关系推理** —— 「哪些客户共享同一个账单实体？」图胜出；向量和 KV 答不了。

生产 Agent 在同一次会话中会发出全部三类查询。单存储记忆对其中两类永远是错的。Mem0 的贡献是把三个存储接到单一的 `add`/`search` 接口后面，并用一个打分函数融合它们。

## 概念

### 三个并行存储

Mem0（arXiv:2504.19413，2025 年 4 月）在 `add(text, user_id, metadata)` 时：

1. 从文本中抽取候选事实（一个由 LLM 驱动的步骤）。
2. 把每条事实写入向量存储（embedding），用于语义搜索。
3. 把每条事实写入以 (user_id, fact_type, entity) 为键的 KV 存储，用于 $O(1)$ 查找。
4. 把每条事实作为带类型的边写入图存储（Mem0g），用于关系查询。

在 `search(query, user_id)` 时：

1. 向量存储按 embedding 余弦返回 top-k。
2. KV 存储按从查询推导出的 (user_id, type, entity) 返回直接命中。
3. 图存储返回从查询实体可达的子图。
4. 打分层融合三者。

### 融合打分

```
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **相关性** —— 向量余弦、KV 精确匹配、图路径权重。
- **重要性** —— 写入时打标签或学习得到（有些事实更重要：姓名、ID、策略）。
- **时效性** —— 距上次写入或读取的时间做指数衰减。

权重按产品调优。聊天 Agent 调高 `w_recency`；合规 Agent 调高 `w_importance`；检索 Agent 调高 `w_relevance`。

### Mem0g 与时间推理

Mem0g 增加了冲突检测器。当新事实与已有边矛盾时，把已有边标记为失效但不删除。时间查询（「用户三月份在哪个城市？」）遍历「该时刻有效」的子图。

这正是 Letta 的失效模式所推广的合规级行为。

### 基准测试数据

Mem0 论文报告（2025）：

- **LoCoMo**（长篇对话记忆）：91.6
- **LongMemEval**（长程情景记忆）：93.4
- **BEAM 1M**（100 万 token 记忆基准）：64.1

对比基线（128k 全上下文 LLM、扁平向量存储、扁平 KV）全都落后 10 分以上。单靠基准不能证明选择合理——运行形态才能——但这些数字表明融合设计不是舍入误差。

### 作用域分类

Mem0 按作用域划分记忆：

- **用户记忆** —— 跨会话持久，以 `user_id` 为键。
- **会话记忆** —— 在单个线程内持久。
- **Agent 记忆** —— 每个 Agent 实例的状态。

每次写入选择一个作用域。检索可以跨作用域查询，并按作用域加权。不加思考地混用作用域，就会产生「助手把 Bob 的项目告诉了 Alice」这类事故。

### 这个模式会在哪里出错

- **Embedding 漂移。** 前几百次查询看起来正确的向量结果，会随语料增长而退化。定期对使用频率最高的 N 条记录重新做 embedding。
- **KV schema 蔓延。** `(user_id, type, entity)` 看着简单，直到每个团队都加自己的 `type`。每季度审计一次 type 集合。
- **图爆炸。** 一个嘈杂的抽取器每条消息能加 50 条边。限制每次 `add` 调用的图写入数量；丢弃低置信度的边。

```figure
ae-memory-fusion
```

## 构建它

`code/main.py` 用标准库实现了三存储模式：

- `VectorStore` —— 用朴素的 token 重叠相似度充当 embedding 的替身。
- `KVStore` —— 以 `(user_id, fact_type, entity)` 为键的字典。
- `GraphStore` —— 带类型的边（subject、relation、object、valid）。
- `Mem0` —— 顶层门面，提供 `add()`、`search()`、融合打分和作用域感知检索。
- 一段多用户、多会话对话的完整运行轨迹。

运行它：

```
python3 code/main.py
```

输出展示三条独立的回忆路径，以及融合后的 top-k。改动 `main()` 顶部的打分权重，观察排序变化。

## 使用它

- **Mem0（Apache 2.0）** —— 生产可用。用 Postgres + Qdrant + Neo4j 自托管，或使用托管云。
- **Letta** —— 核心/回忆/归档三层；自带向量和图后端。
- **Zep** —— 商业替代方案，带时间 KG 和事实抽取。
- **自建方案** —— 当你需要精确控制抽取器（合规）或融合权重（时效性占主导的语音 Agent）时。

## 交付它

`outputs/skill-hybrid-memory.md` 生成三存储记忆脚手架，内置融合打分器、作用域分类和时间失效。

## 练习

1. 把玩具级向量相似度替换为真实的 embedding 模型（sentence-transformers、Ollama、OpenAI embeddings）。在合成的长对话上测量 recall@10。经过 1000 次写入后，排序会漂移吗？
2. 增加时间查询：`search(query, as_of=timestamp)`。只返回在该时刻或之前有效的记录。哪个存储需要改动最多？
3. 实现冲突检测器：如果传入事实与某条图边矛盾，使旧边失效并记录两者。用「用户住在柏林」-> 「用户住在里斯本」来测试。
4. 扩展融合打分器，加入 `user_feedback` 维度（对检索记录点赞）。如何防止刷分（Agent 只返回它本来就喜欢的记录）？
5. 阅读 Mem0 文档（`docs.mem0.ai`）。把玩具实现移植到 `mem0` 客户端调用。在同样的 20 条测试查询上比较检索质量。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 混合记忆（Hybrid memory） | 「向量加图加 KV」 | 三个存储并行写入，检索时融合 |
| 事实抽取 | 「记忆摄入」 | LLM 步骤，把文本拆成 (entity, relation, fact) 元组 |
| 融合打分 | 「相关性排序」 | 相关性、重要性、时效性的加权和 |
| 作用域（Scope） | 「记忆命名空间」 | user / session / agent——决定谁能看到什么 |
| Mem0g | 「记忆图」 | 带时间有效性的类型化边，用于关系查询 |
| 时间失效 | 「软删除」 | 将被推翻的边标记为失效；绝不删除 |
| Embedding 漂移 | 「检索腐化」 | 语料增长导致向量质量下降；定期重新 embedding |

## 延伸阅读

- [Chhikara 等，Mem0（arXiv:2504.19413）](https://arxiv.org/abs/2504.19413) —— 原始论文
- [Mem0 文档](https://docs.mem0.ai/platform/overview) —— 生产 API、SDK、托管云
- [Packer 等，MemGPT（arXiv:2310.08560）](https://arxiv.org/abs/2310.08560) —— 虚拟上下文的前身
- [Letta, Memory Blocks 博客](https://www.letta.com/blog/memory-blocks) —— 三层结构的姊妹设计
