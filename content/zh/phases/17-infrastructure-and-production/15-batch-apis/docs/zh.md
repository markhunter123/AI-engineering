# Batch API——50% 折扣成为行业标准

> 每家主要供应商都提供异步 batch API，折扣 50%，周转约 24 小时。OpenAI、Anthropic、Google，以及大多数推理平台（Fireworks 的 batch 档、Together batch）都实现了同一模式。把 batch 与 prompt 缓存叠加，夜间 pipeline 的成本可降至同步未缓存的约 10%。规则简单得近乎粗暴：只要不是交互式负载，就该上 batch。内容生成 pipeline、文档分类、数据抽取、报告生成、批量标注、目录打标——任何能容忍 24 小时延迟的工作，在迁移到 batch 之前都是在白白扔钱。2026 年的生产模式是把每个新的 LLM 负载分流到三条通道：交互式（同步 + 缓存）、半交互式（异步队列 + 兜底）、batch（夜间运行，叠加缓存输入）。那些假装是交互式、却能容忍数分钟延迟的负载，浪费最多。

**Type:** Learn
**Languages:** Python (stdlib, toy batch-vs-sync cost simulator)
**Prerequisites:** Phase 17 · 14 (Prompt & Semantic Caching)
**Time:** ~45 minutes

## 学习目标

- 说出三家供应商的 batch API（OpenAI、Anthropic、Google）以及共同的 50% 折扣 + 24 小时周转承诺。
- 计算在夜间分类负载上叠加 batch + 缓存输入的成本，并与同步未缓存基线对比。
- 把负载分流为交互式 / 半交互式 / batch，并说明选择该通道的理由。
- 说出两个陷阱：部分交互性（用户期望快于 24 小时）和输出 schema 漂移（batch 文件格式因供应商而异）。

## 问题

你的团队交付了一条夜间报告生成 pipeline。5 万份文档，逐份摘要，对摘要聚类，再起草一份高管简报。同步运行需要 4 小时，每晚 $2,000。你听说了 batch API。

Batch 给你 50% 折扣。你还为系统 prompt（5 万次调用共享）启用了 prompt 缓存。叠加之后，账单降到每晚 $180——约为基线的 9%。同一条 pipeline，三处配置改动。

Batch 是 LLM 成本工具箱里最便宜、却没人拉的那根杠杆。原因多半是组织性的：团队想着「实时」，而 SLA 其实是「早上之前」。本课讲的就是别把 90% 的账单留在桌上。

## 概念

### 三种 batch API

**OpenAI Batch API**：上传 JSONL 文件，内含请求列表。承诺 24 小时周转（实践中通常约 2-8 小时）。输入和输出 token 均享 50% 折扣。端点为 `/v1/batches`。符合缓存条件的输入还能在此基础上享受缓存输入定价。

**Anthropic Message Batches**：上传 JSONL。24 小时周转。50% 折扣。支持 `cache_control`——缓存写入是显式的，读取在 batch 内自动发生。

**Google Vertex AI Batch Prediction**：输入来自 BigQuery 或 GCS。Gemini 同样有 50% 折扣。与 Vertex pipeline 集成。

### 语义：异步，不是慢

Batch 是「我承诺 24 小时内返回」——而不是「这要花 24 小时」。典型的 P50 是 2-6 小时。供应商会在 GPU 库存空闲的低峰时段调度你的 batch。

### 与缓存叠加

5 万份文档摘要，共享同一个 4K token 系统 prompt：

- 同步未缓存：50000 ×（$input × 4000 + $output × 200），按全价。
- 同步缓存：系统 prompt 在首次写入后被缓存；其余 49999 次的输入成本便宜 10 倍。
- Batch 缓存：上述所有，加上读取和写入都享 50% 折扣。

叠加效果：batch + 缓存 = 同步未缓存账单的约 10%。任何在夜间运行、且有共享系统 prompt 的负载都该用这个组合。

### 负载分流

**交互式**——用户等待响应。TTFT 很重要。同步调用 + prompt 缓存。不能 batch。

**半交互式**——用户提交任务，几分钟后回来看。异步队列，batch 不可用时回退到同步。典型例子是中等规模的 RAG 索引。

**Batch**——用户期望「早上之前」或「一小时内」拿到结果。内容 pipeline、规模化分类、离线分析。永远用 batch，永远叠加缓存。

常见错误：因为 pipeline 是生产环境，就把一切都归类为交互式。生产不是延迟规格——SLA 才是。

### 部分交互性陷阱

有些功能看起来是交互式的，但能容忍 5-10 分钟。例如：一份带「刷新」按钮的夜间客户健康报告。用户点击刷新，等 10 分钟没问题。团队却把它做成了同步。50 个并发刷新所花的成本，是通过 batch 处理并用邮件送达的 10 倍。

要问的问题是：「24 小时对这位用户意味着什么？」如果答案是「他们根本不会注意到」，就上 batch。

### 输出 schema 陷阱

Batch 文件格式因供应商而异：

- OpenAI：JSONL，每行一个请求。
- Anthropic：JSONL，每行一条 message；响应格式内嵌。
- Vertex：BigQuery 表，或带 TFRecord 的 GCS 前缀。

写一个跨供应商的「batch 客户端」，意味着为每家供应商写适配代码。宣称支持多供应商 batch 的网关（Portkey、LiteLLM 的部分档位）也只是对原始格式做了薄封装。

### 你应该记住的数字

- 各供应商的 batch 折扣：输入 + 输出统一 50%。
- 周转 SLA：保证 24 小时，典型 P50 为 2-6 小时。
- batch + 缓存输入叠加：约为同步未缓存成本的 10%。
- 负载分流规则：只要能接受 24 小时延迟，就永远用 batch。

```figure
batch-lane-triage
```

## 使用它

`code/main.py` 为 5 万份文档的负载计算同步、同步+缓存、batch、batch+缓存四种成本。以美元和百分比报告节省额。

## 交付它

本课产出 `outputs/skill-batch-triager.md`。给定负载特征，分流为交互式/半交互式/batch，并估算节省。

## 练习

1. 运行 `code/main.py`。对于一条 10 万文档、系统 prompt 3K token、输出 500 token 的 pipeline，计算全栈（batch + 缓存）相对同步基线的节省。
2. 从你熟悉的真实产品中挑三个功能。把每一个分流为交互式/半交互式/batch。
3. 一位用户抱怨报告花了 3 小时。这是 batch 分流错误，还是合理的交互式需求？写出判定标准。
4. 你的 batch API 返回 SLA 是 24 小时，但 P99 是 20 小时。你如何向用户传达——在边界情况下下游系统应有什么行为？
5. 计算盈亏平衡点：共享前缀多长时，batch + 缓存会比用自己的预留 GPU 跑通宵更便宜？

## 关键术语

| 术语 | 人们怎么说 | 它实际指什么 |
|------|----------------|------------------------|
| Batch API | 「异步折扣」 | 50% 折扣，24 小时周转 |
| JSONL | 「batch 格式」 | 每行一个 JSON 请求；OpenAI/Anthropic 的事实标准 |
| Message Batches | 「Anthropic batch」 | Anthropic 的 batch API 产品名 |
| Batch prediction | 「Vertex batch」 | Vertex AI 的 batch API 产品 |
| 周转 SLA | 「24 小时承诺」 | 是保证而非典型值；典型为 2-6 小时 |
| 负载分流 | 「交互性决策」 | 交互式 / 半交互式 / batch 的路由决策 |
| 输出 schema | 「响应格式」 | 各家供应商的 JSONL 布局；不可移植 |
| 叠加折扣 | 「batch + 缓存」 | 两者同时生效时约为未缓存同步账单的 10% |

## 延伸阅读

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch)——JSONL 格式与 `/v1/batches` 语义。
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing)——batch 格式与 `cache_control` 的交互。
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini)——Gemini 的 batch 语义。
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
