# LLM 可观测性技术栈选型

> 2026 年的可观测性市场分为两类。开发平台（LangSmith、Langfuse、Comet Opik）把监控与评估、prompt 管理、会话回放打包在一起。网关/埋点工具（Helicone、SigNoz、OpenLLMetry、Phoenix）专注于遥测。Langfuse 的核心采用 MIT 许可，开源与商业的平衡做得很好（云服务免费额度为每月 5 万事件）。Phoenix 是 OpenTelemetry 原生的，采用 Elastic License 2.0——非常适合漂移/RAG 可视化，但不是持久化的生产后端。Arize AX 采用零拷贝的 Iceberg/Parquet 集成，宣称比单体式可观测性方案便宜 100 倍。LangSmith 在 LangChain/LangGraph 场景领先，每用户每月 39 美元，仅企业版支持自托管。Helicone 基于代理，15-30 分钟即可接入，每月 10 万次请求免费，但在智能体轨迹方面的深度不足。常见的生产模式是：网关（Helicone/Portkey）+ 评估平台（Phoenix/TruLens），用 OpenTelemetry 粘合。

**Type:** Learn
**Languages:** Python (stdlib, toy trace-sampling simulator)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## 学习目标

- 区分开发平台（打包了评估 + prompt + 会话）与网关/遥测工具（只有轨迹 + 指标）。
- 把六种主要工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）对应到各自的许可、定价和最合适的用例。
- 解释 OpenTelemetry 粘合模式，它让你能把网关工具与独立的评估平台组合起来。
- 说出 2026 年的成本差异点（Arize AX 的零拷贝方案 vs 单体式摄取），并给出大致的 100 倍乘数。

## 问题

你上线了一个 LLM 功能。它能用。但你对 prompt 失败、工具循环、延迟回归、成本尖峰或 prompt 缓存命中率一无所知。你 Google「LLM observability」，得到八个工具，它们都声称在三个不同的价位上解决同一个问题。

它们解决的并不是同一个问题。LangSmith 回答「这次 LangGraph 运行为什么失败？」；Phoenix 回答「我的 RAG pipeline 是否在漂移？」；Helicone 回答「哪个应用在烧 token？」；Langfuse 回答「我能不能把整套东西自托管？」。不同的工具，不同的受众。

选型涉及四个维度：技术栈（LangChain？原生 SDK？多厂商？）、许可容忍度（只要 MIT？Elastic 可以吗？商业许可也行？）、预算（免费档？每月 100 美元？每月 1000 美元？）以及自托管（必须？锦上添花？绝不需要？）。

## 概念

### 两类工具

**开发平台**把可观测性与评估、prompt 管理、数据集版本化、会话回放打包在一起。你可以运行实验，看哪个 prompt 有效，再用数据集把新 prompt 与旧的优胜者做回归对比。LangSmith、Langfuse、Comet Opik。

**网关/遥测工具**为推理调用做埋点——prompt、响应、token、延迟、模型、成本。Helicone、SigNoz、OpenLLMetry、Phoenix。风格极简。可以通过 OpenTelemetry 与单独的评估工具组合。

### Langfuse——开源与商业的平衡

- 核心采用 Apache / MIT 许可；可通过 Docker 自托管。
- 云服务免费额度：每月 5 万事件。付费：团队版每月 29 美元。
- 评估、prompt 管理、轨迹、数据集。四类开发平台功能都有合理覆盖。
- 最佳场景：你想要 LangSmith 级别的功能，但必须自托管或保持在开源许可之下。

### Phoenix（Arize）——遥测优先，OpenTelemetry 原生

- Elastic License 2.0；自托管非常简单。
- 擅长 RAG 和漂移可视化。embedding 空间散点图是一等公民。
- 并非为持久化的生产后端而设计——主要是开发期的可观测性。
- 最佳场景：RAG pipeline 开发、漂移调试；生产环境再配一个单独的网关。

### Arize AX——靠规模取胜

- 商业产品。通过 Iceberg/Parquet 实现零拷贝的数据湖集成。
- 宣称在大规模下比单体式可观测性（Datadog 级别）便宜约 100 倍。账是这么算的：你把轨迹以 Parquet 存在自己的 S3 上；Arize 直接读取。
- 最佳场景：每天超过 1000 万条轨迹、已有数据湖、想要 LLM 专用仪表盘又不想付 Datadog 的价格。

### LangSmith——LangChain/LangGraph 优先

- 商业产品，每用户每月 39 美元。仅企业版可自托管。
- 对 LangChain 和 LangGraph 技术栈是同类最佳。如果你不用这两个，它的吸引力就没那么大。
- 最佳场景：团队已押注 LangChain，且愿意付费。

### Helicone——基于代理的最小可用方案

- 把 `OPENAI_API_BASE` 换成 Helicone 代理，15-30 分钟即可完成接入。
- MIT 许可；每月 10 万次请求免费，付费每月 20 美元起。
- 包含故障转移、缓存、速率限制——本身也充当网关。
- 在智能体/多步轨迹方面深度不足。
- 最佳场景：快速起步、单栈应用、需要网关与可观测性二合一。

### Opik（Comet）——开源开发平台

- Apache 2.0，完全开源。
- 功能集与 Langfuse 类似，带有 Comet 的基因。
- 最佳场景：已经使用 Comet 的 ML 团队，想在同一个面板里看 LLM 可观测性。

### SigNoz——OpenTelemetry 优先的全栈 APM

- Apache 2.0。既能做通用 APM，也能通过 OpenTelemetry 处理 LLM。
- 最佳场景：跨服务和 LLM 调用的统一可观测性。

### 粘合剂：OpenTelemetry + GenAI 语义约定

OpenTelemetry 在 2025 年末发布了 GenAI 语义约定（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。消费 OTel 数据的工具可以互操作。正在成形的生产模式是：

1. 每次 LLM 调用都按 GenAI 约定发出 OTel 数据。
2. 日常流量路由到网关（Helicone / Portkey）。
3. 双写到评估平台（Phoenix / Langfuse）用于回归分析。
4. 归档到数据湖（Iceberg），通过 Arize AX 或 DuckDB 做长期分析。

### 陷阱：在错误的层做埋点

在智能体框架内部埋点（例如添加 LangSmith 轨迹）会让你与该框架耦合。在 HTTP/OpenAI SDK 层埋点（通过 OpenLLMetry 或你的网关）则是可移植的。

### 采样——你不可能全部保留

每天超过 100 万次请求时，全量保留轨迹的成本比 LLM 调用本身还高。按规则采样：错误 100%、高成本 100%、成功 5%。聚合指标始终保留；原始数据只保留长尾。

### 值得记住的数字

- Langfuse 免费云：每月 5 万事件。
- LangSmith：每用户每月 39 美元。
- Helicone 免费：每月 10 万次请求。
- Arize AX 宣称：大规模下比单体式方案便宜约 100 倍。
- OpenTelemetry GenAI 约定：2025 年开始落地，2026 年被广泛采用。

```figure
i4-otel-glue
```

## 使用它

`code/main.py` 模拟一天 100 万条轨迹在三种保留策略下的表现（100% 入库、采样、采样 + 错误）。报告存储成本，以及每种策略丢失了什么。

## 交付它

本课产出 `outputs/skill-observability-stack.md`。给定技术栈、规模、预算和许可态度，挑选合适的工具。

## 练习

1. 你的团队使用 LangChain，想要开源、可自托管的可观测性。在 Langfuse 和 Opik 之间选一个并说明理由。
2. 每天 500 万条轨迹，Datadog 报价每月 15 万美元，计算 Arize AX 的盈亏平衡点。
3. 设计一套 OpenTelemetry GenAI 属性集合，作为你所在组织要求每次 LLM 调用都必须携带的规范。
4. 论证 Phoenix 单独是否足以支撑生产。它在什么情况下不够用？
5. Helicone 的代理开销是 20 ms。P99 TTFT 为 300 ms 时，这可以接受吗？如果 SLA 是 100 ms 呢？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| OpenLLMetry | 「LLM 的 OTel」 | 面向 LLM 的开源 OpenTelemetry 埋点库 |
| GenAI conventions | 「OTel 属性」 | LLM 调用的标准 OTel 属性名 |
| LangSmith | 「LangChain 可观测性」 | 与 LangChain 生态捆绑的商业平台 |
| Langfuse | 「开源的 LangSmith」 | MIT 开源，功能集类似 |
| Phoenix | 「Arize 开发工具」 | OpenTelemetry 原生的开发/评估平台 |
| Arize AX | 「规模化可观测性」 | 商业化的零拷贝 Iceberg/Parquet 可观测性 |
| Helicone | 「代理式可观测性」 | 收集 LLM 遥测的 HTTP 代理 + 网关功能 |
| Opik | 「Comet 的 LLM 工具」 | Comet 出品的 Apache 2.0 开源开发平台 |
| Session replay | 「轨迹重放」 | 连同工具调用一起重放完整的智能体会话 |
| Eval | 「离线测试」 | 在标注数据集上运行候选模型/prompt |

## 延伸阅读

- [SigNoz —— 2026 年顶级 LLM 可观测性工具](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse —— Arize AX 替代方案分析](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI —— 搭建 Langfuse、LangSmith、Helicone、Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix 文档](https://docs.arize.com/phoenix)
- [Helicone 文档](https://docs.helicone.ai/)
