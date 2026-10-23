# OpenTelemetry GenAI —— 端到端追踪 tool call

> 一个 Agent 调用了五个工具、三个 MCP 服务器和两个子 Agent。你需要一条贯穿所有这些调用的 trace。OpenTelemetry GenAI 语义约定（v1.37 及以后为稳定属性）是 2026 年的标准，Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 均原生支持。本课会列出必需属性，梳理 span 层级（agent → LLM → tool），并交付一个可插入任意 OTel exporter 的 stdlib span 发射器。

**Type:** Build
**Languages:** Python (stdlib, OTel span emitter)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## 学习目标

- 说出 LLM span 与工具执行 span 所需的 OTel GenAI 属性。
- 构建覆盖 Agent 循环、LLM 调用、tool call 和 MCP 客户端分发的 trace 层级。
- 决定哪些内容需要捕获（选择加入）、哪些需要脱敏（默认）。
- 在不重写工具代码的前提下，把 span 发送到本地收集器（Jaeger、Langfuse）。

## 问题

一个 2026 年 2 月的调试案例：用户报告「我的 Agent 有时要 30 秒才响应，有时只要 3 秒」。没有 trace。日志显示了 LLM 调用，但没有工具分发、没有 MCP 服务器往返、没有子 Agent。你只能猜。最终你发现：某个 MCP 服务器偶尔在冷启动时挂起。

没有端到端追踪，你找不到这个问题。OTel GenAI 解决了它。

这些约定在 2025—2026 年间由 OpenTelemetry semantic-conventions 小组确定下来。它们定义了稳定的属性名，使 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都能解析相同的 span。一次埋点，发往任意后端。

## 概念

### Span 层级

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

整个结构嵌套在一个 trace id 之下。span id 连接父子关系。

### 必需属性

根据 2025—2026 年的 semconv：

- `gen_ai.operation.name` — `"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name` — `"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model` —— 请求的模型字符串（例如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` —— 实际提供服务的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` —— 用于关联的提供方响应 id。

对于工具 span：

- `gen_ai.tool.name` —— 工具标识符。
- `gen_ai.tool.call.id` —— 具体的调用 id。
- `gen_ai.tool.description` —— 工具描述（可选）。

对于 Agent span：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### Span 类型

- `SpanKind.CLIENT` 用于跨越进程边界的调用（LLM 提供方、MCP 服务器）。
- `SpanKind.INTERNAL` 用于 Agent 自身的循环步骤和工具执行。

### 选择加入的内容捕获

默认情况下，span 携带指标和时序，而不携带 prompt 或补全内容。大载荷和 PII 默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 以及特定的内容捕获环境变量即可包含内容。在生产环境启用前请仔细评估。

### span 上的事件

token 级别的事件可以作为 span 事件添加：

- `gen_ai.content.prompt` —— 输入消息。
- `gen_ai.content.completion` —— 输出消息。
- `gen_ai.content.tool_call` —— 记录下来的 tool call。

事件在 span 内按时间排序，便于详细回放。

### Exporter

OTel span 可以导出到：

- **Jaeger / Tempo。** 开源，可本地部署。
- **Langfuse。** 面向 LLM 可观测性；可视化 token 使用量。
- **Arize Phoenix。** 评估与追踪合一。
- **Datadog。** 商业产品；原生解析 `gen_ai.*` 属性。
- **Honeycomb。** 列式存储；便于查询。

它们都说 OTLP 这种线上格式。你的代码无需关心。

### 跨 MCP 的传播

当 MCP 客户端调用服务器时，把 W3C traceparent 头注入请求。Streamable HTTP 支持标准头。Stdio 原生不携带 HTTP 头；规范 2026 路线图讨论了在 JSON-RPC 调用上增加 `_meta.traceparent` 字段。

在该特性发布之前：手动把 traceparent 放进每个请求的 `_meta`。服务器记录 trace id。

### 指标

除 span 之外，GenAI semconv 还定义了指标：

- `gen_ai.client.token.usage` —— 直方图。
- `gen_ai.client.operation.duration` —— 直方图。
- `gen_ai.tool.execution.duration` —— 直方图。

将这些用于不需要逐调用细节的仪表盘。

### AgentOps 层

AgentOps（成立于 2024 年）专注于 GenAI 可观测性。它封装流行框架（LangGraph、Pydantic AI、CrewAI）以自动发出 OTel span。如果你的技术栈使用受支持的框架，它会很有用；否则请使用手动埋点。

```figure
t3-span-waterfall
```

## 使用它

`code/main.py` 向 stdout 输出 OTel 形态的 span（类似 OTLP-JSON 的格式），演示一个 Agent 调用 LLM、分发两个工具并完成一次 MCP 往返。没有真实 exporter——本课聚焦 span 形态和属性集合。可以把输出粘贴到兼容 OTLP 的查看器中，或者直接阅读。

重点观察：

- trace id 在所有 span 之间共享。
- 父子链接通过 `parentSpanId` 编码。
- 必需的 `gen_ai.*` 属性都已填充。
- 内容捕获默认关闭；有一个场景通过环境变量将其开启。

## 交付它

本课产出 `outputs/skill-otel-genai-instrumentation.md`。给定一个 Agent 代码库，该技能会生成一份埋点方案：在何处添加 span、填充哪些属性、面向哪些 exporter。

## 练习

1. 运行 `code/main.py`。数一数 span 的数量，并指出哪些是 CLIENT、哪些是 INTERNAL。

2. 开启内容捕获（环境变量），确认出现 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件。注意这对 PII 的影响。

3. 添加工具执行指标 `gen_ai.tool.execution.duration`，并在每次调用时作为直方图样本发出。

4. 把父级 Agent span 中的 traceparent 传播到 MCP 请求的 `_meta.traceparent` 字段。验证 MCP 服务器会看到相同的 trace id。

5. 阅读 OTel GenAI semconv 规范。找出 semconv 中列出、但本课代码没有发出的一个属性，并补上它。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| OTel | 「OpenTelemetry」 | 面向 trace、指标、日志的开放标准 |
| GenAI semconv | 「GenAI 语义约定」 | 用于 LLM / 工具 / Agent span 的稳定属性名 |
| `gen_ai.*` | 「属性命名空间」 | 所有 GenAI 属性共享该前缀 |
| Span | 「计时操作」 | 有开始、结束和属性的工作单元 |
| Trace | 「跨 span 的谱系」 | 共享同一 trace id 的 span 树 |
| SpanKind | 「CLIENT / SERVER / INTERNAL」 | 关于 span 方向的提示 |
| OTLP | 「OpenTelemetry Line Protocol」 | exporter 的线上格式 |
| 选择加入的内容 | 「prompt / 补全捕获」 | 默认关闭；用环境变量开启 |
| traceparent | 「W3C 头」 | 跨服务传播 trace 上下文 |
| Exporter | 「面向特定后端的发送器」 | 把 span 发送到 Jaeger / Datadog 等的组件 |

## 延伸阅读

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI span、指标和事件的权威约定
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 与工具执行 span 的属性列表
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — Agent 级别的 `invoke_agent` span
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub 上托管的权威来源
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 生产集成演练
