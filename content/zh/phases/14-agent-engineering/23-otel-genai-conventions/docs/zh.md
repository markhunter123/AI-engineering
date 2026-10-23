# OpenTelemetry GenAI 语义约定

> OpenTelemetry 的 GenAI SIG（2024 年 4 月启动）为智能体遥测定义了标准 schema。span 名称、属性和内容捕获规则在各厂商之间趋于一致，因此智能体 trace 在 Datadog、Grafana、Jaeger 和 Honeycomb 中含义相同。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 24 (Observability Platforms)
**Time:** ~60 minutes

## 学习目标

- 说出 GenAI span 类别：model/client、agent、tool。
- 区分 `invoke_agent` 的 CLIENT 与 INTERNAL span，以及各自适用的时机。
- 列出顶层 GenAI 属性：provider 名称、请求模型、数据源 ID。
- 解释内容捕获契约：opt-in、`OTEL_SEMCONV_STABILITY_OPT_IN`、外部引用建议。

## 问题

每个厂商都发明自己的 span 名称。运维团队最终要为每个框架各建一套仪表盘。OpenTelemetry 的 GenAI SIG 通过定义一个生态共同遵循的标准来解决这个问题。

## 概念

### Span 类别

1. **Model / client span。** 覆盖原始 LLM 调用。由 provider SDK（Anthropic、OpenAI、Bedrock）和框架的模型适配器发出。
2. **Agent span。** `create_agent`（智能体构建时）和 `invoke_agent`（运行时）。
3. **Tool span。** 每次工具调用一个；通过父子关系连接到 agent span。

### Agent span 命名

- Span 名称：若已命名，为 `invoke_agent {gen_ai.agent.name}`；否则回退到 `invoke_agent`。
- Span 类型：
  - **CLIENT** — 用于远程智能体服务（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** — 用于进程内智能体框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` — 模型 ID。
- `gen_ai.response.model` — 解析后的模型（可能因路由而与请求不同）。
- `gen_ai.agent.name` — 智能体标识符。
- `gen_ai.operation.name` — `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` — 用于 RAG：咨询了哪个语料库或存储。

Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 都有技术特定的约定。

### 内容捕获

默认规则：instrumentation 不应捕获输入/输出。捕获通过以下属性 opt-in：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产模式：把内容存到外部（S3、你的日志存储），在 span 上只记录引用（指针 ID，而不是正文）。这是把第 27 课的内容投毒防御接入可观测性。

### 稳定性

截至 2026 年 3 月，大多数约定仍为实验性。通过以下方式选择启用稳定预览：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 原生把 GenAI 属性映射到其 LLM Observability schema。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 这种模式容易出错的地方

- **在 span 中捕获完整 prompt。** PII、密钥、客户数据进入运维可读的 trace。应存到外部。
- **缺少 `gen_ai.provider.name`。** 多 provider 仪表盘在缺少归属时会失效。
- **span 没有父子链接。** 工具 span 成为孤儿。始终传播上下文。
- **不设置稳定性 opt-in。** 后端升级时你的属性可能被重命名。

```figure
ae-genai-span-tree
```

## 构建它

`code/main.py` 实现一个符合 GenAI 约定的标准库 span emitter：

- `Span`，带 GenAI 属性 schema。
- `Tracer`，带 `start_span`、嵌套上下文。
- 一次脚本化的智能体运行，发出：`create_agent`、`invoke_agent`（INTERNAL）、每个工具一个 span、LLM 调用的 `chat` span。
- 一个内容捕获模式，把 prompt 存到外部并在 span 上记录 ID。

运行它：

```
python3 code/main.py
```

输出：一棵包含所有必需 GenAI 属性的 span 树，以及一个展示 opt-in 内容引用的「外部存储」。

## 使用它

- **Datadog LLM Observability**（v1.37+）原生映射属性。
- **Langfuse / Phoenix / Opik**（第 24 课）— 为生态提供自动 instrumentation。
- **Jaeger / Honeycomb / Grafana Tempo** — 原始 OTel trace；用 GenAI 属性构建仪表盘。
- **自托管** — 运行带 GenAI processor 的 OTel Collector。

## 交付它

`outputs/skill-otel-genai.md` 把 OTel GenAI span 接入现有智能体，带内容捕获默认值和外部引用存储。

## 练习

1. 用 `invoke_agent`（INTERNAL）+ 每工具 span 为你的第 01 课 ReAct 循环加 instrumentation。发送到 Jaeger 实例。
2. 以「仅引用」模式增加内容捕获：prompt 存入 SQLite，span 属性只携带行 ID。
3. 阅读 `gen_ai.data_source.id` 的规范。把它接入你第 09 课的 Mem0 搜索。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`，验证你的属性不会被 collector 重命名。
5. 构建一个仪表盘：仅用 GenAI 属性回答「哪些工具错误与哪些模型相关」。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| GenAI SIG | 「OpenTelemetry GenAI 小组」 | 定义 schema 的 OTel 工作组 |
| invoke_agent | 「智能体 span」 | 表示一次智能体运行的 span 名称 |
| CLIENT span | 「远程调用」 | 调用远程智能体服务的 span |
| INTERNAL span | 「进程内」 | 进程内智能体运行的 span |
| gen_ai.provider.name | 「provider」 | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | 「RAG 来源」 | 检索命中来自哪个语料库/存储 |
| 内容捕获 | 「prompt 日志」 | opt-in 捕获消息；生产环境存到外部 |
| 稳定性 opt-in | 「预览模式」 | 用于固定实验性约定的环境变量 |

## 延伸阅读

- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 默认发出 GenAI span
- [AutoGen v0.4（Microsoft Research）](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 内置 OTel span
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C trace context 传播
