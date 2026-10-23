# OpenAI Agents SDK：交接、防护栏与追踪

> OpenAI Agents SDK 是构建在 Responses API 之上的轻量级多 Agent 框架。五个原语：Agent、Handoff、Guardrail、Session、Tracing。交接是名为 `transfer_to_<agent>` 的工具。防护栏在输入或输出上触发。追踪默认开启。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 minutes

## 学习目标

- 说出 OpenAI Agents SDK 的五个原语。
- 解释交接：为什么把它们建模为工具、模型看到的名称形态、上下文如何传递。
- 区分输入防护栏、输出防护栏和工具防护栏；解释 `run_in_parallel` 与阻塞模式。
- 用标准库实现一个带交接 + 防护栏 + span 风格追踪的运行时。

## 问题

无法干净地委派的 Agent 会把所有东西塞进一个 prompt。没有防护栏的 Agent 会泄露 PII、输出违反政策的文本，或无限循环。OpenAI 的 SDK 把让多 Agent 工作可管理的三个原语固化下来。

## 概念

### 五个原语

1. **Agent。** LLM + 指令 + 工具 + 交接。
2. **Handoff。** 委派给另一个 Agent。对模型表示为名为 `transfer_to_<agent_name>` 的工具。
3. **Guardrail。** 对输入（仅第一个 Agent）、输出（仅最后一个 Agent）或 tool call（每个函数工具）做校验。
4. **Session。** 跨轮次自动维护对话历史。
5. **Tracing。** 为 LLM 生成、tool call、交接、防护栏内置 span。

### 交接即工具

模型在工具列表中看到 `transfer_to_billing_agent`。调用它会通知运行时：

1. 复制对话上下文（或通过 `nest_handoff_history` beta 压缩）。
2. 用目标 Agent 的指令初始化它。
3. 用目标 Agent 继续运行。

这就是 supervisor 模式（第 13 课 / 第 28 课）的产品化。

### 防护栏

三种：

- **输入防护栏。** 在第一个 Agent 的输入上运行。在任何 LLM 调用之前拒绝不安全或超出范围的请求。
- **输出防护栏。** 在最后一个 Agent 的输出上运行。捕获 PII 泄露、政策违规、格式错误的响应。
- **工具防护栏。** 按函数工具运行。校验参数、检查权限、审计执行。

模式：

- **并行**（默认）。防护栏 LLM 与主 LLM 同时运行。尾部延迟更低。如果触发，主 LLM 的工作被丢弃（浪费 token）。
- **阻塞**（`run_in_parallel=False`）。防护栏 LLM 先运行。如果触发，不会在主调用上浪费 token。

Tripwire 会抛出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### 追踪

默认开启。每次 LLM 生成、tool call、交接和防护栏都会发出一个 span。`OPENAI_AGENTS_DISABLE_TRACING=1` 可关闭。`add_trace_processor(processor)` 把 span 同时分发到你自己的后端。

### Session

`Session` 把对话历史存储在某个后端（SQLite、Redis、自定义）。`Runner.run(agent, input, session=session)` 自动加载并追加。

### 这个模式会在哪里出错

- **交接漂移。** Agent A 交接给 Agent B，B 又交回给 A。加一个跳数计数器。
- **防护栏绕过。** 工具防护栏只对函数工具触发；内置工具（文件读取、网页抓取）需要单独的策略。
- **过度追踪。** span 中的敏感内容。配合 OTel GenAI 内容捕获规则（第 23 课）——内容存到外部，用 ID 引用。

```figure
ae-agent-handoff
```

## 构建它

`code/main.py` 用标准库实现 SDK 的形态：

- `Agent`、`FunctionTool`、`Handoff`（作为一个带转移语义的函数工具）。
- `Runner`，带输入/输出/工具防护栏、交接分派和跳数计数器。
- 一个简单的 span 发射器，用来展示 trace 形状。
- 一个分诊 Agent，根据用户查询交接给账单或支持团队；其中一个输入会触发防护栏。

运行它：

```
python3 code/main.py
```

trace 显示两次成功的交接、一次输入防护栏触发，以及一棵与真实 SDK 输出对应的 span 树。

## 使用它

- **OpenAI Agents SDK** 用于以 OpenAI 为先的产品。
- **Claude Agent SDK**（第 17 课）用于以 Claude 为先的产品。
- **LangGraph**（第 13 课）当你要显式状态和持久恢复时。
- **自建** 当你需要精确控制（语音、多提供方、联邦部署）时。

## 交付它

`outputs/skill-agents-sdk-scaffold.md` 搭建一个 Agents SDK 应用，包含分诊 Agent、交接、输入/输出/工具防护栏、session 存储和 trace 处理器。

## 练习

1. 增加交接跳数计数器：N 次转移后拒绝。追踪其行为。
2. 把 `nest_handoff_history` 实现为一个选项——转移前把先前的消息压缩成一份摘要。
3. 写一个阻塞式输出防护栏。比较会触发它的 prompt 与会通过的 prompt 的延迟。
4. 把 `add_trace_processor` 接到一个 JSON logger。它每个 span 输出什么形状？
5. 阅读 SDK 文档。把你的标准库玩具移植到 `openai-agents-python`。你的建模哪里错了？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Agent | 「LLM + 指令」 | SDK 中的 Agent 类型；持有工具和交接 |
| Handoff | 「转移」 | 模型调用以委派给另一个 Agent 的工具 |
| Guardrail | 「策略检查」 | 对输入 / 输出 / tool call 的校验 |
| Tripwire | 「防护栏触发」 | 防护栏拒绝时抛出的异常 |
| Session | 「历史存储」 | 跨运行持久化的对话记忆 |
| Tracing | 「span」 | 覆盖 LLM + 工具 + 交接 + 防护栏的内置可观测性 |
| 阻塞式防护栏 | 「顺序检查」 | 防护栏先运行；触发时不会浪费 token |
| 并行防护栏 | 「并发检查」 | 防护栏同时运行；延迟更低，触发时浪费 token |

## 延伸阅读

- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) —— 原语、交接、防护栏、追踪
- [Claude Agent SDK 概览](https://platform.claude.com/docs/en/agent-sdk/overview) —— Claude 风格的对应实现
- [Anthropic，Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) —— 什么时候才该用交接
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— Agents SDK 的 span 所映射的标准
