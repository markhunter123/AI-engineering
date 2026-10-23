# Function Calling 深入剖析——OpenAI、Anthropic、Gemini

> 三家前沿供应商在 2024 年收敛到同一个 tool call 循环，然后又在其他一切上分道扬镳。OpenAI 用 `tools` 和 `tool_calls`。Anthropic 用 `tool_use` 和 `tool_result` 块。Gemini 用 `functionDeclarations` 和唯一 id 关联。本课并排对比三者，让你为一家供应商写的代码在移植时不会断裂。

**Type:** Build
**Languages:** Python (stdlib, schema translators)
**Prerequisites:** Phase 13 · 01 (the tool interface)
**Time:** ~75 minutes

## 学习目标

- 说出 OpenAI、Anthropic 和 Gemini function-calling 载荷之间的三处形状差异（声明、调用、结果）。
- 把一份工具声明转换为三家供应商的格式，并预测严格模式约束会在哪里不同。
- 在各供应商中使用 `tool_choice` 强制、禁止或自动选择 tool call。
- 了解各供应商的硬性限制（工具数量、schema 深度、参数长度），以及违反限制时它们各自发出的错误特征。

## 问题

function-calling 请求的形状因供应商而异。下面是 2026 年生产栈中的三个具体例子：

**OpenAI Chat Completions / Responses API。** 你传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是必须由你解析的 JSON 字符串。严格模式（`strict: true`）通过受约束解码强制 schema 合规。

**Anthropic Messages API。** 你传入 `tools: [{name, description, input_schema}]`。响应以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 返回。`input` 已经解析好（是对象，不是字符串）。你用一条新的 `user` 消息回复，其中包含 `{type: "tool_result", tool_use_id, content}` 块。

**Google Gemini API。** 你传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应以 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 到达，其中 `id` 在 Gemini 3 及以上版本中唯一，用于并行调用关联。你用 `{functionResponse: {name, id, response}}` 回复。

同样的循环。不同的字段名、不同的嵌套、字符串与对象的不同约定、不同的关联机制。一个在 OpenAI 上写天气 Agent 的团队，仅管道接通就要为移植到 Anthropic 花两天，再为 Gemini 花一天。

本课构建一个翻译器，把三种格式统一为一份规范工具声明，并在边界处路由。第 13 阶段 · 17 把同一模式泛化为 LLM 网关。

## 概念

### 共同结构

每个供应商都需要五样东西：

1. **工具列表。** 每个工具的名称、描述和输入 schema。
2. **工具选择。** 强制某个工具、禁止工具，或让模型决定。
3. **调用发出。** 指明工具和参数的结构化输出。
4. **调用 id。** 把响应关联到正确的调用（对并行很重要）。
5. **结果注入。** 把结果与调用绑定起来的一条消息或块。

### 逐字段的形状差异

| 方面 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 声明外层 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| schema 字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | assistant 消息上的 `tool_calls[]` | 类型为 `tool_use` 的 `content[]` | 类型为 `functionCall` 的 `parts[]` |
| 参数类型 | 字符串化的 JSON | 已解析的对象 | 已解析的对象 |
| id 格式 | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| 结果块 | 角色 `tool`，`tool_call_id` | 带 `tool_result`、`tool_use_id` 的 `user` | 带匹配 `id` 的 `functionResponse` |
| 强制指定工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格 schema | `strict: true` | schema 即 schema（始终强制执行） | 请求级别的 `responseSchema` |

### 你实际会撞上的限制

- **OpenAI。** 每次请求 128 个工具。schema 深度 5。参数字符串 <= 8192 字节。严格模式要求不使用 `$ref`，不允许有重叠的 `oneOf`/`anyOf`/`allOf`，每个属性都必须列在 `required` 中。
- **Anthropic。** 每次请求 64 个工具。schema 深度实际上无界，但实用上限为 10。没有严格模式开关；schema 是契约，模型倾向于遵守。
- **Gemini。** 每次请求 64 个函数。schema 类型是 OpenAPI 3.0 的子集（与 JSON Schema 2020-12 略有差异）。并行调用自 Gemini 3 起有唯一 id。

### `tool_choice` 的行为

三种人人支持的模式，但命名不同。

- **Auto。** 模型自行选择工具或文本。默认。
- **Required / Any。** 模型必须至少调用一个工具。
- **None。** 模型不得调用工具。

外加每家供应商独有的一种模式：

- **OpenAI。** 按名称强制指定工具。
- **Anthropic。** 按名称强制指定工具；`disable_parallel_tool_use` 标志区分单次与多次。
- **Gemini。** `mode: "VALIDATED"` 让每个响应都经过 schema 校验器，无论模型意图如何。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）在一条 assistant 消息中发出多个调用。你把它们全部执行，然后用一条批量的 tool 角色消息回复，每个 `tool_call_id` 一条记录。Anthropic 历史上只做单调用；`disable_parallel_tool_use: false`（截至 Claude 3.5 为默认）启用多个。Gemini 2 允许并行调用，但不提供稳定 id；Gemini 3 加入 UUID，使乱序响应可以干净地关联。

### 流式

三者都支持流式 tool call。线上的格式不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的增量块陆续到达。你持续累积，直到 `finish_reason: "tool_calls"`。
- **Anthropic。** block-start / block-delta / block-stop 事件。`input_json_delta` 块携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）发出带 `functionCallId` 的块，使多个并行调用可以交错。

第 13 阶段 · 03 深入并行与流式重组。本课聚焦声明与单调用形状。

### 错误与修复

无效参数错误的形态也不一样。

- **OpenAI（非严格）。** 模型返回 `arguments: "{bad json}"`，你的 JSON 解析失败，你注入一条错误消息并重新调用。
- **OpenAI（严格）。** 校验在解码过程中完成；无效 JSON 不可能出现，但 `refusal` 可能出现。
- **Anthropic。** `input` 可能包含意外字段；schema 只是建议性的。请在服务端校验。
- **Gemini。** OpenAPI 3.0 的怪癖：对象字段上的 `enum` 会被静默忽略；请自行校验。

### 翻译器模式

代码中的规范工具声明如下（形状由你选择）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数把它转换成三种供应商形状。`code/main.py` 中的框架正是这样做的，然后把一次假 tool call 在每家供应商的响应形状里往返一遍。无需网络——本课讲的是形状，不是 HTTP。

生产团队会把这一翻译器包进 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）。第 13 阶段 · 17 交付一个网关，在三者中任意一个前面暴露 OpenAI 形状的 API。

```figure
function-call-args
```

## 使用它

`code/main.py` 定义了一个规范的 `Tool` dataclass 和三个翻译器，分别产出 OpenAI、Anthropic 和 Gemini 的声明 JSON。随后它把每种形状的手工构造的供应商响应解析为同一个规范调用对象，证明语义在表层之下完全一致。运行它，并排对比三份声明。

值得关注的几点：

- 三个声明块只在最外层和字段名上不同。
- 三个响应块的区别在于调用所在的位置（顶层 `tool_calls`、`content[]` 块、`parts[]` 条目）。
- 一个 `canonical_call()` 函数从三种响应形状中提取 `{id, name, args}`。

## 交付它

本课产出 `outputs/skill-provider-portability-audit.md`。给定一个针对某家供应商的 function-calling 集成，该 skill 会生成可移植性审计：它依赖了哪家供应商的哪些限制、哪些字段需要改名，以及移植到另外两家时什么会坏掉。

## 练习

1. 运行 `code/main.py`，验证三份供应商声明 JSON 都序列化同一个底层 `Tool` 对象。修改规范工具，增加一个 enum 参数，确认只有 Gemini 翻译器需要处理 OpenAPI 怪癖。

2. 为每家供应商增加一个 `ListToolsResponse` 解析器，提取模型在 `list_tools` 或发现调用之后返回的工具列表。OpenAI 原生没有这一项；请记下这种不对称。

3. 实现 `tool_choice` 转换：把规范的 `ToolChoice(mode="force", tool_name="x")` 映射到三种供应商形状。然后映射 `mode="any"` 和 `mode="none"`。对照本课的差异表检查。

4. 从三家供应商中挑一家，从头到尾读它的 function-calling 指南。找出其 schema 规范中另外两家不支持的一个字段。候选：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 写一个测试向量：一个参数违反已声明 schema 的 tool call。让它跑一遍各家供应商的校验器（第 01 课的标准库实现可作为代理），记录哪些错误会触发。记录在严格性方面你会为生产选择哪家供应商。

## 关键术语

| 术语 | 人们怎么说 | 它实际指什么 |
|------|----------------|------------------------|
| Function calling | “工具使用” | 供应商层面用于发出结构化 tool call 的 API |
| Tool declaration | “工具规格” | 名称 + 描述 + JSON Schema 输入载荷 |
| `tool_choice` | “强制 / 禁止” | auto / required / none / 指定名称模式 |
| Strict mode | “schema 强制执行” | OpenAI 的标志，用受约束解码匹配 schema |
| `tool_use` block | “Anthropic 的调用形状” | 带 id、name、input 的内联内容块 |
| `functionCall` part | “Gemini 的调用形状” | 包含 name、args、id 的 `parts[]` 条目 |
| Arguments-as-string | “字符串化 JSON” | OpenAI 以 JSON 字符串而非对象返回参数 |
| Parallel tool calls | “单轮扇出” | 一条 assistant 消息中的多个 tool call |
| Refusal | “模型拒绝” | 仅严格模式下出现的拒绝块，替代调用 |
| OpenAPI 3.0 subset | “Gemini schema 怪癖” | Gemini 使用带细微差异的类 JSON Schema 方言 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — 包括严格模式与并行调用的权威参考
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 与 `tool_result` 块语义
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — 并行调用、唯一 id 与 OpenAPI 子集
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的企业级接口面
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式 schema 强制执行的细节
