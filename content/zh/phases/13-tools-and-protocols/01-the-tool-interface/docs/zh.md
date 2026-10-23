# 工具接口——为什么智能体需要结构化 I/O

> 语言模型产出 token，程序执行动作。两者之间的鸿沟就是工具接口：一份让模型请求动作、由宿主实际执行的契约。2026 年的每一套技术栈——OpenAI、Anthropic 和 Gemini 上的 function calling，MCP 的 `tools/call`，A2A 的任务部分——都是同一个四步循环的不同编码。本课为这个循环命名，并展示运行它所需的最小机制。

**Type:** Learn
**Languages:** Python (stdlib, no LLM)
**Prerequisites:** Phase 11 (LLM completion APIs)
**Time:** ~45 minutes

## 学习目标

- 解释为什么一个只能生成文本的 LLM 无法仅凭自身对真实世界采取动作。
- 画出四步工具调用循环（描述 → 决策 → 执行 → 观察），并说明每一步由谁负责。
- 把工具描述写成三部分：名称、JSON Schema 输入，以及一个确定性的执行器函数。
- 区分纯工具和有副作用的工具，并说明这种划分对安全为何重要。

## 问题

LLM 输出的是下一个 token 的概率分布。这就是它的全部输出面。如果你问聊天模型「班加罗尔现在天气如何」，它能写出一句看似合理的话，却无法真正调用天气 API。这句话可能碰巧正确，也可能已经是三天前的旧信息。

填补这一鸿沟正是工具接口的用途。宿主程序——你的智能体运行时、Claude Desktop、ChatGPT、Cursor 或自定义脚本——向模型公布一组可调用的工具。当模型判断需要执行动作时，它会发出结构化载荷，指明工具名和参数。宿主解析载荷，真正执行工具，再把结果回传。循环持续进行，直到模型判断不再需要调用。

这份契约的第一版在 2023 年 6 月以 OpenAI 的 “functions” 参数形式发布。Anthropic 随后在 Claude 2.1 中加入 `tool_use` 块。几个月后 Gemini 加入 `functionDeclarations`。如今每个供应商都暴露同样的形状：输入是 JSON Schema 类型的工具列表，输出是 JSON 载荷形式的工具调用。Model Context Protocol（2024 年 11 月）将这份契约泛化，使一份工具注册表可以服务所有模型。A2A（2026 年 4 月，v1.0）为智能体之间的委派叠加了同一原语。

四步循环是这一切之下的不变量。第 13 阶段的其余内容都是它的展开。

## 概念

### 第一步：描述

宿主为每个工具声明三个字段。

- **名称。** 稳定、机器可读的标识符。用 `get_weather`，而不是 “weather thing”。
- **描述。** 一段自然语言简介。「当用户询问某个城市的当前状况时使用。不要用于历史数据。」
- **输入 schema。** 一个 JSON Schema 对象（2020-12 草案），描述工具的参数。

模型拿到这份清单。现代供应商会用各自特定的模板，把这些声明序列化进系统 prompt，因此作为调用方的你只接触结构化形式。

### 第二步：决策

给定用户消息和可用工具，模型在三种行为中选择一种。

1. **直接用文本回答。** 不发起工具调用。
2. **调用一个或多个工具。** 发出结构化调用对象。在 `parallel_tool_calls: true` 下（OpenAI 和 Gemini 默认开启，Anthropic 需显式开启），模型可以在同一轮中发出多个调用。
3. **拒绝。** 严格模式的结构化输出可以产生带类型的 `refusal` 块，而不是一次调用。

工具调用载荷有三个稳定字段：调用 `id`、工具 `name` 和 JSON `arguments` 对象。id 的存在是为了让宿主把稍后的结果与具体调用对应起来——当并行调用乱序返回时，这一点至关重要。

### 第三步：执行

宿主收到调用，按已声明的 schema 校验参数，然后运行执行器。参数无效意味着模型幻觉出了字段或用错了类型——这在弱模型上是非常常见的失败模式。生产环境宿主对无效参数通常采取三种做法之一：快速失败并把错误暴露给模型；用受约束的解析器修复 JSON；或把校验错误放进 prompt 重试模型。

执行器本身是普通代码。Python、TypeScript、一条 shell 命令、一次数据库查询。它产出一个结果，通常是字符串，但也可以是任意 JSON 值或结构化内容块（MCP 中的文本、图像或资源引用）。结果必须可序列化。

### 第四步：观察

宿主把工具结果追加到对话中（作为 `tool` 角色消息，带匹配的 `id`），然后重新调用模型。此时模型在上下文中已有工具输出，可以给出最终回答或请求更多调用。这个过程持续到模型不再发出调用，或宿主触达迭代次数的安全上限。

### 信任划分

从安全角度看，工具分为两类。

- **纯工具。** 只读、确定性、无副作用。`get_weather`、`search_docs`、`get_current_time`。可以安全地试探性调用。
- **有后果的工具。** 改变状态、花钱、触碰用户数据。`send_email`、`delete_file`、`execute_trade`。必须加闸门。

Meta 2026 年为智能体安全提出的 “Rule of Two” 指出：单轮最多只能组合以下三项中的两项——不可信输入、敏感数据、有后果的动作。工具接口正是你执行这条规则的地方——通过拒绝调用、要求用户确认或提升权限范围。完整的安全章节见第 13 阶段 · 15，智能体级权限策略见第 14 阶段 · 09。

### 循环在哪里运行

| 场景 | 谁负责描述 | 谁负责决策 | 谁负责执行 |
|---------|---------------|-------------|--------------|
| 单轮 function calling（OpenAI/Anthropic/Gemini） | 应用开发者 | LLM | 应用开发者 |
| MCP | MCP server | LLM（经由 MCP client） | MCP server |
| A2A | Agent Card 发布方 | 调用方智能体 | 被调用方智能体 |
| Web 浏览器（function-calling 智能体） | 浏览器扩展 / WebMCP | LLM | 浏览器运行时 |

无论在哪里，都是同样的四步。列名变了，结构没变。

### 为什么不直接让模型输出 JSON？

「让模型用 JSON 回复」是 function calling 之前的做法。它在前沿模型上的失败率约为 5% 到 15%，在更小的模型上高得多。失败模式包括缺括号、尾逗号、幻觉字段和类型错误。你随后需要 JSON 修复环节、重试或受约束的解码器。

原生 function calling 更好，原因有三。第一，供应商在精确的调用形状上对模型做了端到端训练，因此在严格模式下合法 JSON 的比例升至 98% 到 99%。第二，调用载荷位于自己的协议槽位，而不是自由文本内部——因此工具调用永远不会泄漏到用户可见的回复里。第三，供应商用受约束解码强制执行 schema 合规（OpenAI 的严格模式、Anthropic 的 `tool_use`、Gemini 的 `responseSchema`）。输出保证通过校验。

第 13 阶段 · 02 会并排走查三家供应商的 API。第 13 阶段 · 04 深入结构化输出。

### 熔断器

当模型停止发出调用，或宿主达到最大轮数时，循环终止。生产环境宿主把这个上限设在 5 到 20 轮之间。超过这个范围，你几乎肯定陷入了模型无法退出的循环。Claude Code 默认 20 轮，OpenAI Assistants 为 10 轮，Cursor 的 agent 模式为 25 轮。

另一种做法——无界循环——每隔半年就会以「智能体一夜之间烧掉 400 美元 API 调用费」的事故复盘形式出现。不要在缺少上限的情况下上线。

第 14 阶段 · 12 深入讲解错误恢复与自愈；第 17 阶段覆盖生产环境的速率限制。

### 第 13 阶段接下来走向哪里

- 第 02 到 05 课打磨供应商层面的工具调用表面。
- 第 06 到 14 课把循环泛化进 MCP。
- 第 15 到 18 课保护循环免受恶意服务器、对抗性用户和未认证远程认证面的攻击。
- 第 19 到 22 课把这一模式扩展到智能体间协作、可观测性、路由与打包。
- 第 23 课用所有原语交付一套完整生态。

其余每一课都是对这个四步循环的展开。请把它当作不变量记在心里。

```figure
tp-tool-loop
```

## 使用它

`code/main.py` 在没有 LLM 的情况下运行四步循环。一个假的「决策者」函数通过对用户消息做模式匹配来模拟模型；执行器、schema 校验器和观察步骤的框架都是真实的。运行它，可以看到完整的请求/响应编排与可打印的中间状态，然后在后续课程中把假决策者换成任意真实供应商。

值得关注的几点：

- 工具注册表中每个工具有三个字段：name、description、schema，以及一个执行器引用。
- 校验器是一个最小 JSON Schema 子集（类型、required、enum、min/max），只用标准库编写。第 13 阶段 · 04 提供更完整的实现。
- 循环把迭代次数限制在五次。生产智能体恰恰需要这种熔断器。

## 交付它

本课产出 `outputs/skill-tool-interface-reviewer.md`。给定一份工具定义草稿（名称 + 描述 + schema + 执行器轮廓），该 skill 会从循环适配度审计它：名称是否机器稳定，描述是否是完整的使用说明，schema 是否正确使用 JSON Schema 2020-12，以及纯工具与有后果工具的划分是否明确。

## 练习

1. 在 `code/main.py` 中新增第四个工具 `get_stock_price(ticker)`。把它的描述写成 “Use when the user asks for a current stock price by ticker. Do not use for historical prices or market summaries.”。运行框架，确认假决策者把提到 ticker 的查询路由到新工具。

2. 破坏 schema 校验器。传入一个 `arguments` 对象缺少必填字段的调用，确认宿主在执行前拒绝它。再传入一个带额外未知字段的调用。做决定：宿主应当拒绝还是忽略？用安全论证为你的选择辩护。

3. 把框架中的每个工具分类为纯工具或有后果工具。给需要的注册表条目加上 `consequential: true` 标记，并修改循环，使得每当选中一个有后果工具时打印一行 “would confirm with user”。这就是每个生产宿主都需要的确认闸门的形状。

4. 在纸上画出四步循环，并用上方的供应商列表为你最常用的客户端（Claude Desktop、Cursor、ChatGPT 或自定义技术栈）填表。与第 13 阶段 · 06 中 MCP 专用的变体对照。

5. 从头到尾读一遍 OpenAI 的 function-calling 指南。找出一个存在于请求中、但不在本课呈现的四步循环里的字段。解释它增加了什么，以及为什么它是便利而非本质。

## 关键术语

| 术语 | 人们怎么说 | 它实际指什么 |
|------|----------------|------------------------|
| Tool | “模型能调用的东西” | 名称 + JSON Schema 类型的输入 + 执行器函数构成的三元组 |
| Function calling | “原生工具使用” | 供应商层面的 API 支持，用于发出结构化工具调用而非散文 |
| Tool call | “模型的行动请求” | 模型发出的、带 `id`、`name`、`arguments` 的 JSON 载荷 |
| Tool result | “工具返回的内容” | 执行器的输出，封装在带匹配 id 的 `tool` 角色消息中 |
| Parallel tool calls | “一次调用许多个” | 单个模型轮次中的多个调用对象，彼此独立且可按 id 排序 |
| Strict mode | “保证 JSON” | 受约束解码，强制模型输出通过已声明 schema 的校验 |
| Pure tool | “只读工具” | 无副作用；可以安全地重复运行 |
| Consequential tool | “动作工具” | 改变外部状态；需要闸门、审计或用户确认 |
| Four-step loop | “工具调用周期” | 描述 → 决策 → 执行 → 观察 |
| Host | “智能体运行时” | 持有工具注册表、调用模型并运行执行器的程序 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — OpenAI 风格工具声明与调用形状的权威参考
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — Claude 的 `tool_use` / `tool_result` 块格式
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 中的 `functionDeclarations` 与并行调用语义
- [Model Context Protocol — Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) — 工具接口当前的无状态、供应商无关泛化
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 每个现代工具 API 都使用的 schema 方言
