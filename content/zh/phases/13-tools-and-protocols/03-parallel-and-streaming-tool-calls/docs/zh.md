# 并行工具调用与流式工具调用

> 三次独立的天气查询串行执行就是三次往返。并行执行时，总耗时坍缩为最慢的那一次调用。现在每家前沿供应商都能在单轮中发出多个工具调用。收益是真实的，管道是微妙的。本课走查两个部分：并行扇出和流式参数重组，重点讲 id 关联这个陷阱。

**Type:** Build
**Languages:** Python (stdlib, thread pool + streaming harness)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 minutes

## 学习目标

- 解释 `parallel_tool_calls: true` 为何存在，以及何时应关闭它。
- 在并行扇出期间，把流式参数块关联到正确的工具调用 id。
- 把不完整的 `arguments` 字符串重组成完整 JSON，且不提前解析。
- 运行三城市天气基准，展示串行与并行的延迟差异。

## 问题

没有并行调用时，回答「班加罗尔、东京和苏黎世天气如何」的智能体会这样做：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

三次 LLM 往返，每次还要付出执行器延迟。大约是理想墙钟时间的 4 倍。

有并行调用时：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

一次 LLM 往返。执行器时间是三者中的最大值，而不是总和。OpenAI、Anthropic 和 Gemini 上的生产基准显示，扇出工作负载的墙钟时间减少 60% 到 70%。

代价是关联复杂度。当三个调用乱序完成时，结果必须携带匹配的 `tool_call_id`，模型才能把它们对上。当结果流式到达时，你必须在执行前把部分参数片段组装成完整 JSON。Gemini 3 加入唯一 id 的部分原因，就是解决现实中两个并行调用指向同一工具时无法区分的问题。

## 概念

### 启用并行

- **OpenAI。** `parallel_tool_calls: true` 默认开启。设为 `false` 可强制串行。
- **Anthropic。** 通过 `disable_parallel_tool_use: false` 启用并行（Claude 3.5 及以上的默认值）。设为 `true` 即为串行。
- **Gemini。** 始终支持并行；`tool_config.function_calling_config.mode = "AUTO"` 让模型自行决定。

在以下情况关闭并行：工具之间存在顺序依赖（先 `create_file` 再 `write_file`）、一个调用的输出要作为另一个调用的输入，或者速率限制器无法承受扇出。

### id 关联

模型发出的每个调用都有 `id`。宿主返回的每个结果都必须带上同一个 id。没有它，结果就是有歧义的。

- **OpenAI。** 每条 tool 角色消息上的 `tool_call_id`。
- **Anthropic。** 每个 `tool_result` 块上的 `tool_use_id`。
- **Gemini。** 每个 `functionResponse` 上的 `id`（Gemini 3 及以上；Gemini 2 按名称匹配，同名并行调用会出问题）。

### 并发运行调用

宿主把每个调用的执行器放到自己的线程、协程或远程 worker 上运行。最简单的框架用线程池；生产环境用 asyncio 配合 `asyncio.gather` 或结构化并发。完成顺序不可预测——id 才是标识符。

一个常见 bug：按调用列表顺序而非完成顺序回复结果。这通常也能工作，因为模型只关心 `tool_call_id`；但如果某个结果丢失或重复，乱序提交会让调试更困难。最好按完成顺序回复，并显式携带 id。

### 流式工具调用

当模型流式输出时，`arguments` 分片到达。三个并行调用的三股块流会在线上交错。你需要为每个 id 维护一个累加器。

按供应商看形状：

- **OpenAI。** 每个块是 `choices[0].delta.tool_calls[i].function.arguments`（部分字符串）。块中带有 `index`（调用列表中的位置）。你按 index 累积，在 `id` 首次出现时记下它，并在 `finish_reason = "tool_calls"` 时解析 JSON。
- **Anthropic。** 流事件先是 `message_start`，然后每个类型为 `tool_use` 的块一个 `content_block_start`（包含 id、name 和空的 input）。`content_block_delta` 事件携带 `input_json_delta` 块。`content_block_stop` 关闭每个块。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 及以上）发出的块带 `functionCallId`，因此调用可以干净地交错。在 Gemini 3 之前，流式一次返回一个完整调用。

### 部分 JSON 与提前解析陷阱

`arguments` 不完整时不能解析。像 `{"city": "Beng` 这样的部分 JSON 无效，会抛出异常。正确的门控信号是供应商的调用结束信号：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop` 或 Gemini 的流结束事件。只有到那时才尝试 `json.loads`。更稳健的做法是使用增量 JSON 解析器，在结构完成时逐个产出事件；OpenAI 的流式指南推荐它来实现实时「思考中」指示器之类的 UX。花括号计数作为完整性测试并不可靠（引号字符串或转义内容中的花括号会导致误判），只能当作非正式的调试启发式。

### 乱序完成

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

宿主的回复仍然必须引用 id：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

在 OpenAI 或 Anthropic 上，回复中的顺序不影响正确性。Gemini 接受任意顺序，只要 id 匹配。

### 基准：串行 vs 并行

`code/main.py` 中的框架模拟三个延迟分别为 400、600 和 800 ms 的执行器。串行执行总共 1800 ms。并行执行为 max(400, 600, 800) = 800 ms。差异是常数而非比例，因此节省量会随工具数量增长。

现实中的注意事项：并行调用会给下游 API 带来压力。对一个有速率限制的服务做 10 路扇出会失败。第 13 阶段 · 17 覆盖网关级的背压；重试语义计划在未来阶段讲解。

### 流式扇出的墙钟时间

如果模型本身是流式的，你可以在某个调用的参数一完成时就开始执行，而不必等所有调用都定稿。这是 OpenAI 文档中提到、但并非所有 SDK 都暴露的优化。本课的框架做到了这一点：模拟流一旦产出完整的参数对象，宿主就启动该调用。

```figure
tp-parallel-fanout
```

## 使用它

`code/main.py` 分为两半。第一半使用 `concurrent.futures.ThreadPoolExecutor` 串行和并行地运行三次模拟天气调用，并打印墙钟时间。第二半重放一个假的流式响应——三个并行调用的 `arguments` 块交错在同一条流上——并用 `StreamAccumulator` 按 id 重组。没有 LLM，没有网络，只有重组逻辑。

值得关注的几点：

- 串行计时器达到 1.8 秒。在相同的假延迟下，并行计时器达到 0.8 秒。
- 累加器通过按 id 缓冲来处理乱序到达的块，只在每个调用的 JSON 完整时才解析。
- 执行器在某个 id 的参数定稿时立即启动，而不是等所有流结束。

## 交付它

本课产出 `outputs/skill-parallel-call-safety-check.md`。给定一个工具注册表，该 skill 会审计哪些工具可以安全并行、哪些存在顺序依赖、哪些会压垮下游速率限制——返回一份修订后的注册表，附带每个工具的 `parallel_safe` 标记。

## 练习

1. 运行 `code/main.py`，并改变模拟延迟。确认并行与串行的比例大致为 `max/sum`（由于线程调度、序列化和框架开销，真实运行会略微偏离理想值）。在什么样的延迟分布下，并行不再有意义？

2. 扩展累加器，通过丢弃缓冲区并发出 `cancelled` 事件来处理「调用在流中途被取消」的情况。哪家供应商明确记录了这种情况？查看 Anthropic 的 `content_block_stop` 语义和 OpenAI 的 `finish_reason: "length"` 行为。

3. 用 `asyncio.gather` 替换线程池。对两者做基准测试。由于上下文切换成本更低，你应该会在异步上看到小幅收益——但前提是执行器执行真实的 I/O。

4. 选择两个不应并行的工具（例如先 `create_file` 再 `write_file`）。向注册表添加一个 `ordering_dependency` 图，并基于该图对并行扇出做闸控。这是依赖感知调度的最小机制，未来的智能体工程阶段会将其形式化。

5. 阅读 OpenAI 的并行 function calling 章节和 Anthropic 的 `disable_parallel_tool_use` 文档。找出 Anthropic 建议关闭并行的那一类真实工具场景。（提示：对同一资源的有后果变更。）

## 关键术语

| 术语 | 人们怎么说 | 它实际指什么 |
|------|----------------|------------------------|
| Parallel tool calls | “单轮扇出” | 模型在一条 assistant 消息中发出多个工具调用 |
| `parallel_tool_calls` | “OpenAI 的开关” | 启用或禁用多次调用发出 |
| `disable_parallel_tool_use` | “Anthropic 的反向开关” | 退出标志；默认启用并行 |
| Tool call id | “关联句柄” | 每个调用的标识符，结果消息必须回显 |
| Accumulator | “流缓冲” | 按 id 为部分 `arguments` 块维护的字符串缓冲区 |
| Out-of-order completion | “最快者先到” | 并行调用以不可预测的顺序完成；id 是粘合剂 |
| Dependency graph | “顺序约束” | 输出会喂给其他工具输入的工具；不能并行 |
| Parse-early trap | “JSON.parse 爆炸了” | 试图解析不完整的 `arguments` 字符串 |
| `streamFunctionCallArguments` | “Gemini 3 特性” | 每次调用带唯一 id 的流式参数块 |
| Completion-order reply | “不要等所有调用” | 结果一到就回复，以 id 为键 |

## 延伸阅读

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 默认行为与退出标志
- [Anthropic — Parallel tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use) — `disable_parallel_tool_use` 与结果批量处理
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3 起按 id 关联的并行调用
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI 流的块式参数重组
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — 带 `input_json_delta` 的 `content_block_delta`
