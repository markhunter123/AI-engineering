# Agent 的 actor 模型——异步消息与类型化运行时

> 把 Agent 当作 actor：异步消息交换、事件驱动处理器、故障隔离、天然的并发。AutoGen v0.4（Microsoft Research，2025 年 1 月）围绕这一模型重新设计了 Agent 编排；该框架现已进入维护模式，Microsoft Agent Framework（2025 年 10 月公开预览）是其生产级继任者。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## 学习目标

- 描述 actor 模型：Agent 即 actor，消息是唯一的 IPC，每个 actor 的失败相互隔离。
- 说出 AutoGen v0.4 的三层 API——Core、AgentChat、Extensions——以及各自的用途。
- 解释为什么把消息投递与处理解耦，能带来故障隔离和天然并发。
- 用 Python 实现一个标准库 actor 运行时，并把双 Agent 代码审查流程移植到上面。

## 问题

大多数 Agent 框架是同步的：一个 Agent 生产，另一个消费，都在同一个调用栈里。失败会炸掉整个栈。并发是后来硬加上去的。分布式需要重写。

AutoGen v0.4 的答案是 actor 模型。每个 Agent 是一个带私有收件箱的 actor。消息是唯一的交互方式。运行时把投递与处理解耦。失败被隔离在单个 actor 内。并发是原生的。分布式只是换一种传输方式。

## 概念

### Actor

一个 actor 拥有：

- 私有状态（外部永远不能直接访问）。
- 一个收件箱（消息队列）。
- 一个处理器：`receive(message) -> effects`，effect 可以是「回复」「发送给其他 actor」「创建新 actor」「更新状态」「停止自身」。

两个 actor 不能共享内存。它们只能发送消息。

### 三层 API

AutoGen v0.4 把接口面分成三层：

1. **Core。** 底层 actor 框架。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换，事件驱动。
2. **AgentChat。** 任务驱动的高层 API（取代 v0.2 的 ConversableAgent）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** 集成——OpenAI、Anthropic、Azure、工具、记忆。

### 为什么解耦很重要

在 v0.2 模型中，同步调用 `agent_a.chat(agent_b)` 会让 agent_a 阻塞，直到 agent_b 返回。在 v0.4 中，`send(agent_b, msg)` 把消息放进 agent_b 的收件箱，然后立即返回。运行时稍后投递。这带来三个后果：

- **故障隔离。** agent B 崩溃不会让 agent A 崩溃——运行时在 B 的处理器里捕获失败，并决定怎么处理（记录、重试、死信）。
- **天然并发。** 同时有许多消息在途；actor 并发处理自己的收件箱。
- **天生适合分布式。** 无论 actor 在进程内还是在另一台主机上，收件箱 + 传输都是同一个抽象。

### 拓扑

- **RoundRobinGroupChat。** Agent 按固定顺序轮流执行。
- **SelectorGroupChat。** 选择器 Agent 根据对话上下文决定下一个是谁。
- **Magentic-One。** 面向网页浏览、代码执行、文件处理的参考多 Agent 团队，构建在 AgentChat 之上。

### 可观测性

内置 OpenTelemetry 支持。每条消息发出一个 span；Tool call 按 2026 年 OTel GenAI 语义约定（第 23 课）携带 `gen_ai.*` 属性。

### 状态：维护模式

2026 年初：AutoGen v0.7.x 对研究和原型开发来说是稳定的。微软已将活跃开发转移到 Microsoft Agent Framework，即生产级继任者（2025 年 10 月 1 日公开预览；1.0 GA 原定于 2026 年第一季度末）。AutoGen 的模式可以干净地向前移植——actor 模型才是持久的思想。

```figure
actor-mailbox
```

## 构建它

`code/main.py` 实现一个标准库 actor 运行时：

- `Message` —— 带类型的负载，包含 `sender`、`recipient`、`topic`、`body`。
- `Actor` —— 抽象类，含 `receive(message, runtime)`。
- `Runtime` —— 带共享队列、投递、故障隔离的事件循环。
- 双 actor 演示：`ReviewerAgent` 审查代码，`ChecklistAgent` 执行检查清单；两者交换消息直到达成共识。

运行它：

```
python3 code/main.py
```

trace 展示了消息投递、一个 actor 中的模拟故障（没有让另一个崩溃），以及收敛到共同结论。

## 使用它

- **AutoGen v0.4/v0.7**（维护模式）—— 对研究、原型和多 Agent 模式是稳定的。
- **Microsoft Agent Framework** —— 生产级继任者（2025 年 10 月公开预览）；在更新的 API 中保留了同样的 actor 模型思想。
- **LangGraph swarm 拓扑**（第 13 课）—— 通过共享工具交接的类似模式。
- **自建 actor 运行时** —— 当你需要特定的传输方式（NATS、RabbitMQ、gRPC）时。

## 交付它

`outputs/skill-actor-runtime.md` 为给定的多 Agent 任务生成一个最小 actor 运行时，外加团队模板（RoundRobin 或 Selector）。

## 练习

1. 增加死信队列：当处理器抛异常时，把失败的消息停放起来供人工检查。在你这个玩具里 DLQ 被命中的频率如何？
2. 实现 `SelectorGroupChat`：一个选择器 actor 根据对话状态决定谁来处理下一条消息。
3. 增加分布式传输：把进程内队列换成 JSON-over-HTTP 服务器，让 actor 可以运行在不同进程中。
4. 为每条消息接一个 OTel span（或空操作替身）。按第 23 课发出 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的架构文章。把你的玩具移植到真正的 `autogen_core` API。你跳过了哪些在生产中很重要的东西？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Actor | 「Agent」 | 私有状态 + 收件箱 + 处理器；没有共享内存 |
| 消息 | 「事件」 | 带类型的负载；actor 之间唯一的交互方式 |
| 收件箱 | 「邮箱」 | 每个 actor 的待处理消息队列 |
| 运行时 | 「Agent 宿主」 | 路由消息并隔离故障的事件循环 |
| Topic | 「频道」 | actor 之间命名的发布-订阅路由 |
| 故障隔离 | 「让它崩溃」 | 一个 actor 失败不会拖垮其他 actor |
| RoundRobinGroupChat | 「固定轮转团队」 | Agent 按顺序轮流执行 |
| SelectorGroupChat | 「按上下文路由的团队」 | 选择器挑选下一个执行者 |
| Magentic-One | 「参考团队」 | 面向网页 + 代码 + 文件的多 Agent 小队 |

## 延伸阅读

- [AutoGen v0.4，Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) —— 重新设计的文章
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) —— 图形态的替代方案
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— AutoGen 默认发出的 span
