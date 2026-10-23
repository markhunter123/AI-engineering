# 有状态图编排——持久化执行与 checkpoint

> 智能体就是一台状态机：节点是函数，边是转移，状态在每个节点之后写入 checkpoint。任何失败都能从最后一个成功的 checkpoint 恢复。LangGraph 是 2026 年这种底层有状态编排模型的参考实现。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## 学习目标

- 描述 LangGraph 的核心模型：带类型化状态的状态机、函数节点、条件边，以及节点执行后的 checkpoint。
- 说出文档强调的四种能力：持久化执行、流式输出、human-in-the-loop、全面的记忆。
- 解释 LangGraph 支持的三种编排拓扑：supervisor、peer-to-peer（swarm）、hierarchical（嵌套子图）。
- 用标准库实现一个状态图，包含类型化状态、条件边，以及 checkpoint/恢复循环。

## 问题

智能体和工作流有同一个问题：一次 40 步的运行在第 38 步失败时，你希望从第 38 步恢复，而不是从头开始。把状态当作二等公民的模型，逼得运维人员在一个默认每次全新运行的库外面拼凑重试逻辑。

LangGraph 的设计答案是：状态是一等公民的类型化对象，变更显式声明，每个节点之后都持久化 checkpoint。恢复就是一次 `load_state(session_id)` 调用。

## 概念

### 图

一个图由以下要素定义：

- **状态类型。** 一个带类型的 dict（或 Pydantic 模型），每个节点都会读取并修改它。
- **节点。** 纯函数 `(state) -> state_update`。返回值在节点返回后合并进状态。
- **边。** 节点之间的条件转移或直接转移。
- **入口与出口。** `START` 和 `END` 哨兵节点标记边界。

示例：一个包含 `classify`、`refund`、`bug`、`sales`、`done` 节点的智能体——把路由工作流表示成一张图。

### 持久化执行

每个节点返回后，运行时把状态序列化，写入 checkpointer（SQLite、Postgres、Redis 或自定义实现）。在第 N 步失败时，运行时可以 `resume(session_id)`，带着精确的状态从第 N+1 步继续。

LangGraph 文档明确列出了依赖这一能力的生产用户：Klarna、Uber、J.P. Morgan。卖点不是图的形状，而是图的形状加上 checkpoint 之后，恢复变得廉价。

### 流式输出

每个节点都可以产出部分输出。图按节点增量向调用方推送事件，UI 随图的运行实时更新。

### 人类参与（human-in-the-loop）

在节点之间检查和修改状态。实现方式：在关键节点前暂停，把状态展示给人，接受修改，然后恢复。checkpointer 让这件事很容易，因为状态本来就已经序列化了。

### 记忆

短期记忆（一次运行内——对话历史存放在状态里）与长期记忆（跨运行——通过 checkpointer 加上独立的长期存储持久化）。LangGraph 通过工具与外部记忆系统（Mem0、自定义）集成。

### 三种拓扑

1. **Supervisor。** 中央路由 LLM 把任务分派给专家子智能体。`langgraph-supervisor` 里提供 `create_supervisor()`（不过 LangChain 团队在 2026 年建议直接通过 tool calling 实现，以获得更多上下文控制）。
2. **Swarm / peer-to-peer。** 智能体通过共享的工具面直接交接，没有中央路由器。
3. **Hierarchical。** 监督者管理子监督者，用嵌套子图实现。

### 这个模式会在哪里出错

- **checkpoint 太小。** 只对对话轮次做 checkpoint，工具状态和记忆写入就无法恢复。完整状态必须可序列化。
- **非确定性节点。** 恢复的前提是相同输入产生相同的状态更新。随机种子、墙上时钟、外部 API 都必须被捕获。
- **滥用条件边。** 每条边都是条件边的图，是一台无法推理的状态机。优先使用线性链，偶尔分支。

```figure
langgraph-state
```

## 构建它

`code/main.py` 用标准库实现一个状态图：

- `State` —— 带类型的 dict，包含 `messages`、`step`、`route`、`output`、`human_approval`。
- `Node` —— 可调用对象，接收状态并返回更新 dict。
- `StateGraph` —— 节点 + 边 + 条件边 + 运行 + 恢复。
- `SQLiteCheckpointer`（内存中的模拟实现）—— 每个节点之后序列化状态；`load(session_id)` 恢复。
- 演示图：classify -> branch(refund / bug / sales) -> human gate -> send。

运行它：

```
python3 code/main.py
```

trace 显示了第一次运行在 human gate 处失败、状态被持久化，然后恢复并产出最终输出。

## 使用它

- **LangGraph** —— 参考实现，可用于生产。使用 `create_react_agent`、`create_supervisor`，或自己建图。
- **AutoGen v0.4**（第 14 课）—— actor 模型方案，适合高并发场景。
- **Claude Agent SDK**（第 17 课）—— 托管 harness，内置 session 存储。
- **自建** —— 当你需要精确控制状态形状或 checkpointer 后端时。

## 交付它

`outputs/skill-state-graph.md` 在任意目标运行时中生成 LangGraph 风格的状态图，并接好 checkpoint 与恢复。

## 练习

1. 增加一条从 `classify` 到 `end` 的条件边：分类置信度低于阈值时触发。在人工手动设置 `route` 之后恢复运行。
2. 把模拟 SQLite 的假实现换成真正的 SQLite checkpointer。测量每一步的序列化开销。
3. 实现并行边：两个节点并发运行，用自定义 reducer 合并。不可变状态在这里带来了什么？
4. 阅读 `langgraph-supervisor` 参考实现。把玩具移植到 `create_supervisor`，比较两者的 trace 形状。
5. 加入流式输出：每个节点运行时产出部分状态。增量到达时立即打印。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 状态图（state graph） | 「智能体即状态机」 | 类型化状态 + 节点 + 边 + reducer |
| Checkpointer | 「持久化后端」 | 在每个节点后序列化状态，使恢复成为可能 |
| Reducer | 「状态合并器」 | 把当前状态与节点更新合并的函数 |
| 条件边 | 「分支」 | 由作用于状态的函数选出的边 |
| 子图 | 「嵌套图」 | 作为节点用在另一张图里的图 |
| 持久化执行 | 「从失败处恢复」 | 从最后一个成功节点带着精确状态重启 |
| Supervisor | 「路由 LLM」 | 面向专家子智能体的中央分派器 |
| Swarm | 「P2P 智能体」 | 智能体通过共享工具交接；没有中央路由器 |

## 延伸阅读

- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) —— 参考文档
- [langgraph-supervisor 参考](https://reference.langchain.com/python/langgraph/supervisor/) —— supervisor 模式 API
- [AutoGen v0.4，Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) —— actor 模型替代方案
- [Claude Agent SDK 概览](https://platform.claude.com/docs/en/agent-sdk/overview) —— session 存储与子智能体
