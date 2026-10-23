# 智能体状态机——图、节点与检查点

> 手写的 ReAct 循环就是一个 `while True`。同样的循环写成显式图之后，你就可以对它做检查点、中断、分支和时间旅行。智能体本身没有变，变的是包在它外面的运行框架。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**Time:** ~75 minutes

## 问题

你上线了一个 function calling 智能体。它跑了三轮都没问题，然后出了岔子：模型调用了一个返回 500 的工具，用户中途改了主意，或者智能体未经人工签核就决定给一笔订单退款。`while True:` 循环没有任何钩子。你没法暂停它，没法回退它，也没法分叉出「如果模型当时选了另一个工具会怎样」。一旦走出 demo 阶段，这个智能体就变成一个黑盒——要么成功了，要么没有，中间发生了什么无从复现。

想通之后，下一步就显而易见了。智能体本来就是一个状态机——系统提示词，加上消息历史，加上待执行的工具调用，加上下一个动作。把这个状态机显式化：用节点表示「模型思考」「工具运行」「人工审批」，用边表示它们之间的条件跳转。图一旦显式，运行框架就能免费得到四样东西：检查点（在步骤之间保存状态）、中断（暂停等待人工）、流式（流式输出 token 和中间事件）和时间旅行（回到之前的状态，换一个分支重试）。

这一抽象的代表实现是 LangGraph。它不是 LangChain 意义上的智能体框架（「这里有个 AgentExecutor，祝你好运」），而是一个图运行时，状态、持久化和中断都是一等公民。智能体循环由你画出来，而不是手写出来。

## 概念

![LangGraph StateGraph：节点、边与检查点存储器](../assets/langgraph-stategraph.svg)

一个 `StateGraph` 包含三样东西。

1. **State（状态）。** 一个贯穿整张图的类型化字典（TypedDict 或 Pydantic 模型）。每个节点接收完整状态并返回部分更新，LangGraph 按字段用 *reducer* 合并这些更新——需要累加的列表用 `operator.add`，默认行为是覆盖。
2. **Nodes（节点）。** 形如 `state -> partial_state` 的 Python 函数。每个节点是一个离散步骤：「调用模型」「运行工具」「总结」。
3. **Edges（边）。** 节点之间的跳转。静态边只指向一个地方；条件边接收路由函数 `state -> next_node_name`，让图能根据模型输出分支。

你编译这张图。编译会固定拓扑、挂载检查点存储器（可选，但生产环境必不可少），并返回一个可运行对象。调用时传入初始状态和 `thread_id`。每一步执行都会持久化一个检查点，以 `(thread_id, checkpoint_id)` 为键。

### 四种超能力

**Checkpointing（检查点）。** 每次节点跳转都把新状态写入存储（测试用内存，生产用 Postgres/Redis/SQLite）。用相同的 `thread_id` 再次调用图即可恢复，图会从暂停的地方继续。

**Interrupts（中断）。** 用 `interrupt_before=["human_review"]` 标记一个节点，执行会在该节点运行前停下。状态会持久化。你的 API 回复用户「等待批准」。之后对同一 `thread_id` 发起请求并带上 `Command(resume=...)`，执行就会继续。

**Streaming（流式）。** `graph.stream(state, mode="updates")` 在状态增量产生时逐个产出。`mode="messages"` 流式输出模型节点内部的 LLM token。`mode="values"` 产出完整快照。在 UI 上展示什么，由你决定。

**Time-travel（时间旅行）。** `graph.get_state_history(thread_id)` 返回完整的检查点日志。把任意一个先前的 `checkpoint_id` 传给 `graph.invoke`，就能从那个点分叉。它非常适合调试（「如果模型当时选了工具 B 会怎样？」），也适合回放生产轨迹做回归测试。

### reducer 才是关键

每个状态字段都有一个 reducer。大多数默认值就够了——新值覆盖旧值。但消息列表需要 `operator.add`，让新消息追加而不是替换。并行边的更新也通过 reducer 合并。如果两个节点都更新 `messages`，而你又忘了 `Annotated[list, add_messages]`，第二个会静默胜出，你会丢掉半轮对话。reducer 是这个库里唯一微妙的地方；弄对它，其余部分自然组合。

### 四个节点组成的 ReAct 图

生产级的 ReAct 智能体就是四个节点、两条边：

1. `agent`——用当前消息历史调用 LLM。返回 assistant 消息（可能包含 tool_calls）。
2. `tools`——执行最后一条 assistant 消息里所有的 tool_calls，并把工具结果作为 tool 消息追加。
3. 从 `agent` 出发的条件边：最后一条消息带 tool_calls 就路由到 `tools`，否则路由到 `END`。
4. 从 `tools` 回到 `agent` 的静态边。

就这样。大约 40 行代码，你就得到完整的 ReAct 循环（Thought → Action → Observation → Thought → …），外加检查点、中断和流式。

### StateGraph 与 Send（扇出）

`Send(node_name, state)` 让节点派发并行子图。例如：智能体决定同时查询三个检索器。每个 `Send` 都会为目标节点派生一次并行执行；它们的输出通过状态 reducer 合并。LangGraph 就是这样在不使用线程原语的情况下表达 orchestrator-workers（编排者-工作者）模式。

### 子图

一个编译后的图可以作为另一个图的节点。外层图只看到一个节点；内层图有自己的状态和检查点。团队就是这样构建 supervisor-worker 智能体的：supervisor 图把用户意图路由到按领域划分的 worker 子图。

```figure
l5-state-graph-ledger
```

## 构建它

### 步骤 1：状态与节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 就是让消息列表累加而非覆盖的 reducer。忘了它是 LangGraph 最常见的 bug。

### 步骤 2：用线程运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每次更新都是一个字典 `{node_name: state_delta}`。前端可以把这些更新流式推给 UI，让用户看到「智能体正在思考……调用 search_web……拿到结果……作答」。

### 步骤 3：加入人机回环中断

标记一个节点，让执行在它运行前暂停。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

状态、检查点和线程在中断期间全部持久化。除了执行过程中，没有任何东西留在内存里。

### 步骤 4：用时间旅行调试

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

输入传 `None` 表示从指定检查点重放；传入值则先把它作为更新追加到该检查点的状态，再继续执行。这样就能复现一次出错的智能体运行，而不必重跑整段对话。

### 步骤 5：为生产环境更换检查点存储器

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis 和 Postgres 的检查点实现都已内置。`MemorySaver` 只用于测试。任何需要跨重启保留的东西，都应该用真正的存储。

## 技能

> 你把智能体建构成图，而不是 `while True` 循环。

在伸手去拿 LangGraph 之前，先做 60 秒设计：

1. **命名节点。** 每个离散的决策或带副作用的动作都是一个节点：「智能体思考」「工具运行」「审核者批准」「响应流式输出」。如果列不出来，说明任务还没形成智能体该有的形状。
2. **声明状态。** 用最小的 TypedDict，为每个列表字段配 reducer。不要把一切都塞进 `messages`；把任务特有的字段（工作中的 `plan`、`budget` 计数器、`retrieved_docs` 列表）提到顶层。
3. **画出边。** 除非下一步取决于模型输出，否则都用静态边。每条条件边都需要一个带命名分支的路由函数。
4. **一开始就选好检查点存储器。** 测试用 `MemorySaver`，其他情况用 Postgres/Redis/SQLite。没有检查点存储器就别上线——没有它就没有恢复、没有中断、没有时间旅行。
5. **在工具运行之前决定是否中断，而不是之后。** 审批放在进入带副作用节点的边上，这样能在造成危害前取消；校验放在模型输出的边上，这样能以较低代价拒绝错误调用。
6. **默认流式。** UI 用 `mode="updates"`，模型节点内部的 token 级流式用 `mode="messages"`，评估期间的完整快照用 `mode="values"`。

拒绝上线没有检查点存储器的 LangGraph 智能体。拒绝上线在副作用 *之后* 才中断的智能体。拒绝上线 `messages` 字段没有把 `add_messages` 作为 reducer 的智能体。

## 练习

1. **简单。** 用计算器工具和网页搜索工具实现上面的四节点 ReAct 图。验证对于两轮对话，`list(app.get_state_history(config))` 至少返回四个检查点。
2. **中等。** 添加一个在 `agent` 之前运行的 `planner` 节点，把结构化的 `plan: list[str]` 写入状态。让 `agent` 把计划步骤标记为完成。如果 `plan` 在检查点恢复后丢失（reducer 写错），测试应当失败。
3. **困难。** 用 `Send` 构建一个在三个子图（`researcher`、`writer`、`reviewer`）之间路由的 supervisor 图。每个子图都有自己的状态和检查点存储器。在外层图上加 `interrupt_before=["writer"]`，让人工可以批准研究简报。确认从先前检查点进行时间旅行时，只重跑分叉出来的那个分支。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| StateGraph | 「LangGraph 那张图」 | 编译之前，你向其中添加节点和边的构建器对象。 |
| Reducer | 「字段怎么合并」 | 节点为该字段返回更新时应用的函数 `(old, new) -> merged`；默认覆盖，`add_messages` 则追加。 |
| Thread | 「一个会话 ID」 | 一个 `thread_id` 字符串，划定一个会话的所有检查点范围。 |
| Checkpoint | 「一个暂停的状态」 | 节点跳转后完整图状态的持久化快照，以 `(thread_id, checkpoint_id)` 为键。 |
| Interrupt | 「暂停等人工」 | `interrupt_before` / `interrupt_after` 在节点边界停止执行；用 `Command(resume=...)` 恢复。 |
| Time-travel | 「从之前的步骤分叉」 | `graph.invoke(None, config_with_old_checkpoint_id)` 从该检查点向前重放。 |
| Send | 「并行子图派发」 | 节点可以返回的构造器，用于派生目标节点的 N 个并行执行。 |
| Subgraph | 「把编译好的图当作节点」 | 作为另一个图中节点使用的编译后 StateGraph；保留自己的状态作用域。 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)——StateGraph、reducer、检查点存储器和中断的权威参考。
- [LangGraph 概念：state、reducers、checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/)——本课使用的思维模型，直接来自官方。
- [LangGraph 持久化与检查点](https://langchain-ai.github.io/langgraph/concepts/persistence/)——关于 Postgres/SQLite/Redis 存储、检查点命名空间和 thread ID 的细节。
- [LangGraph 人机回环（Human-in-the-loop）](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)——`interrupt_before`、`interrupt_after`、`Command(resume=...)` 以及编辑状态的模式。
- [Yao 等，《ReAct: Synergizing Reasoning and Acting in Language Models》（ICLR 2023）](https://arxiv.org/abs/2210.03629)——每个 LangGraph 智能体都在实现的模式；读它了解推理轨迹的来由。
- [Anthropic——Building effective agents（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents)——该优先选用哪些图形状（链、路由、orchestrator-workers、evaluator-optimizer），以及何时选用。
- 第 11 阶段 · 09（Function Calling）——每个 LangGraph 智能体节点都会复用的工具调用原语。
- 第 11 阶段 · 14（Model Context Protocol）——通过 MCP 适配器接入 LangGraph `ToolNode` 的外部工具发现机制。
- 第 11 阶段 · 17（Agent 框架权衡）——何时选 LangGraph 而不是 CrewAI、AutoGen 或 Agno。
