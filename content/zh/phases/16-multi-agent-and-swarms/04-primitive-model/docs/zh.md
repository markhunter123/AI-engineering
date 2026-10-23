# 多 Agent 原语模型

> 只有四个原语——Agent、交接（handoff）、共享状态、编排器——它们张成一个四维设计空间；2026 年发布的主流多 Agent 框架（AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework）都是这个空间里的点。本课从零构建这四个原语，用一个玩具系统跑遍全部四种编排方式，再把每个主流框架映射到同一组坐标轴上，让你用一段话读懂任何新发布的框架。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 (Agent Engineering), Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 minutes

## 问题

每六个月就会有一个新的多 Agent 框架发布。2023 年的 AutoGen，2024 年的 CrewAI，2024 年的 LangGraph 和 OpenAI Swarm，2025 年 4 月的 Google ADK，2026 年 2 月的 Microsoft Agent Framework RC。每份新闻稿都宣称自己是「正确的抽象」。

如果你试图一个一个地学，你会精疲力竭。API 看起来各不相同。文档对「Agent」是什么的定义互相矛盾。一个框架把它的共享内存称为「黑板」，另一个称为「消息池」，第三个称为「StateGraph」。你开始怀疑这个领域只是在空转。

并非如此。在营销包装之下，那四个原语是稳定的。学一次，就能用一段话读懂每个新框架。

## 概念

### 四个原语

1. **Agent（Agent）**——一段 system prompt 加一个工具列表。无状态；每次运行都从它的 system prompt 和当前消息历史开始。
2. **交接（Handoff）**——把控制权从一个 Agent 结构化地转移给另一个 Agent。从机制上说，它可以是一次返回新 Agent 的 tool call，也可以是一条按条件跳转的图边。
3. **共享状态（Shared state）**——任何能被不止一个 Agent 读取（有时还能写入）的数据结构。消息池、黑板、键值存储、向量记忆。
4. **编排器（Orchestrator）**——决定下一个由谁发言的角色。可选方式：显式图（确定性）、LLM 发言者选择器（软性）、上一个发言者的交接调用（OpenAI Swarm），或基于队列的调度器（群体架构）。

这就是全部设计空间。每个框架只是为每条轴挑选默认值，其余都是表面语法。

### 2026 年的每个框架如何映射到它

| 框架 | Agent | 交接 | 共享状态 | 编排器 |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | 工具返回 Agent | 调用方自己的问题 | LLM 的下一次交接调用 |
| AutoGen v0.4 / AG2 | `ConversableAgent` | GroupChat 上的发言者选择器 | 消息池 | 选择器函数（LLM 或轮询） |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | 任务输出串联 | 管理者 LLM 或静态顺序 |
| LangGraph | 节点函数 | 图边 + 条件 | `StateGraph` reducer | 图本身，确定性 |
| Microsoft Agent Framework | Agent + 编排模式 | 因模式而异 | thread / context | 因模式而异 |
| Google ADK | Agent + A2A 卡片 | A2A 任务 | A2A artifact | 由宿主决定 |

表面差异看起来巨大。往下一层看：还是那四个旋钮。

### 为什么这很重要

一旦你看见了这些原语，框架对比就变成一张简短的检查清单：

- 编排器是信任 LLM 来做路由（Swarm），还是在代码里固定路由（LangGraph）？
- 共享状态是完整历史（GroupChat）还是投影视图（StateGraph reducer）？
- Agent 之间能否修改彼此的 prompt（CrewAI 管理者），还是只能交接（Swarm）？

这三个问题能回答 80% 的「哪个框架适合某个问题」。你不再去寻找「最好的多 Agent 框架」，而是开始针对你真正关心的那条轴做设计。

### 无状态洞见

除了共享状态，每个原语都是无状态的。Agent 是 (prompt, tools) 的函数。交接是一次 function call。编排器是一个调度器。**系统中唯一有状态的东西就是共享状态。** 所有有趣的 bug 都住在那里：记忆投毒（第 15 课）、消息顺序、版本管理、写入竞争。

把共享状态藏起来的框架（Swarm）把问题推给调用方。把共享状态集中起来的框架（LangGraph checkpoint、AutoGen 消息池）让它可检查，但把协调成本转移到了共享状态的实现上。

### 单个原语剖析

#### Agent

```
Agent = (system_prompt, tools, model, optional_name)
```

没有记忆。没有状态。两个拥有相同 system prompt 和工具的 Agent 可以互换。所有看起来像是「每个 Agent 各自的状态」的东西，实际上都在共享状态或交接协议里。

#### 交接

```
Handoff = (from_agent, to_agent, reason, payload)
```

有三种实现占主导地位：

- **函数返回**——工具返回下一个 Agent。这是 OpenAI Swarm 的模式。Agent 把路由信息放进自己的工具 schema 中。
- **图边**——LangGraph。边是声明式的。LLM 产出一个值，条件选择下一个节点。
- **发言者选择**——AutoGen GroupChat。一个选择器函数（有时它本身就是一次 LLM 调用）读取消息池并挑选下一个发言者。

#### 共享状态

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

最低限度是一个消息列表。通常还有更多：结构化 artifact（CrewAI 任务输出）、带类型的上下文（LangGraph reducer）、外部记忆（MCP、向量数据库）。

两种拓扑：**完整消息池**（每个 Agent 看到每条消息）和**投影视图**（Agent 看到按角色划分的视图）。完整消息池简单，但扩展性差。投影视图可扩展，但需要提前设计 schema。

#### 编排器

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种形态：

- **静态**——图在构建时固定（LangGraph 确定性、CrewAI Sequential）。
- **LLM 选择**——LLM 读取消息池并选择下一个发言者（AutoGen、CrewAI Hierarchical）。
- **交接驱动**——当前 Agent 通过调用交接工具来决定（Swarm）。
- **队列驱动**——工作者从共享队列拉取任务；没有显式的下一发言者（群体架构、Matrix）。

### 框架之间变化的部分

一旦原语固定下来，剩下的设计决策是：

- **记忆策略**——临时记忆还是持久 checkpoint（LangGraph checkpointer）。
- **安全边界**——谁可以批准一次交接（human-in-the-loop）。
- **成本核算**——每个 Agent 的 token 预算。
- **可观测性**——追踪交接、持久化状态以便回放。

这些都可以构建在原语之上。它们本身都不是新原语。

```figure
a5-primitive-radar
```

## 构建它

`code/main.py` 用约 150 行标准库 Python 实现这四个原语。没有真实 LLM——每个 Agent 都是一个脚本化的策略，让注意力保持在协调结构上。

该文件导出：

- `Agent`——一个 dataclass，包含名称、system prompt、工具、策略函数。
- `Handoff`——一个返回新 Agent 的函数。
- `SharedState`——一个线程安全的消息池。
- `Orchestrator`——三个变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（模拟）。

演示把同一条三 Agent pipeline（研究 → 写作 → 审查）分别跑过三种编排器，最后打印消息池。你能看到，输出只在*谁决定下一个*上有所不同；Agent 和共享状态在各次运行中完全相同。

运行：

```
python3 code/main.py
```

预期输出：三次编排器运行，每个模式一次。每次打印最终消息池。如果研究员提前决定结束，交接驱动的那次运行会到达更少的 Agent——这就是 LLM 路由权衡的微缩版。

## 使用它

`outputs/skill-primitive-mapper.md` 是一个 skill，读取任何多 Agent 代码库或框架文档并返回四原语映射。在一个新框架发布时运行它，可以在深入阅读文档之前获得一段话的理解。

## 交付它

在采用新框架之前，先为它写出原语映射。如果你写不出来，要么文档不完整，要么该框架正在发明第五个原语（罕见——检查是否有你没见过的共享状态变体）。

把这份映射钉在你的架构文档里。新成员加入时，先发映射，再发 API 文档。框架版本变化时，diff 映射，而不是 diff changelog。

## 练习

1. 用不同的 Agent 策略运行 `code/main.py` 三次。观察编排器的选择如何改变哪些 Agent 被运行。
2. 实现第四种编排器类型：队列驱动的版本，由 Agent 轮询共享状态来获取工作。可能发生什么死锁？你如何检测它？
3. 拿 LangGraph 的快速入门（https://docs.langchain.com/oss/python/langgraph/workflows-agents）并用四个原语重写它。LangGraph 的哪些抽象是一一对应的，哪些只是便利包装？
4. 阅读 OpenAI Swarm 的 cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。识别四个原语中 Swarm 让哪一个用起来最顺手，又把哪一个推给了调用方。
5. 在这张表里找一个完全隐藏共享状态的框架。解释当 Agent 需要在多次交接之间协调、又不能重读历史时，会发生什么问题。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Agent | 「带工具的 LLM」 | 一个 `(system_prompt, tools, model)` 三元组。无状态。 |
| Handoff | 「控制权转移」 | 一次结构化调用，指定下一个 Agent 和可选的 payload。有三种实现：函数返回、图边、发言者选择。 |
| Shared state | 「记忆」/「上下文」 | 多 Agent 系统中唯一有状态的部分。消息池或黑板。 |
| Orchestrator | 「协调器」 | 决定下一个由谁运行的角色。静态图、LLM 选择器、交接驱动或队列驱动。 |
| Primitive | 「抽象」 | 每个框架都要参数化的四条轴之一。不是某个框架的功能特性。 |
| Message pool | 「共享聊天历史」 | 完整历史的共享状态。容易推理，但扩展性差。 |
| Projected state | 「作用域视图」 | 面向特定角色的共享状态视图。可扩展，但需要设计 schema。 |
| Speaker selection | 「下一个谁发言」 | 一种编排器模式：由一个函数（通常是 LLM）从一组 Agent 中挑选下一个。 |

## 延伸阅读

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) —— 对交接驱动编排最清晰的阐述
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/) —— GroupChat + 发言者选择是 LLM 选择式编排的参考
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) —— 图边编排与基于 reducer 的共享状态
- [CrewAI introduction](https://docs.crewai.com/en/introduction) —— role-goal-backstory Agent，Sequential / Hierarchical 流程
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2) —— Microsoft 把 v0.4 转为维护模式后，仍在活跃开发的 AutoGen v0.2 分支
