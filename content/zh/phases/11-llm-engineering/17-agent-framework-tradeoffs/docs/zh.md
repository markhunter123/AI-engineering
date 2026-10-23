# Agent 框架权衡——图、角色与 Actor 编排

> 每个框架都在卖同一个 demo（研究 Agent 写出一份报告），也都在掩盖同一个 bug（状态 schema 与编排层打架）。选那个抽象与你问题形状相匹配的框架；其余的，都是你要写两遍的胶水代码。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 16 (LangGraph)
**Time:** ~45 minutes

## 问题

你有一个需要多次 LLM 调用的任务。也许是研究工作流（计划、搜索、总结、引用）；也许是代码审查 pipeline（解析 diff、批评、打补丁、验证）；也许是一个多轮助手，会订机票、写邮件、提交报销单。你选了一个框架。

三天后，你发现框架的抽象开始漏了。CrewAI 给了你角色，但当「研究员」需要把结构化计划交给「写手」时，它就跟你对着干。AutoGen 给了你 Agent 之间的聊天，却没有一等状态，于是你的 checkpoint 只能是对话日志的 pickle。LangGraph 给了你状态图，却逼你在还不知道 Agent 会做什么之前就命名每一次跳转。Agno 给了你单 Agent 抽象，一旦你想扇出到三个并发工作者，它就开始尖叫。

解法不是「挑最好的框架」，而是让框架的核心抽象匹配你问题的形状。本课就来画这张地图。

## 概念

![Agent 框架矩阵：核心抽象与问题形状](../assets/framework-matrix.svg)

2026 年的格局由四个框架主导。它们的核心抽象并不相同。

| 框架 | 核心抽象 | 最适合 | 最不适合 |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph`——类型化状态、节点、条件边、checkpoint 存储器。 | 需要显式状态和人机回环中断的工作流；需要时间旅行调试的生产 Agent。 | 拓扑未知、由角色驱动的松散头脑风暴。 |
| **CrewAI** | `Crew`——角色（目标、背景故事）、任务、流程（顺序或层级）。 | 带简短线性或层级计划的角色扮演、人设驱动工作流。 | 任何超出 crew 轮次历史的有状态需求；复杂分支。 |
| **AutoGen** | `ConversableAgent` 对——两个或更多 Agent 轮流发言，直到退出条件满足。 | 思考从对话中自然涌现的多 Agent *对话*（师生、提议者-批评者、执行者-审阅者）。 | 已知 DAG 的确定性工作流；任何需要跨重启持久状态的需求。 |
| **Agno** | `Agent`——单个 LLM + 工具 + 记忆，可组合成团队。 | 快速搭建的单 Agent 和轻量团队；多模态能力强，内置存储驱动。 | 深层、显式分支、带自定义 reducer 的图。 |

### 「抽象」到底意味着什么

一个框架的核心抽象，就是你在白板上向别人解释架构时画的那个东西。

- **LangGraph** → 你画一张图。节点是步骤，边是跳转，每个位置上的状态对象都是有类型的。心智模型是状态机。
- **CrewAI** → 你画一张组织架构图。每个角色有岗位描述，由经理分派任务。心智模型是一支小型专家团队。
- **AutoGen** → 你画一个 Slack 私聊。两个 Agent 互相发消息；需要主持人时再加第三个。心智模型是聊天。
- **Agno** → 你画一个单独的盒子，旁边挂着工具。把几个盒子并排放就是团队。心智模型是「自带电池的 Agent」。

### 状态问题

状态，是大多数框架选型在生产中崩掉的地方。

- **LangGraph。** 类型化状态（`TypedDict` 或 Pydantic 模型）、按字段的 reducer、一等 checkpoint 存储器（SQLite/Postgres/Redis）。恢复、中断和时间旅行都是免费的。*（见第 11 阶段 · 16。）*
- **CrewAI。** 状态通过 `context` 字段以字符串形式在任务之间流动，或者通过 `output_pydantic` 结构化为对象。开箱没有按 crew 的持久存储；如果 crew 必须扛过重启，得自己接一个。
- **AutoGen。** 状态就是聊天历史和用户自定义的 `context`。对话记录可以持久化；任意工作流状态则不行，除非你自己写适配器。
- **Agno。** 内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB），通过 `storage=` 挂到 `Agent` 上——对话会话和用户记忆自动持久化。它不是完整的图 checkpoint 存储器，而是会话存储。

### 分支问题

任何不简单的 Agent 都会分支。关键是谁来做分支决定。

- **LangGraph**——由你决定，通过条件边。路由是一个带命名分支的 Python 函数。分支在编译后的图中是一等公民；Checkpoint 存储器会记录走了哪个分支。
- **CrewAI**——层级模式下由经理决定；顺序模式下你在构建时决定。路由隐含在任务列表中；在经理的 prompt 之外，没有一等的「if」。
- **AutoGen**——由 Agent 通过聊天决定。分支从「下一个谁说话」中自然涌现。`GroupChatManager` 选择下一位发言者；你可以手写 `speaker_selection_method`，但默认由 LLM 驱动。
- **Agno**——Agent 通过下一步调用哪个工具来决定。团队有 coordinator/router/collaborator 模式；再复杂的分支就是开发者的责任了。

### 可观测性问题

- **LangGraph**——通过 LangSmith 或任意 OTel exporter 接入 OpenTelemetry。每次节点跳转都是一个 trace span；Checkpoint 同时也是可回放的轨迹。LangSmith 是官方选项；Langfuse/Phoenix 也有适配器。
- **CrewAI**——自 2025 年末起支持一等 OpenTelemetry；可与 Langfuse、Phoenix、Opik、AgentOps 集成。
- **AutoGen**——通过 `autogen-core` 集成 OpenTelemetry；AgentOps 和 Opik 提供了连接器。追踪粒度按 Agent 消息划分，而不是按节点。
- **Agno**——内置 `monitoring=True` 开关，外加 OpenTelemetry exporter；与 Langfuse 深度集成，可做会话追踪。

### 成本与延迟

四个框架都会增加每次调用的开销（框架逻辑、校验、序列化）。开销从低到高的大致顺序是：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要来自框架额外做了多少 LLM 路由。CrewAI 的层级经理要花 token 决定下一个谁上；AutoGen 的 `GroupChatManager` 也一样。LangGraph 只在你写 `llm.invoke` 的地方花 token。Agno 的单 Agent 路径很薄。

当每次运行的成本重要时，优先用显式路由（LangGraph 的边、AutoGen 的 `speaker_selection_method`），而不是 LLM 选择的路由。

### 互操作性

- **LangGraph** ↔ **LangChain** 的工具、检索器、LLM。一等 MCP 适配器（把工具作为 MCP server 导入）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain 工具、LlamaIndex 工具和 MCP 工具都能适配进来。通过 `allow_delegation=True` 实现 crew 之间的委派。
- **AutoGen** → `FunctionTool` 可以包装任意 Python 可调用对象；提供 MCP 适配器。Agent 对 Agent 的模式与 AG2 生态耦合较紧。
- **Agno** → 使用 `@tool` 装饰器或 BaseTool 子类；提供 MCP 适配器；工具可以在 Agent 和团队之间共享。

## 技能

> 你能用一句话解释，为什么某个框架适合某个 Agent 问题。

动手前的检查清单：

1. **画出形状。** 这是一张图（类型化状态、命名跳转）？一场角色扮演（专家交接工作）？一段聊天（Agent 聊到结束）？还是一个带工具的单 Agent？
2. **决定谁来分支。** 开发者决定分支 → LangGraph。经理 Agent 决定 → CrewAI 层级模式。聊天中涌现 → AutoGen。Tool call 决定 → Agno。
3. **检查状态需求。** 你需要从 checkpoint 恢复吗？需要时间旅行吗？需要运行中的人工中断吗？如果是，默认选 LangGraph；Agno 的会话可以覆盖对话范围内的状态。
4. **检查成本预算。** LLM 选择路由每轮都要多花 token。如果 Agent 一天运行数千次，优先选显式路由。
5. **把框架开销计入预算。** 每个框架都是一项额外依赖。如果任务只是两次 LLM 调用加一个工具，写 30 行纯 Python；没有框架比没有框架更便宜。

在你能够画出那张图、那张组织架构图、那段聊天或那个 Agent 盒子之前，拒绝伸手去拿框架。拒绝选择那种逼你为真正需要的功能去跟它的状态模型对抗的框架。

## 决策矩阵

| 问题形状 | 首选框架 | 原因 |
|---------------|---------------------|-----|
| 带类型化状态、人工审批、长时间运行的工作流 DAG | LangGraph | 一等状态、checkpoint 存储器、中断、时间旅行。 |
| 角色分明的研究、写作 pipeline | CrewAI（顺序）或 LangGraph 子图 | 在 CrewAI 中按任务分角色很省事；分支变复杂时用 LangGraph 扩展。 |
| 提议者-批评者或师生对话 | AutoGen | 双 Agent 聊天就是它的原生形态。 |
| 带工具、会话、记忆的单 Agent | Agno | 搭建最薄，内置存储和记忆。 |
| 带 reducer 的数千路并行扇出 | LangGraph + `Send` | 唯一提供一等并行派发 API 的框架。 |
| 快速原型、不绑定框架 | 纯 Python + 提供商 SDK | 没有框架就是最快的框架。 |

```figure
l5-framework-fit
```

## 练习

1. **简单。** 拿同一个任务——「调研 Anthropic 的总部，写一份 200 词的简报，附上来源」——分别用 LangGraph（四个节点：计划、搜索、写作、引用）和 CrewAI（三个角色：研究员、写手、编辑）实现。报告每次运行的 token 成本和代码行数。
2. **中等。** 用 AutoGen（研究员 ↔ 写手聊天，编辑通过 `GroupChat` 加入）和 Agno（带 `search_tools` 与 `write_tools` 的单 Agent，外加会话存储）实现同一任务。按以下三项给四个实现排序：(a) 每次运行成本，(b) 崩溃后恢复能力，(c) 在写作步骤前注入人工审批的能力。
3. **困难。** 写一个决策树脚本 `pick_framework.py`，接收一段简短的问题描述（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`），返回一个推荐框架和一句话理由。用你自己设计的六个案例来验证它。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Orchestration（编排） | 「Agent 之间如何协同」 | 决定下一个运行哪个节点、角色或 Agent 的那一层。 |
| Durable state（持久状态） | 「重启后能恢复」 | 能扛过进程死亡的状态，挂在 checkpoint 或会话存储上。 |
| LLM-selected routing（LLM 选择路由） | 「让模型来定」 | 由规划 LLM 每轮挑选下一步；灵活，但每个决策都要花 token。 |
| Explicit routing（显式路由） | 「开发者来定」 | 由 Python 函数或静态边选择下一步；成本低且可审计。 |
| Crew（团队） | 「一个 CrewAI 团队」 | 角色 + 任务 + 流程（顺序或层级），绑定成一个可运行对象。 |
| GroupChat | 「AutoGen 的多 Agent 聊天」 | 在 N 个 Agent 之间受管理的对话，配有发言者选择器。 |
| Team（Agno） | 「多 Agent 版 Agno」 | 在一组 Agent 之上的 route / coordinate / collaborate 模式。 |
| StateGraph | 「LangGraph 的图」 | 类型化状态、节点、条件边、checkpoint 存储器的抽象。 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)——StateGraph、checkpoint 存储器、中断、时间旅行。
- [CrewAI 文档](https://docs.crewai.com/)——Crews、Flows、Agents、Tasks、Processes。
- [AutoGen 文档](https://microsoft.github.io/autogen/)——ConversableAgent、GroupChat、团队、工具。
- [Agno 文档](https://docs.agno.com/)——Agent、Team、Workflow、存储、记忆。
- [Anthropic——Building effective agents（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents)——与框架无关的模式库（prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer）。
- [Yao 等，《ReAct: Synergizing Reasoning and Acting》（ICLR 2023）](https://arxiv.org/abs/2210.03629)——每个框架都在包装的那套循环。
- [Wu 等，《AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation》（2023）](https://arxiv.org/abs/2308.08155)——AutoGen 的设计论文。
- [Park 等，《Generative Agents: Interactive Simulacra of Human Behavior》（UIST 2023）](https://arxiv.org/abs/2304.03442)——CrewAI 式人设栈所依托的角色扮演基础。
- 第 11 阶段 · 16（LangGraph）——本课用来做对照的框架。
- 第 11 阶段 · 19（Reflexion）——一个能干净映射到 LangGraph、但放到 CrewAI 里很别扭的模式。
- 第 11 阶段 · 22（生产可观测性）——无论选哪个框架，该如何埋点。
