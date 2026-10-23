# 基于角色的智能体团队——角色、任务与流程

> 四个原语：Agent、Task、Crew、Process。两种顶层形态：Crews（自主的、基于角色的协作）与 Flows（事件驱动、确定性）。CrewAI 是 2026 年的参考实现，它的文档说得很直白：「任何生产级应用都应从 Flow 开始。」

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75 minutes

## 学习目标

- 说出 CrewAI 的四个原语（Agent、Task、Crew、Process），以及各自负责什么。
- 区分 Sequential、Hierarchical 和计划中的 Consensus 流程；为每种工作负载选择一种。
- 区分 Crews（自主的、基于角色）与 Flows（事件驱动、确定性），并解释文档给出的生产建议。
- 用 `@tool` 装饰器和 `BaseTool` 子类接入工具；思考结构化输出与自由文本的取舍。
- 说出 CrewAI 的四种记忆类型，以及各自何时值得使用。
- 用标准库实现一个三智能体 crew（researcher、writer、editor），产出一份简报。
- 识别 CrewAI 的三种失败模式：prompt 膨胀、manager LLM 开销、脆弱的交接。

## 问题

采用多智能体框架的团队都会撞上同一堵墙。「自主协作」在演示里听起来很好。然后客户报了一个 bug，你需要确定性回放。或者财务问，一个由 LLM 路由的 crew 每次运行要花多少钱。或者 on-call 需要知道凌晨 3 点哪个智能体卡住了。

自由形式的 LLM 路由 crew 无法干净地回答这些问题。纯 DAG 能全部回答，但会失去头脑风暴型智能体所需的探索性形态。

CrewAI 的拆分坦诚面对这种权衡。Crews 用于协作式、基于角色、探索性的工作。Flows 用于事件驱动、代码掌控、可审计的生产环境。同一个框架，两种形态，按场景选择。

## 概念

### 四个原语

CrewAI 的接口面很小。记住这些，剩下的都是配置。

- **Agent。** `role + goal + backstory + tools + (optional) llm`。backstory 不可或缺。它塑造语气、判断力，以及智能体何时停止。工具是智能体可以调用的函数（详见下文）。
- **Task。** `description + expected_output + agent + (optional) context + (optional) output_pydantic`。可复用的工作单元。`expected_output` 是契约。`context` 列出一个上游任务，其输出会传入。`output_pydantic` 强制产出结构化形状。
- **Crew。** 容器。持有 `agents` 列表、`tasks` 列表、`process`，以及可选的 `memory` + `verbose` + `manager_llm` 设置。
- **Process。** 执行策略。Sequential、Hierarchical、Consensus（计划中）。决定运行的形态。

智能体之间不直接看到彼此。Task 引用 Agent。Crew 给 Task 排序。Process 决定由谁挑选下一个 Task。这就是完整的心智模型。

> **已对照** CrewAI 0.86（2026-05）验证。较新版本可能重命名或合并 process 类型；在依赖具体形态之前，请查阅 [CrewAI Processes 文档](https://docs.crewai.com/concepts/processes)。

### Sequential、Hierarchical 与 Consensus

- **Sequential。** Task 按声明顺序运行。第 N 个任务的输出作为 `context` 提供给第 N+1 个任务。成本最低。最可预测。适用于顺序固定的场景。
- **Hierarchical。** 一个 manager Agent（独立的 LLM 调用）在专家之间路由。CrewAI 会根据你的 `manager_llm` 配置或默认值创建 manager。manager 在每一轮挑选下一个任务，可以拒绝或重新路由。适用于你有至少四个专家、且顺序确实取决于先前输出的场景。
- **Consensus。** 计划中，公共 API 尚未实现。文档保留了这个名字，用于未来基于投票的流程。目前不要依赖它。

Hierarchical 在每次专家调用之上增加一次按轮的 LLM 调用（manager）。在五步运行中，token 成本可能增至原来的三倍。只有当你需要路由时才为之付费。

### Crews 与 Flows

这是文档在 2026 年开篇给出的划分框架。

- **Crew。** LLM 驱动的自主性。框架在运行时决定形态。适合：研究、头脑风暴、初稿，以及路径本身就是答案一部分的任何工作。难以回放。难以测试。原型成本低。
- **Flow。** 你拥有的事件驱动图。`@start` 标记入口。`@listen(topic)` 标记当另一个步骤发出该 topic 时触发的步骤。每个步骤都是普通 Python（内部可以调用 Crew）。适合：生产。可观测。可测试。确定性。

文档 2026 年的生产建议：从 Flow 开始。当自主性值得其成本时，在 Flow 步骤内部通过 `Crew.kickoff()` 调用把 Crew 嵌进来。Flow 给你审计轨迹，Crew 给你探索能力。组合使用，而不是二选一。

### 工具集成

给 Agent 提供工具有三种方式。选择满足需求的最简单一种。

1. **`@tool` 装饰器。** 纯函数变成工具。签名就是 schema；docstring 是 LLM 看到的描述。最适合一次性辅助函数。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` 子类。** 基于类的工具，带有显式的 args schema、异步支持、重试。适用于工具有状态（客户端、缓存）或需要结构化参数的场景。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **内置工具包。** CrewAI 提供第一方适配器：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一次导入即可接入。

结构化输出使用 Pydantic。在 Task 上传入 `output_pydantic=MyModel`。CrewAI 会用模型校验 LLM 的响应，要么强制转换，要么重试。再配上一段紧凑的 `expected_output` 字符串。自由文本输出对草稿没问题；结构化输出才是下游 Flow 能消费的东西。

### 记忆钩子

CrewAI 开箱提供四种记忆类型。它们可以组合：一个 Crew 可以同时启用全部四种。

> **已对照** CrewAI 0.86（2026-05）验证。近期版本把所有内容都路由到一个统一的 `Memory` 系统，包裹这四种存储。下面的概念模型仍然成立，但在更新版本中公共类表面可能收敛为单一的 `Memory` 入口；当前 API 请查阅 [CrewAI 记忆文档](https://docs.crewai.com/concepts/memory)。

- **短期记忆。** 单次运行内的对话缓冲区。运行结束时清空。
- **长期记忆。** 跨运行持久化。存储在向量数据库中（默认 Chroma，可替换）。按与当前任务的相似度检索。
- **实体记忆。** 按实体记录事实。「客户 X 使用的是企业版套餐。」以实体为键，而非相似度。跨运行保留。
- **上下文记忆。** 组装时检索。在 Agent 需要的那一刻拉取相关记忆，而不是预加载。

在 Crew 上用 `memory=True` 或按类型配置启用。由你配置的 embeddings 提供方支撑（默认 OpenAI，可替换为本地）。记忆是 CrewAI 相对更轻框架的价值所在之一；纯 LangGraph 需要你自己接好其中的每一项。

### 基于角色的团队适合什么时候

- 三到六个有命名角色和协作流程的智能体。起草、审查、规划、头脑风暴。
- 路由决策中，LLM 对下一步的判断本身就是价值（Hierarchical）。
- 团队更愿意读 `role + goal + backstory` 而不是读图定义的任何场景。

### 不适合的时候

- 严格顺序的确定性 DAG。使用 LangGraph（第 13 课）。图形态才是正确的抽象；CrewAI 的角色框架反而是摩擦。
- 亚秒级延迟预算。Hierarchical 增加往返。即便是 Sequential，也会逐个发送包含 backstory 和先前输出的 prompt。
- 单智能体循环。跳过框架；一个智能体循环（第 1 课）加一个工具注册表更短。

第 17 课（智能体框架取舍）用矩阵说明了这一点。简版：CrewAI 位于「协作式、基于角色」的角落。

### 依赖形态

独立于 LangChain。Python 3.10 到 3.13。使用 `uv`。星标数：见 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（2026-05 快照）。AWS Bedrock 集成有文档；厂商基准报告在 QA 工作负载上相对 LangGraph 有大幅加速，但方法学（数据集、硬件、评估指标）未公开，因此框架厂商的数字只能作为方向性参考。

### 这个模式会在哪里出错

- **backstory 导致的 prompt 膨胀。** 每个智能体 2000 字的 backstory，五个智能体的 crew 在第一次工具调用之前就烧掉了上下文预算。把 backstory 控制在 200 字以内。在智能体之间复用措辞；不要把公司文风重复五遍。
- **manager LLM 的 token 税。** Hierarchical 流程在每次专家调用前增加一次 manager LLM 调用。在五任务的 crew 上，就是六次 LLM 调用而不是五次，而且 manager 调用携带完整任务列表加先前输出。除非路由依赖输出，否则改用 Sequential。
- **脆弱的交接。** 第 N 个任务的 `expected_output` 是「一份大纲」。第 N+1 个任务把它作为 `context` 读取，试图解析三个小节。LLM 却产出了四个。下游 Agent 开始即兴发挥。解法是给第 N 个任务加 `output_pydantic`，让第 N+1 个任务读取类型化对象，而不是自由文本。
- **把 Crew 当生产。** 没有 Flow 包装就把自由形式的 Crew 上线。输出波动大；无法回放；on-call 无法把一次坏运行和好运行做 diff。用 Flow 包装。

```figure
ae-crew-vs-flow
```

## 构建它

`code/main.py` 用标准库实现两种形态，外加一个三智能体 crew。

形态：

- `Agent`、`Task` dataclass，对应 CrewAI 的表面。
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行任务，把输出作为 `context` 传递。
- `HierarchicalCrew.kickoff(topic)` 增加一个 manager Agent，每轮挑选下一个专家，遇到 「done」 停止。
- `Flow`，带 `@start` 和 `@listen(topic)` 装饰器、一个微型事件循环和一条 trace。
- `tool(name)` 装饰器，对应 CrewAI 的 `@tool` 形态。
- `Memory`，包含 `short_term`、`long_term`、`entity` 存储；模拟相似度使用 numpy。
- Mock LLM 响应是硬编码字符串，按 role 加输入前缀键控。无网络。确定性。

具体演示：researcher、writer、editor 组成的 crew 产出关于 「agent engineering 2026」 的简报。Researcher 拉取（模拟的）来源。Writer 起草。Editor 收紧。同一个 crew 再跑一遍 Flow，展示确定性形态。

运行它：

```bash
python3 code/main.py
```

Trace 覆盖：Sequential crew 通过 `context` 传递输出；Hierarchical crew 由 manager 挑选（researcher、writer、editor，然后 「done」）；flow 用显式 topic（`researched`、`drafted`、`edited`）运行同样的三个步骤；通过 `@tool` 路由的工具调用；以及跨两次 kickoff 存活的长期记忆。

Crew 的 trace 是流动的；manager 原则上可以重新排序。Flow 的 trace 是固定的。这个选择本身就是这节课的内容。

## 使用它

- **CrewAI Flow** 用于生产。即便 Flow 只是一个调用 `Crew.kickoff()` 的步骤。Flow 提供审计边界。
- **CrewAI Crew（Sequential）** 用于顺序清晰的协作工作，尤其是初稿和审查循环。
- **CrewAI Crew（Hierarchical）** 当路由取决于输出，且你有至少四个专家时。
- **LangGraph**（第 13 课）用于显式状态机、持久恢复、严格顺序。
- **AutoGen v0.4**（第 14 课）用于 actor 模型的并发和故障隔离。
- **OpenAI Agents SDK**（第 16 课）用于以 OpenAI 为先、带交接和防护栏的产品。
- **Claude Agent SDK**（第 17 课）用于以 Claude 为先、带子智能体和 session 存储的产品。

## 交付它

`outputs/skill-crew-or-flow.md` 为任务选择 Crew 或 Flow，并搭建最小实现。硬性拒绝：没有 backstory 的 Crew、没有显式 topic 的 Flow、少于三个专家的 Hierarchical。

## 陷阱

- **把 backstory 当调料。** 它会塑造输出。每个智能体测试三个变体；方差是真实的。选定一个，冻结它。
- **跳过 `expected_output`。** 每个任务没有契约，下游任务就会捡到 LLM 产出的任何东西。Crew 跑完了；审计失败了。
- **记忆总是开着。** 长期记忆每次运行都写入。向量数据库不断增长。检索越来越嘈杂。把写入范围限制在事实确实持久存在的任务上。
- **manager prompt 漂移。** Hierarchical 的 manager prompt 是隐式的。如果路由变得奇怪，在 verbose 模式下 dump 出来读一读。
- **Crew 中的工具副作用。** Crew 调用工具的次数可能超出预期。POST、DELETE、支付应放在 Flow 步骤里，绝不能放进 Crew 工具。

## 练习

1. 把 Sequential crew 改造成 Flow。数一数波动性下降的接触点，并记下可读性下降的地方。
2. 给 crew 增加实体记忆：客户相关的事实跨 kickoff 保留。验证检索能拉到正确的实体。
3. 实现一个 Hierarchical 流程，manager 在 writer 的输出至少有三段之前拒绝路由给 editor。追踪重试过程。
4. 为一个（模拟的）网页搜索接入 `BaseTool` 子类。比较 trace 形状与 `@tool` 装饰器版本。
5. 给 editor 任务加 `output_pydantic=Brief`，其中 `Brief` 包含 `title`、`summary`、`sections`。让 writer 任务输出一次格式错误的 JSON；在 trace 中验证 CrewAI 的重试行为。
6. 阅读 CrewAI 文档的简介。把玩具移植到真正的 `crewai` API。标准库版本跳过了哪些保证？
7. 把 AgentOps 或 Langfuse（第 24 课）接到一次真实运行上。标准库版本中你漏掉了哪些 trace？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Agent | 「人设」 | 角色 + 目标 + backstory + 工具 |
| Task | 「工作单元」 | 描述 + 预期输出 + 执行者 + 可选的结构化输出 |
| Crew | 「智能体团队」 | Agent + Task + Process 的容器 |
| Process | 「执行策略」 | Sequential / Hierarchical / Consensus（计划中） |
| Flow | 「确定性工作流」 | 事件驱动、代码掌控、可测试 |
| Backstory | 「人设 prompt」 | 塑造 Agent 语气与判断力的背景 |
| `@tool` | 「函数工具」 | 把函数变成 Agent 可调用工具的装饰器 |
| `BaseTool` | 「类工具」 | 带 args schema、重试、异步支持的基于类的工具 |
| 实体记忆 | 「按实体的事实」 | 作用域限定在客户 / 账户 / 问题上的记忆 |
| 长期记忆 | 「跨运行记忆」 | 由向量支撑、在 kickoff 之间存活的记忆 |
| 上下文记忆 | 「即时检索」 | 在 Agent 需要的那一刻拉取的记忆 |
| Manager LLM | 「路由智能体」 | Hierarchical 流程中挑选下一个任务的额外 LLM |
| `expected_output` | 「任务契约」 | 告诉 Agent（以及审计）应返回何种形状的字符串 |

## 延伸阅读

- [CrewAI 文档简介](https://docs.crewai.com/en/introduction)：概念与推荐的生产路径
- [CrewAI Flows 指南](https://docs.crewai.com/en/concepts/flows)：事件驱动形态、`@start`、`@listen`
- [CrewAI 工具参考](https://docs.crewai.com/en/concepts/tools)：`@tool`、`BaseTool`、内置工具包
- [CrewAI 记忆](https://docs.crewai.com/en/concepts/memory)：短期、长期、实体、上下文
- [Anthropic，Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：多智能体何时有帮助、何时没有
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview)：状态机替代方案
