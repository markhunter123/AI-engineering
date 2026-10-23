# 编排模式：监督者、蜂群、层级式

> 四种编排模式在 2026 年的框架中反复出现：监督者-工作者、蜂群 / 点对点、层级式、辩论。Anthropic 的指导：「关键在于为你的需求构建正确的系统。」从简单开始；只有当单个智能体加五种工作流模式不够用时，才增加拓扑。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 25 (Multi-Agent Debate)
**Time:** ~60 minutes

## 学习目标

- 说出四种反复出现的编排模式，以及各自适用的场景。
- 描述 2026 年 LangChain 的建议：基于工具调用的监督 vs 监督者库。
- 解释 Anthropic 的「构建正确的系统」原则，以及它如何约束拓扑选择。
- 在同一个脚本化 LLM 上，用 stdlib 实现全部四种模式。

## 问题

团队在真正需要之前就伸手去够「多智能体」。四种模式在框架中反复出现；一旦你能叫出它们的名字，就能选对模式——或者完全跳过拓扑。

## 概念

### 监督者-工作者（Supervisor-worker）

- 一个中心路由 LLM 把任务分派给专家智能体。
- 决策：回到自身循环、移交给专家、终止。
- 专家之间不互相交谈；所有路由都经过监督者。

框架：LangGraph `create_supervisor`、Anthropic orchestrator-workers、CrewAI Hierarchical Process。

**2026 年 LangChain 建议：** 通过直接工具调用来做监督，而不是 `create_supervisor`。这样能更精细地控制上下文工程——你决定每位专家确切看到什么。

### 蜂群 / 点对点（Swarm / peer-to-peer）

- 智能体通过共享工具面直接移交。
- 没有中心路由器。
- 延迟低于监督者（跳数更少）。
- 更难推理（没有单一控制点）。

框架：LangGraph swarm 拓扑、OpenAI Agents SDK handoffs（当所有智能体都能移交给其他所有智能体时）。

### 层级式（Hierarchical）

- 监督者管理子监督者，子监督者再管理工作者的结构。
- 在 LangGraph 中实现为嵌套子图；在 CrewAI 中为嵌套 crew。
- 能扩展到大规模智能体群体，代价是运维复杂度。

何时需要：当单个监督者的上下文预算无法容纳所有专家的描述时。

### 辩论

- 并行提议者 + 迭代交叉批评（第 25 课）。
- 严格来说不是编排——更接近验证——但在框架中作为一种拓扑选项出现。

### 自主 crew vs 确定性 flow

CrewAI 形式化了两类部署模式：

- **Flow** 用于确定性的事件驱动自动化（生产环境的推荐起点）。
- **Crew** 用于自主的、基于角色的协作。

这与上述四种模式正交，但可映射到拓扑：Flow 通常是监督者或层级式；Crew 通常是带 LLM 路由器的监督者。

### Anthropic 的指导

「在 LLM 领域的成功不在于构建最复杂的系统，而在于为你的需求构建正确的系统。」

决策顺序：

1. 单智能体 + 工作流模式（第 12 课）——从这里开始。
2. 监督者-工作者——当你有 2-4 位专家时。
3. 蜂群——当延迟比推理清晰度更重要时。
4. 层级式——只有当监督者的上下文预算不够时。
5. 辩论——当准确率比成本更重要时。

### 这个模式容易出错的地方

- **拓扑先行思维。** 在明确多智能体要解决什么问题之前就说「我们需要多智能体」。
- **蜂群中的来回弹跳。** A -> B -> A -> B。使用跳数计数器。
- **虚假层级。** 因为「企业级」而设三层；实际只有两个团队。合并掉。

```figure
orchestration-pattern
```

## 构建它

`code/main.py` 用 stdlib 针对一个脚本化 LLM 实现全部四种模式：

- `Supervisor`——中心路由器。
- `Swarm`——带直接移交的点对点网络。
- `Hierarchical`——监督者之上的监督者。
- `Debate`——并行提议 + 批评。

每种模式处理同一个三意图任务（退款 / 缺陷 / 销售）。轨迹形状不同。

运行它：

```
python3 code/main.py
```

输出：各模式的轨迹 + 操作计数。监督者最清晰；蜂群最短；层级式最深；辩论最昂贵。

## 使用它

- **LangGraph** 用于监督者与层级式（嵌套子图）。
- **OpenAI Agents SDK** 用于 handoffs-as-tools（监督者形态）。
- **CrewAI Flow** 用于生产级确定性流程。
- **自定义实现** 用于辩论，或当你需要精确控制时。

## 交付它

`outputs/skill-orchestration-picker.md` 选择一个拓扑并实现它。

## 练习

1. 移除路由器，把监督者-工作者转换为蜂群。什么会坏？什么会变好？
2. 给蜂群添加跳数计数器：3 次移交后拒绝。它能抓住 A->B->A 的弹跳吗？
3. 为一个 12 位专家的领域构建两级层级式系统。不做嵌套时，上下文预算在哪里失效？
4. 在生产形态的工作负载上对四种模式做性能剖析。哪种模式在哪个指标上胜出（延迟、成本、准确率、可调试性）？
5. 阅读 Anthropic 的「Building Effective Agents」文章。把你的每条生产流程映射到四种模式之一。有无法干净映射的吗？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 监督者-工作者（Supervisor-worker） | 「路由器 + 专家」 | 中心 LLM 分派给专家；专家之间互不交谈 |
| 蜂群（Swarm） | 「点对点」 | 通过共享工具直接移交；无中心路由器 |
| 层级式（Hierarchical） | 「监督者的监督者」 | 面向大规模群体的嵌套子图 |
| 辩论（Debate） | 「提议 + 批评」 | 并行提议者，交叉批评（第 25 课） |
| 基于工具调用的监督（Tool-call-based supervision） | 「不用库的监督者」 | 把监督者实现为直接工具调用，以便控制上下文 |
| Crew | 「自主团队」 | CrewAI 基于角色的协作模式 |
| Flow | 「确定性工作流」 | CrewAI 的事件驱动生产模式 |

## 进一步阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) —— 五种模式 + 智能体 vs 工作流
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 监督者、蜂群、层级式
- [CrewAI docs](https://docs.crewai.com/en/introduction) —— Crew vs Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) —— 辩论模式
