# 监督者 / 编排器-工作者模式

> 一个主导 Agent 负责规划与委派；专业化的工作者在各自的并行上下文中执行，然后回报结果。这就是 Anthropic 的 Research 系统背后的模式（Claude Opus 4 作为主导，Sonnet 4 作为子 Agent），在内部研究评估中比单 Agent Opus 4 高出 90.2%。Anthropic 的工程文章指出，BrowseComp 上 80% 的方差仅用 token 用量就能解释——多 Agent 之所以胜出，很大程度上是因为每个子 Agent 都获得了一个全新的上下文窗口。本课从原语出发构建 supervisor 模式，并梳理 2026 年生产部署中的工程经验。

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## 问题

研究是单 Agent 系统会失败的典型任务。你问「2023 到 2026 年之间多 Agent 系统发生了什么变化？」单个 Agent 按顺序读五篇论文，用它们的正文填满一半上下文，然后必须把它们放在一起推理。读到第五篇时，它已经忘了第一篇。它无法并行。

监督者（supervisor）模式解决了这个问题：一个主导 Agent 规划搜索，把每个子问题委派给一个工作者，最后进行综合。每个工作者都有一个属于自己的 200k token 窗口，只处理一个狭窄的问题。主导 Agent 从不接触原始论文——只看工作者的摘要。

Anthropic 的生产 Research 系统在内部研究评估中，相比单个 Opus 4 报告了 +90.2% 的提升。同一篇文章还指出，BrowseComp 上 80% 的方差*仅用 token 用量*就能解释。每个子 Agent 拥有全新上下文是主要机制。

## 概念

### 模式

```
                 ┌──────────────┐
                 │   Lead       │  plans, decomposes,
                 │  (Opus 4)    │  synthesizes
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ Worker1 │  │ Worker2 │  │ Worker3 │
      │(Sonnet) │  │(Sonnet) │  │(Sonnet) │
      └─────────┘  └─────────┘  └─────────┘
         fresh       fresh        fresh
         context     context      context
```

主导 Agent 从不阅读原始材料。工作者在主导 Agent 综合之前也互不可见。每个箭头都是一次携带狭窄 artifact 的交接。

### 为什么它有效

三个机制：

1. **每个子 Agent 拥有全新上下文。** 一个探索「FIPA-ACL 渊源」的工作者，不会背上前者花在规划上的 40k token。它得到一个 200k 窗口，只为一个问题服务。
2. **通过 prompt 实现专业化。** 主导 Agent 的 prompt 是「分解并综合」，不是「做研究」。每个工作者的 prompt 都很窄：「找出 X 中发生了什么变化」。聚焦的 prompt 产生聚焦的输出。
3. **并行。** 工作者并发运行。实际耗时大致是 `max(worker_times) + plan + synthesis`，而不是 `sum(worker_times)`。

### 工程经验（Anthropic 2025）

Anthropic 的文章列出的几条生产经验到 2026 年仍然适用：

- **按查询复杂度调整投入规模。** 简单查询：一个 Agent，3-10 次 tool call。复杂查询：10 个以上 Agent。这必须由主导 Agent 自己估算，而不是由调用方指定。
- **先宽后窄。** 先分解成宽泛的子问题，如果答案值得深入，再为每个子问题派生更多工作者。
- **彩虹部署（rainbow deployment）。** Agent 是长时运行且有状态的。传统的蓝绿部署行不通。Anthropic 使用彩虹部署：新版本逐步铺开，同时旧版本继续排空流量。
- **token 用量占主导。** 多 Agent 消耗的 token 约为单 Agent 的 15 倍。只有任务价值足以覆盖成本时才使用它。

### 转向图原生

LangGraph 最初随附一个 `langgraph-supervisor` 库，提供高层的 `create_supervisor` 辅助函数。2025 年，LangChain 把推荐做法转向直接用 tool calling 实现 supervisor 模式，因为 tool call 能更好地控制*supervisor 看得见什么*（上下文工程）。该库仍然可用；文档现在推荐 tool-calling 形式。

### 失败模式

- **主导 Agent 幻觉出计划。** 如果主导 Agent 生成的子问题并没有真正分解原问题，工作者就会对错误的目标做精确的研究。
- **工作者过度探索。** 没有明确的范围边界时，工作者会漂移到指定子问题之外，污染综合阶段。
- **综合时冲突。** 两个工作者返回互相矛盾的事实。主导 Agent 必须要么重新询问（增加一轮），要么明确指出分歧。静默地选择一方是最糟糕的失败：用户永远不会知道曾经出现过分歧。

### 什么时候不该用 supervisor

- **顺序任务。** 如果第 2 步确实需要第 1 步的输出，并行毫无收益。改用 pipeline（CrewAI Sequential、LangGraph 线性图）。
- **简单查询。** 单 Agent 处理得更快更便宜。在派生工作者之前，先执行主导 Agent 的「按复杂度调整投入」检查。
- **严格确定性。** supervisor 使用 LLM 选择的委派方式。当审计/回放比适应性更重要时，静态图更合适。

```figure
supervisor-hierarchy
```

## 构建它

`code/main.py` 使用 `threading` 实现一个由三个并行工作者组成的 supervisor。主导 Agent 把查询分解成子问题，工作者针对每个子问题并发运行，最后由主导 Agent 综合。没有真实 LLM——工作者是脚本化的，用来模拟抓取并总结。

关键结构：

- `Lead.plan(query)` 把查询拆成 3 个子问题。
- `Worker.run(sub_q)` 返回一个假的摘要（在生产中可以是任何使用工具的 Agent）。
- `Lead.run(query)` 在线程中启动工作者，join 后做综合。

运行：

```
python3 code/main.py
```

输出会展示计划、带起止时间戳的并行工作者轨迹，以及最终综合。你能看到实际的耗时收益：三个耗时 0.3 秒的工作者，总共约 0.35 秒跑完，而不是 0.9 秒。

## 使用它

`outputs/skill-supervisor-designer.md` 接收用户查询并产出 supervisor 模式设计：主导 Agent 的 system prompt、工作者角色、子问题分解规则，以及综合模板。在构建新的研究型 Agent 系统之前使用它。

## 交付它

部署 supervisor 模式前的检查清单：

- **模型搭配。** 主导 Agent 用推理级模型（Opus 级、`o3` 级）。工作者用更快、更便宜的模型（Sonnet、`o4-mini`）。
- **工作者超时。** 任何超过中位运行时间 2 倍的工作者都会被终止；主导 Agent 要么用更窄的范围重新派生它，要么不带它继续。
- **每个工作者的 token 上限。** 硬性限制（比如预期综合输入的 10 倍）防止失控的工作者耗尽预算。
- **可观测性。** 追踪主导 Agent 的计划、每个工作者的 tool call，以及综合过程。这是任何事后调试的基础。
- **彩虹式发布。** 有状态的长时运行 Agent 需要渐进的版本过渡，而不是热切换。

## 练习

1. 运行 `code/main.py`，然后修改主导 Agent，让它派生 5 个工作者而不是 3 个。观察实际耗时变化。在这个演示中，工作者数量达到多少时，派生开销会超过并行节省的时间？
2. 实现工作者超时：杀掉任何运行超过 0.5 秒的工作者，让主导 Agent 用剩余结果做综合。你需要什么可观测性才能知道某个工作者被裁掉了？
3. 在主导 Agent 的综合步骤中加入冲突检测：如果两个工作者返回互相矛盾的答案，主导 Agent 应记录分歧，而不是选择其一。如何在不调用 LLM 的情况下检测矛盾？
4. 阅读 Anthropic 的 Research 系统工程文章。列出这个玩具演示要上生产还需要采用的三项实践。
5. 对比 LangGraph 的 `create_supervisor`（旧版）与新的 tool-calling 推荐做法。哪一种能让你更好地控制 supervisor 看得见的内容？为什么 Anthropic 明确只把子答案、而不是工作者的原始上下文传入综合步骤？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Supervisor | 「主导 Agent」 | 负责规划、委派和综合的编排器 Agent。它自己不做实际工作。 |
| Worker | 「子 Agent」 | 由 supervisor 调用的聚焦 Agent，范围狭窄，拥有自己的上下文窗口。 |
| Orchestrator-worker | 「supervisor 模式」 | 同一件事的不同叫法。2026 年的文献两种都在用。 |
| Fresh context | 「干净窗口」 | 工作者的上下文从它的 system prompt 和分配的问题开始，而不是主导 Agent 的历史。 |
| Rainbow deployment | 「渐进式发布」 | 长时运行的有状态 Agent 需要带版本管理的排空与替换，而不是蓝绿部署。 |
| Token dominance | 「上下文才是变量」 | 据 Anthropic，研究评估中 80% 的方差来自总 token 用量，而非模型选择。 |
| Scale effort | 「按复杂度匹配 Agent 数量」 | 主导 Agent 估算查询难度，据此派生 1 个或 10 个以上工作者。 |
| Synthesis conflict | 「工作者意见不一致」 | 两个工作者返回矛盾事实；主导 Agent 必须暴露分歧，不能静默选择一方。 |

## 延伸阅读

- [Anthropic engineering — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) —— supervisor 模式的生产参考
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) —— tool-calling supervisor 现在是推荐形式
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) —— 旧版辅助函数，2026 年的生产环境仍在使用
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) —— 基于交接的 supervisor 变体
