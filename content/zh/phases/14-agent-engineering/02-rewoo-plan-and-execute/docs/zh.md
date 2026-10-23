# ReWOO 与 Plan-and-Execute：解耦规划

> ReAct 在同一个流里交错思考与行动。ReWOO 把二者分开：先做一份完整的计划，再执行。token 用量减少到五分之一，HotpotQA 准确率提升 4 个点，还能把规划器蒸馏进一个 7B 模型。Plan-and-Execute 对它做了泛化；Plan-and-Act 把它扩展到了网页导航。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 minutes

## 学习目标

- 解释为什么 ReWOO 的 Planner / Worker / Solver 拆分比 ReAct 的交错循环更省 token、更稳健。
- 实现一个计划 DAG、一个按依赖顺序执行的执行器，以及一个组合 worker 输出的 solver——全部使用 stdlib。
- 借助 2026 年的「五种工作流模式」框架（Anthropic），判断一个任务应当采用先规划后执行，还是交错的 ReAct。
- 识别何时需要 Plan-and-Act 的合成计划数据来处理长时程的网页或移动端任务。

## 问题

ReAct 交错的思考-行动-观察循环简单而灵活，但每次工具调用都必须携带完整的先前上下文——包括此前的每一个思考。token 用量随深度呈二次增长。更糟的是：当某个工具在循环中途失败，模型必须根据错误观察重新推导整个计划。

ReWOO（Xu 等，arXiv:2305.18323，2023 年 5 月）注意到了这一点，并押了一注：先把整件事规划好，并行获取证据，最后组装答案。一次 LLM 调用做规划，N 次工具调用取证据（可以并行），一次 LLM 调用求解。代价是灵活性降低（计划是静态的），换来的是好得多的 token 效率和更清晰的失败模式。

## 概念

### 三个角色

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner 产出一个 DAG。每个节点指定一个工具、它的参数，以及依赖哪些更早的节点（使用 `#E1`、`#E2` 这样的引用）。Worker 按拓扑顺序执行节点。Solver 把所有内容缝合起来。

### 为什么 token 能少五倍

ReAct 的 prompt 长度随步数线性增长。到第 10 步时，prompt 里包含思考 1、行动 1、观察 1、思考 2、行动 2、观察 2，依此类推。每个中间步骤还会冗余地包含原始 prompt。

ReWOO 只需付出一个规划器 prompt（较大）、N 个小型 worker prompt（每个只有工具调用，不带链条）和一个 solver prompt。在 HotpotQA 上，论文测得 token 约减少到五分之一，同时绝对准确率提升 4 个点。

### 为什么它更稳健

在 ReAct 中，如果 worker 3 失败，循环必须在流中途从错误中推理出来。在 ReWOO 中，worker 3 返回一个错误字符串；solver 结合原始计划在上下文中看到它，可以优雅降级。失败定位是按节点的，而不是按步骤的。

### 规划器蒸馏

论文的第二个结果：由于规划器看不到观察结果，你可以用 175B 教师模型的规划器输出微调一个 7B 模型。小模型负责规划；推理时不再需要大模型。这如今已是标准做法——2026 年许多生产智能体使用小规划器配大执行器，或者反过来。

### Plan-and-Execute（2023）

LangChain 团队 2023 年 8 月的文章把 ReWOO 泛化成一个模式名称：Plan-and-Execute。前置规划器输出一个步骤列表，执行器逐条执行，可选的 re-planner 可以在观察到结果后修订。它比 ReWOO 更接近 ReAct（re-planner 把观察结果带回规划环节），但保留了 token 节省。

### Plan-and-Act（Erdogan 等，arXiv:2503.09572，ICML 2025）

Plan-and-Act 把这个模式扩展到长时程的网页与移动端智能体。关键贡献是合成计划数据：一个带标注的轨迹生成器产出计划显式的训练数据。它被用来微调规划器模型，使其在 WebArena 一类任务上超过 30–50 步后仍能工作——在那些任务中，单条 ReAct 轨迹会失去连贯性。

### 如何选择

| 模式 | 适用场景 |
|---------|------|
| ReAct | 短任务、环境未知、需要反应式的异常处理 |
| ReWOO | 工具已知的结构化任务、对 token 敏感、证据可并行获取 |
| Plan-and-Execute | 类似 ReWOO，但支持部分执行后的重新规划 |
| Plan-and-Act | 长时程（>30 步）、网页 / 移动端 / 计算机使用 |
| Tree of Thoughts | 搜索的代价是值得的（第 04 课） |

Anthropic 2024 年 12 月的建议：从最简单的开始。如果任务只是一次工具调用加一段摘要，不要建 ReWOO。如果任务是 40 步的研究任务，不要只用 ReAct。

```figure
rewoo-plan
```

## 构建它

`code/main.py` 实现了一个玩具版 ReWOO：

- `Planner`——一个脚本化策略，根据 prompt 输出计划 DAG。
- `Worker`——通过注册表分派每个节点的工具调用。
- `Solver`——脚本化的组装逻辑，读取证据并产出最终答案。
- 依赖解析——`#E1` 这样的引用会被替换成更早的 worker 输出。

演示回答的问题：「法国首都的人口是多少，按百万取整？」它用一个两步计划：(1) 查首都，(2) 查人口，然后求解。

运行它：

```
python3 code/main.py
```

轨迹先展示完整计划，然后是 worker 结果，最后是 solver 的组装。把 token 数（我们打印的是粗略的字符数）与 ReAct 式交错运行对比——在这类结构化任务上 ReWOO 更胜一筹。

## 使用它

LangGraph 把 Plan-and-Execute 作为一份配方提供（ReAct 用 `create_react_agent`，plan-execute 用自定义图）。CrewAI 的 Flows 直接编码了这个模式：你先定义任务，由 Flow DAG 执行它们。Plan-and-Act 的合成数据方法基本还停留在研究阶段；运行时模式（显式计划 DAG）已经通过 LangGraph 和 CrewAI Flows 进入生产。

## 交付它

`outputs/skill-rewoo-planner.md` 在给定工具目录的情况下，根据用户请求生成 ReWOO 计划 DAG。它在交给执行器之前会校验计划（无环、每个引用都能解析、每个工具都存在）。

## 练习

1. 对相互独立的计划节点并行执行 worker。在一个含 2 个并行组的 6 节点 DAG 上，这能带来什么收益？
2. 增加一个 re-planner 节点，在任意 worker 返回错误时触发。对 ReWOO 做的最小改动是什么，才能把它变成 Plan-and-Execute？
3. 把 `Planner` 换成小模型（7B 级别），`Solver` 保留在前沿模型上。比较端到端质量——这个拆分在哪里失效？
4. 阅读 ReWOO 论文中关于规划器蒸馏的第 4 节。在概念上复现 175B -> 7B 的结果：你需要什么样的训练数据，又该如何给计划质量打分？
5. 把这个玩具实现移植到 Plan-and-Act 的轨迹形态：计划是一个序列，而不是 DAG。哪些权衡会发生变化？

## 关键术语

| 术语 | 人们常怎么说 | 实际含义 |
|------|----------------|------------------------|
| ReWOO | 「不带观察的推理」 | 先规划，再并行获取证据，最后求解——规划 prompt 中没有观察结果 |
| Plan-and-Execute | 「LangChain 的 plan-execute 模式」 | 带可选 re-planner 节点的 ReWOO |
| Plan-and-Act | 「扩展版 plan-execute」 | 显式的规划器 / 执行器拆分，配合面向长时程任务的合成计划训练数据 |
| Evidence reference | 「#E1、#E2……」 | 计划节点中的占位符，分派时替换为先前的 worker 输出 |
| Planner distillation | 「小规划器，大执行器」 | 用大教师模型的规划器轨迹微调小模型 |
| Token efficiency | 「更少的往返」 | 论文中在 HotpotQA 上相比 ReAct token 减少到五分之一 |
| DAG executor | 「拓扑分派器」 | 按依赖顺序运行计划节点；每一层可并行 |

## 进一步阅读

- [Xu 等，ReWOO: Decoupling Reasoning from Observations（arXiv:2305.18323）](https://arxiv.org/abs/2305.18323)——经典论文
- [Erdogan 等，Plan-and-Act（arXiv:2503.09572）](https://arxiv.org/abs/2503.09572)——使用合成计划的扩展版规划器-执行器
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview)——框架配方
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)——选择能起作用的最简单模式
