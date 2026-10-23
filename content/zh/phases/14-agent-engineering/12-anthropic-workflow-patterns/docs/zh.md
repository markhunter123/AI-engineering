# Anthropic 的工作流模式：简单优于复杂

> Schluntz 与 Zhang（Anthropic，2024 年 12 月）区分了工作流（预定义路径）与智能体（动态工具使用）。五种工作流模式覆盖了大多数情况。从直接调用 API 开始。只有当步骤无法预测时才加入智能体。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 minutes

## 学习目标

- 说出 Anthropic 的五种工作流模式：prompt 链式调用、路由、并行化、编排者-工作者、评估者-优化者。
- 解释智能体与工作流的区别，以及各自的工程成本。
- 判断何时选择工作流而非智能体（以及反过来）。
- 用标准库针对脚本化 LLM 实现全部五种模式。

## 问题

团队会为只想要一次函数调用的问题，去抓多智能体框架。成本是真实的：框架增加了层，遮蔽 prompt、隐藏控制流，并招致过早的复杂化。Schluntz 与 Zhang 2024 年 12 月的文章是被引用最多的行业反思：从简单开始，只有当复杂度能挣回成本时才添加它。

## 概念

### 工作流 vs 智能体

- **工作流。** LLM 和工具通过预定义的代码路径编排。工程师拥有图。
- **智能体。** LLM 动态指挥自己的工具，采取自己的步骤。模型拥有图。

两者各有用武之地。工作流更便宜、更快、更容易调试。智能体解锁开放式问题，但也让失效模式更难推理。

### 增强型 LLM

全部五种模式的基础：一个 LLM 接入三种能力——搜索（检索）、工具（动作）、记忆（持久化）。任何 API 调用都可以使用它们。

### 五种模式

1. **Prompt 链式调用。** 调用 1 的输出是调用 2 的输入。适用于任务有清晰的线性分解时。步骤之间可选加入程序化闸门。

2. **路由。** 一个分类器 LLM 挑选下游要调用哪个 LLM 或工具。适用于类别不同的输入需要不同处理时（一级支持 vs 退款 vs bug vs 销售）。

3. **并行化。** 并发运行 N 个 LLM 调用，聚合结果。两种形态：分区（不同的块）和投票（同一 prompt，运行 N 次，多数表决/综合）。

4. **编排者-工作者。** 一个编排者 LLM 动态决定运行哪些工作者（也是 LLM），并综合它们的输出。类似智能体循环，但编排者不会无限循环。

5. **评估者-优化者。** 一个 LLM 提出答案，另一个 LLM 评估它。迭代直到评估者通过。这是 Self-Refine（第 05 课）的推广。

### 工作流胜过智能体之处

- **可预测的任务。** 如果你能枚举步骤，就应该枚举。
- **成本受限的任务。** 工作流的步数有界；智能体可能失控膨胀。
- **合规受限的任务。** 审计员想读图，而不是从轨迹中推断图。

### 智能体胜过工作流之处

- **开放式研究。** 下一步取决于上一步返回了什么。
- **变长任务。** 从几分钟到几小时的工作，步数未知。
- **新领域。** 当你还不知道正确的工作流时——先探索，后固化。

### 上下文工程的姊妹篇

「Effective context engineering for AI agents」（Anthropic 2025）形式化了相邻的学科：200k 窗口是预算，不是容器。该放什么、何时压缩、何时让上下文增长。详见第 14 阶段关于上下文压缩的课程（编号调整前为本课程的第 14 阶段第 06 课）。

```figure
workflow-chain
```

## 构建它

`code/main.py` 针对一个 `ScriptedLLM` 实现全部五种工作流模式：

- `prompt_chain(input, steps)` —— 顺序执行。
- `route(input, classifier, handlers)` —— 分类 + 分派。
- `parallel_vote(prompt, n, aggregator)` —— N 次运行，聚合。
- `orchestrator_workers(task, workers)` —— 编排者挑选工作者。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` —— 循环直到通过。

运行它：

```
python3 code/main.py
```

每种模式都会打印自己的轨迹。每种模式的代码总共约 10-15 行；框架的成本以千行计。

## 使用它

- 多数任务直接调用 API。
- 只有当模式确实需要持久状态（LangGraph）、actor 模型并发（AutoGen v0.4）或角色模板（CrewAI）时才用框架。
- 当你想要 Claude Code harness 的形态又不想重建它时，选择 Claude Agent SDK。

## 交付它

`outputs/skill-workflow-picker.md` 根据任务描述挑选正确的模式，包含决策理由，以及工作流不够用时的智能体重构路径。

## 练习

1. 用置信度阈值实现路由。低于阈值 -> 升级给人类。对于一级支持用例，阈值落在哪里？
2. 给 `parallel_vote` 增加超时。一个调用挂起时会发生什么？缺少投票时如何聚合？
3. 把 `evaluator_optimizer` 改造成 bandit：跨迭代保留 top-2 输出，这样迟来的好结果不会被迟来的坏结果覆盖。
4. 把 prompt 链式调用与路由结合：一个路由器挑三条链之一。与单个大 prompt 方案相比，测量 token 成本。
5. 选一个你的生产功能。画出工作流图。数一数步骤。这里用智能体真的更好吗？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 工作流（Workflow） | 「预定义流程」 | 工程师拥有的 LLM 与工具调用图 |
| 智能体（Agent） | 「自主 AI」 | 模型拥有的图；动态指挥工具 |
| 增强型 LLM | 「带工具的 LLM」 | LLM + 搜索 + 工具 + 记忆；原子单元 |
| Prompt 链式调用 | 「顺序调用」 | 调用 N 的输出是调用 N+1 的输入 |
| 路由 | 「分类器分派」 | 挑选哪条链/哪个模型处理输入 |
| 并行化 | 「扇出」 | N 个并发调用；按分区或投票聚合 |
| 编排者-工作者 | 「调度智能体」 | 编排者 LLM 动态挑选专家 LLM |
| 评估者-优化者 | 「提议者 + 裁判」 | 迭代直到评估者通过；Self-Refine 的推广 |

## 延伸阅读

- [Anthropic, Building Effective Agents（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents) —— 五种工作流模式
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) —— 姊妹学科
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) —— 有状态图何时能挣回成本
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) —— 产品化的编排者-工作者模式
