# 长时间运行的后台智能体：持久化执行

> 生产环境中的长时程智能体不会跑在 `while True` 里。每一次 LLM 调用都变成一个活动，带有 checkpoint、重试和重放。Temporal 的 OpenAI Agents SDK 集成于 2026 年 3 月 GA。Claude Code Routines（Anthropic）无需本地常驻进程即可运行定时调度的 Claude Code 调用。会话在等待人类输入时暂停，能挺过部署，并从以 `thread_id` 为键的最新 checkpoint 恢复。这套新的人机体验背后是一个老模式——工作流编排——只多了一种新输入：LLM 调用作为非确定性活动，必须在恢复时确定性地重放。

**Type:** Learn
**Languages:** Python (stdlib, minimal durable-execution state machine)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~60 minutes

## 问题

设想一个运行四小时的智能体。它调用三个工具、两次提示用户、进行四十次 LLM 调用。中途，它所在的主机重启了。会发生什么？

- 在朴素的 `while True` 循环中：一切丢失。运行从头开始。三个工具调用（带有真实副作用）再次执行。用户为已经批准过的事情再次被提示。四十次 LLM 调用被重复计费。
- 使用持久化执行：运行从最近的 checkpoint 恢复。已完成的活动不会重新执行；它们的结果从持久日志中重放。用户无需重新批准已经批准过的事。已完成的 LLM 调用不会被重复计费。

这是工作流引擎已经交付了十年的模式（Temporal、Cadence、Uber 的 Cherami）。新变化在于，LLM 调用现在也成了一种活动——非确定性、昂贵、带副作用——并且它们与这个模式契合得很好。

本课贯穿始终的主题是：长时程可靠性会衰减（METR 观察到「35 分钟退化」——成功率随时程大致呈二次方下降）。持久化执行使运行时长可以超出可靠性画像的支撑范围。如果设计正确，这是一种新的安全失败方式；如果设计错误，就是一种新的不安全失败方式。

## 概念

### 活动、工作流与重放

- **工作流（Workflow）**：确定性的编排代码。定义活动的顺序、分支和等待。必须是确定性的，才能从事件日志重放而不出现意外的分叉。
- **活动（Activity）**：一个非确定性的、可能失败的工作单元。LLM 调用、工具调用、文件写入、HTTP 请求。每个活动都会记录其输入以及（完成后的）输出。
- **事件日志**：持久化的后备存储。每次活动的开始、完成、失败、重试，以及每个工作流决策都会被记录。
- **重放（Replay）**：恢复时，工作流代码从头重新运行；每个已完成的活动直接返回其记录的结果，而不重新执行。只有尚未完成的活动才会真正运行。

这与 React 基于虚拟 DOM 重新渲染，或 Git 从提交重建工作树是一样的形态。编排器的确定性正是让持久化成本低廉的原因。

### 为什么 LLM 调用适合这个模式

LLM 调用是：
- 非确定性的（temperature > 0；即便是 temperature 0，跨模型版本也会漂移）。
- 昂贵的（金钱与延迟）。
- 可能失败的（速率限制、超时）。
- 有副作用的（如果它们调用了工具）。

这正是活动的特征画像。把每次 LLM 调用包装为一个活动，就能获得指数退避重试、跨重启的 checkpointing，以及可重放用于调试的轨迹。

### 以 `thread_id` 为键的 checkpoint

LangGraph、Microsoft Agent Framework、Cloudflare Durable Objects 和 Claude Code Routines 都收敛到了相同的 API 形态：用 `thread_id`（或等价物）标识会话；每次状态转换都持久化到某种后端（默认 PostgreSQL，开发用 SQLite，缓存用 Redis）；恢复时读取最新的 checkpoint。

后端的选择很重要：

- **PostgreSQL**：持久、可查询、能挺过部署。LangGraph 的默认选择。
- **SQLite**：仅限本地开发；跨主机会丢数据。
- **Redis**：快，但除非配置了 AOF/快照，否则是临时的。
- **Cloudflare Durable Objects**：透明分布；按唯一键划定范围；可存活数小时到数周。

### 把人类输入当作一等状态

先提议后提交（第 15 课）需要一个持久的「等待人类」状态。工作流暂停，外部队列保存待处理请求，一次批准可以从确切的那一点恢复。没有持久化，这只是尽力而为；有了它，隔夜的批准到达后，工作流第二天早上就能继续。

### 35 分钟退化

METR 观察到，所测量的每一类智能体在连续运行约 35 分钟后都会出现可靠性衰减。任务时长翻倍，失败率大约变为四倍。持久化执行并不能修复这一点；它让你能运行得比可靠性画像支撑的更久。安全的模式是把持久化与「重新进入时需要新 HITL」的 checkpoint 结合起来，并结合预算熔断开关（第 13 课）——无论墙钟时间多长，都对总算力设上限。

### 持久化执行不适用的场景

- 运行时间不到几分钟且无人类输入的场景。开销大于收益。
- 严格只读的信息检索。
- 正确性要求在一个上下文窗口内端到端完成的任务（某些推理任务；某些一次性生成任务）。

```figure
memory-consolidation
```

## 使用它

`code/main.py` 用标准库 Python 实现了一个最小的持久化执行引擎。它支持：

- `@activity` 装饰器，把输入和输出记录到 JSON 事件日志。
- 一个按顺序编排活动的工作流函数。
- 一个 `run_or_replay(workflow, event_log)` 函数，重放已完成的活动而不重新执行它们。

驱动程序模拟一个三活动的工作流，在中途崩溃，并展示 (a) 朴素重试重新执行所有内容，对比 (b) 重放只运行缺失的活动。

## 交付它

`outputs/skill-durable-execution-review.md` 审查一个候选的长时运行智能体部署是否具备正确的持久化执行形态：活动、确定性、checkpoint 后端、人类输入状态，以及恢复时的 HITL 策略。

## 练习

1. 运行 `code/main.py`。观察朴素重试与重放在活动执行次数上的差异。改变崩溃点，展示重放次数随之变化。

2. 把玩具引擎改为显式使用 `thread_id`。模拟两个共享该引擎的并发会话，确认它们的事件日志不会冲突。

3. 取玩具引擎中的一个活动，引入一处非确定性（工作流决策内部的墙钟时间戳）。演示重放时的分叉。解释真实引擎如何处理这种情况（副作用注册、`Workflow.now()` API）。

4. 阅读 LangChain 的「Runtime behind production deep agents」文章。列出运行时持久化的每一种状态，并说明每种状态覆盖的失效模式。

5. 为一个 6 小时的自主编码任务设计 checkpoint 策略。在哪里 checkpoint？崩溃恢复是什么样子？哪些需要新的 HITL？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| 工作流 | 「智能体的脚本」 | 确定性编排代码；可从事件日志重放 |
| 活动 | 「一个步骤」 | 非确定性单元（LLM 调用、工具调用）；前后均有日志 |
| 事件日志 | 「后备存储」 | 每次状态转换的持久记录 |
| 重放 | 「恢复」 | 重新运行工作流；已完成活动返回记录结果而不重新执行 |
| Checkpoint | 「存档点」 | 以 thread_id 为键的持久化状态；恢复时最新者生效 |
| thread_id | 「会话键」 | 划定持久状态范围的标识符 |
| 35 分钟退化 | 「可靠性衰减」 | METR：成功率随时程大致按二次方下降 |
| 非确定性 | 「重放时漂移」 | 墙钟、随机数、LLM 输出；必须注册为副作用 |

## 延伸阅读

- [Anthropic —— Claude Code Agent SDK：智能体循环](https://code.claude.com/docs/en/agent-sdk/agent-loop) —— 预算、轮次与恢复语义。
- [Microsoft —— Agent Framework：人在回路与 checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) —— RequestInfoEvent 形态。
- [LangChain —— The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) —— 具体的运行时需求。
- [OpenAI Agents SDK + Temporal 集成（Trigger.dev 公告）](https://trigger.dev) —— LLM 调用的活动形态。
- [Anthropic —— 在实践中测量智能体自主性](https://www.anthropic.com/research/measuring-agent-autonomy) —— 35 分钟退化的出处。
