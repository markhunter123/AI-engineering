# 生产环境扩展 —— 队列、checkpoint 与持久性

> 把多 Agent 系统扩展到数千个并发运行需要**持久执行**（durable execution）——工作队列加 checkpoint，使任何 worker 都能在任意崩溃后恢复任意运行，前提是租约处理、幂等副作用和确定性重放都已到位。LangGraph 的运行时是参考范例：它在每个 super-step 之后写入 checkpoint，以 `thread_id` 为键（默认 Postgres）；worker 崩溃会释放租约，由另一个 worker 恢复。Agent 可以无限期休眠等待人类输入。**MegaAgent**（arXiv:2408.09955）为每个 Agent 运行生产者-消费者队列，含三种状态（Idle / Processing / Response）和两层协调（组内聊天 + 组间管理聊天）。**Fiber/async** 在 LLM 流式场景中优于每任务一线程：线程 99% 的时间都在等待 token 而空转，fiber 在 I/O 上协作式让出。反方观点：Ashpreet Bedi 的「Scaling Agentic Software」主张在负载证明需要之前坚持 **FastAPI + Postgres + 别无他物**——简单架构能走得比预期更远。本课构建持久 checkpoint 日志、带状态转换的逐 Agent 工作队列、异步 vs 线程演示，并落实务实的「从简开始」原则。

**Type:** Learn + Build
**Languages:** Python (stdlib, `asyncio`, `sqlite3`)
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 13 (Shared Memory)
**Time:** ~75 minutes

## 问题

一个多 Agent 系统原型在内存事件循环中、在一台笔记本上以三个 Agent 运行良好。你把它移到生产环境：

- Agent 有时会运行数小时（长时间研究、等待人类介入）。
- worker 进程会崩溃。重启会丢失状态。
- 峰值负载是平均值的 10 倍；你需要水平扩展。
- 用户按次 Agent 运行付费；你需要精确一次的计费语义。

内存事件循环一项都做不到。你需要在下面有一层持久执行。2026 年的经典选项是：

1. 带 checkpoint 的工作流引擎（Temporal、LangGraph runtime）。
2. 带状态存储的消息队列（Postgres + SQS/RabbitMQ）。
3. Actor 模型框架（MegaAgent 的逐 Agent 生产者-消费者）。
4. 手写 FastAPI + Postgres（Bedi 的论点）。

本课为每一种构建一个缩影。

## 概念

### 持久执行模式

持久执行引擎在每个「步骤」（用 LangGraph 的话说，super-step）之后持久化完整的程序状态。崩溃时：

```
worker crashes mid-step
  -> lease timeout
  -> another worker picks up the thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects
```

要使其工作，需要：

- **可序列化状态。** 所有 Agent 状态都必须可持久化。持有活数据库连接的函数闭包无法存活。
- **确定性恢复。** 给定相同状态和相同输入，Agent 产生相同动作（或把 LLM 调用交给外部确定性预言机）。
- **幂等副作用。** 外部调用（tool call、支付）必须幂等，或使用去重键。

LangGraph 在每个 super-step 后写 checkpoint；Temporal 在每个 activity 后写；Restate 使用事件溯源日志。三者实现的是同一模式。

### 每步一 checkpoint 的运行时

LangGraph 的运行时是范例：每个 Agent 有一个 `thread_id`；状态是带类型的 dict；每个 super-step 向 checkpoints 表写入一行。恢复时，运行时从最后一个 checkpoint 重放，而不是从头开始。Agent 可以 `interrupt()` 等待人类输入；运行时会持久化并释放 worker。输入到达时，任何 worker 都可以恢复。

这是 2026 年 4 月的参考生产设计。

### MegaAgent 的逐 Agent 队列

arXiv:2408.09955 描述了一个规模实验：一个集群中数千个并发 Agent。架构：

```
agent i:
  state ∈ {Idle, Processing, Response}
  in_queue   <- messages addressed to agent i
  out_queue  -> replies + side effects

coordinators:
  intra-group chat  (agents in the same group)
  inter-group admin chat  (high-level routing)
```

两层协调让组内对话保持密集，而组间保持稀疏——这是让成本在数千个 Agent 下保持线性的模式。

### 异步 vs 每任务一线程

LLM 调用是 I/O 密集型的。等待下一个 token 的线程 99% 的时间处于空闲。每个线程约占 1MB 内存；在 10,000 个并发调用下，光栈就要 10GB。

Fiber（Python `asyncio`、Go goroutine、Rust `tokio`）在 I/O 上协作式让出。同样的 10,000 个调用可以轻松放进一个进程。在 LLM Agent 的规模上，异步不是优化——它就是架构。

例外：CPU 密集的后处理（embedding、tokenizer 技巧）仍然需要线程或进程。把 I/O 层与 CPU 层分开。

### Bedi 的反方观点

「Scaling Agentic Software」（Ashpreet Bedi，2026）认为，大多数团队在测量负载之前就过度设计。务实的默认做法：

- FastAPI + Postgres。
- 每次 Agent 运行是一行；用乐观并发就地更新状态。
- 通过 `pg_notify` 或简单的 Celery worker 执行后台任务。
- 在应用代码中实现重试策略。

对于约 100 个并发 Agent 运行以内、任务可控的负载，这通常就足够了。测出它失败时再升级。

规则是：当你遇到简单架构无法解决的具体问题时，再采用持久执行框架。过早采用会把时间浪费在不会带来回报的仪式上。

### 精确一次语义

对于付费的 Agent 运行，你需要「有效精确一次」（至少一次投递 + 幂等消费者）。工程手段：

- **每次运行的去重键。** 把它包含在每次副作用调用中。
- **发件箱模式。** 副作用先写入一张表，再由单独进程执行。两个步骤都幂等。
- **补偿事务。** 当副作用成功但其跟踪写入失败时，安排一次补偿。

这些是数据库工程模式，并非 LLM 特有。LLM 带来的额外负担只是 LLM 调用很慢；其余都是标准分布式系统问题。

### 彩虹部署

Anthropic 的多 Agent 研究系统使用「彩虹部署」：多个版本的 Agent 运行时并行运行，这样长时运行的 Agent 不必在每次代码部署时被杀死。在一部分流量上对新版本做金丝雀发布；旧版本的 Agent 完成后退役旧版本。

这对长时运行的有状态系统是标准做法；2026 年的适配点在于 Agent 可以存活数小时，因此部署周期必须相适应。

### 经典生产清单

- 持久状态（checkpoint、快照，或发件箱 + 可重放日志）。
- 幂等副作用。
- 用于 LLM 调用的异步 I/O 层。
- 带去重的至少一次投递。
- 面向有状态工作负载的彩虹/金丝雀部署。
- 可观测性：逐 Agent 追踪、super-step 审计、重试计数器。

```figure
sw-checkpoint-replay
```

## 构建它

`code/main.py` 实现了：

- `CheckpointStore`——基于 SQLite 的 checkpoint 日志，以 thread-id 为键。每个 super-step 追加一行。
- `run_with_checkpoint(agent, thread_id)`——模拟运行中途崩溃；第二个 worker 从最后一个 checkpoint 恢复。
- `AgentQueue`——逐 Agent 的 Idle / Processing / Response 状态机，带一个小型工作队列。
- `demo_async_vs_threads()`——通过 asyncio 和线程分别运行 500 个并发模拟「LLM 调用」；报告墙钟时间和峰值内存（近似值）。

运行：

```
python3 code/main.py
```

预期输出：模拟崩溃后 checkpoint 恢复成功；异步版本在 1 秒内处理 500 个并发调用；线程版本需要数秒，并且每个并发单元的内存占用高出几个数量级。

## 使用它

`outputs/skill-scaling-advisor.md` 就持久执行的选择提供建议：FastAPI + Postgres、LangGraph runtime、Temporal 或自研。依据负载、状态保留需求和部署频率进行校准。

## 交付它

经典生产加固：

- **从简开始（Bedi 的规则）。** FastAPI + Postgres，直到你测出它失败。
- **先加监控再优化。** 每次运行的延迟直方图、每步耗时、重试次数、失败分类。
- **副作用用发件箱模式。** 尤其是支付和外部 API 调用。
- **彩虹部署。** 部署期间绝不要杀死进行中的 Agent 运行。
- **在遇到具体问题时采用持久执行引擎（Temporal / LangGraph / Restate）：** 长达数小时的人类介入等待、跨区域协调、复杂的重试/补偿策略。
- **I/O 层用异步。** 线程只用于 CPU 密集的后处理。

## 练习

1. 运行 `code/main.py`。确认 checkpoint 恢复有效；测量异步与线程的并发差异。
2. 实现一张**发件箱**表：每次 tool call 先写入发件箱，再由单独的 goroutine/任务执行。通过把 tool call 运行两次来验证幂等性。
3. 模拟一次**彩虹部署**：两个并发的运行时版本；把一半的新 thread_id 路由到每个版本；确认旧版本上进行中的线程不被打断。
4. 阅读 LangGraph 的运行时文档（下方链接）。确定运行时中哪些功能在手写 FastAPI + Postgres 版本中最难复现。这是采用它的理由，还是可以推迟？
5. 阅读 MegaAgent（arXiv:2408.09955）第 3 节。两层协调（组内 + 组间管理聊天）是显式的。勾勒你将如何把它映射到两个队列家族的消息队列。

## 关键术语

| 术语 | 常见说法 | 真实含义 |
|------|----------------|------------------------|
| 持久执行 | 「持久化程序状态」 | 引擎在每个 super-step 后写入状态；崩溃恢复是确定性的。 |
| Super-step | 「事务边界」 | Checkpoint 之间的工作单元。LangGraph 术语。 |
| thread_id | 「Agent 运行标识」 | 绑定 checkpoint 和恢复逻辑的键。 |
| 幂等性 | 「可安全重试」 | 重复副作用的结果与一次尝试相同。 |
| 发件箱模式 | 「解耦副作用」 | 把意图写入表；由单独的执行器执行并标记完成。 |
| 至少一次投递 | 「可能重复」 | 消息队列语义；去重键使消费者有效一次。 |
| 彩虹部署 | 「版本重叠」 | 长时工作负载期间多个运行时版本并发。 |
| 异步 fiber | 「协作式让出」 | 用户态并发；对 I/O 密集型负载而言比线程便宜。 |
| Checkpoint | 「状态快照」 | super-step 边界处的序列化状态；恢复的键。 |

## 延伸阅读

- [LangChain — 生产级深度 Agent 背后的运行时](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — LangGraph 运行时设计
- [MegaAgent](https://arxiv.org/abs/2408.09955) — 逐 Agent 生产者-消费者队列；数千并发 Agent 下的两层协调
- [Matrix](https://arxiv.org/abs/2511.21686) — 以消息队列为协调基质的去中心化框架
- [Temporal 文档](https://docs.temporal.io/) — 持久执行的参考工作流引擎
- [Anthropic — 多 Agent 研究系统](https://www.anthropic.com/engineering/multi-agent-research-system) — 包括彩虹部署在内的生产经验
