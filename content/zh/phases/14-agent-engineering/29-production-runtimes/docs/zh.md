# 生产运行时：队列、事件、定时

> 生产 Agent 运行在六种运行时形态上：请求-响应、流式、持久化执行、基于队列的后台、事件驱动、定时。先选形态，再选框架。可观测性在每一种形态中都是承重的。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 22 (Voice)
**Time:** ~60 minutes

## 学习目标

- 说出六种生产运行时形态，并把每一种对应到框架 / 产品模式。
- 解释持久化执行（LangGraph）为何对长程任务重要。
- 描述事件驱动运行时，以及 Claude Managed Agents 何时适用。
- 解释「可观测性承重」这一主张对多步 Agent 的含义。

## 问题

生产 Agent 以 Jupyter notebook 无法暴露的方式失败：第 37 步的网络超时、用户在语音通话中途挂断、cron 作业因机器重启而死亡、后台 worker 内存耗尽。运行时形态决定了哪些失败可以被承受。

## 概念

### 请求-响应

- 同步 HTTP。用户等待完成。
- 只适用于短任务（<30s）。
- 技术栈：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- 可观测性：标准 HTTP 访问日志 + OTel span。

### 流式

- 用 SSE 或 WebSocket 逐步输出。
- LiveKit 将其扩展到 WebRTC，用于语音 / 视频（第 22 课）。
- 技术栈：任何支持流式的框架 + 能处理 SSE/WS 的前端。
- 可观测性：每个分块的耗时、首 token 延迟、尾延迟。

### 持久化执行

- 每一步之后对状态做 checkpoint；失败后自动恢复。
- AutoGen v0.4 的 actor 模型把故障隔离在单个 Agent 内（第 14 课）。
- LangGraph 的核心差异化能力（第 13 课）。
- 当步数未知且恢复成本高昂时不可或缺。

### 基于队列 / 后台

- 作业进入队列，worker 取走，结果通过 webhook 或 pub/sub 回流。
- 对长程 Agent 不可或缺（按 Anthropic 计算机使用公告的说法，每个任务数十到数百步）。
- 技术栈：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）、自定义。
- 可观测性：队列深度、每作业延迟分布、DLQ 大小。

### 事件驱动

- Agent 订阅触发器：新邮件、PR 打开、cron 触发。
- Claude Managed Agents 开箱即用地覆盖这一点（第 17 课）。
- CrewAI Flows（第 15 课）组织事件驱动的确定性工作流。
- 可观测性：触发源、事件到启动的延迟、Agent 延迟。

### 定时

- 周期性运行的 cron 形态 Agent。
- 与持久化执行结合，让失败的夜间运行在下一个 tick 恢复。
- 技术栈：Kubernetes CronJob + 持久化框架；托管方案（Render cron、Vercel cron）。

### 2026 年的部署模式

- **CrewAI Flows** 用于事件驱动生产。
- **Agno** 无状态 FastAPI，用于 Python 微服务。
- **Mastra** 服务器适配器（Express、Hono、Fastify、Koa）用于嵌入。
- **Pipecat Cloud / LiveKit Cloud** 用于托管语音（第 22 课）。
- **Claude Managed Agents** 用于托管的长时异步任务。

### 可观测性是承重的

没有 OpenTelemetry GenAI span（第 23 课）以及 Langfuse/Phoenix/Opik 后端（第 24 课），你无法调试在第 40 步失败的多步 Agent。这对生产不是可选项。它决定了你是「快速调试」还是「加上更多日志从头重放」。

### 生产运行时容易出错的地方

- **形态选错。** 为 5 分钟的任务选择请求-响应。用户挂断；worker 堆积；重试叠加。
- **没有 DLQ。** 没有死信队列的队列 worker。失败的作业消失无踪。
- **不透明的后台工作。** 后台 Agent 运行但不导出轨迹。失败在用户报告之前不可见。
- **跳过持久化状态。** 任何超过 30 秒、且你无法承受重启的运行都需要持久化执行。

```figure
wb-runtime-shapes
```

## 构建它

`code/main.py` 是一个 stdlib 多形态演示：

- 请求-响应端点（普通函数）。
- 流式处理器（生成器）。
- 带 DLQ 的基于队列的 worker。
- 事件触发器注册表。
- cron 形态的调度器。

运行它：

```bash
python3 code/main.py
```

输出：五条轨迹，展示每种形态在同一任务上的行为。相同的 Agent 逻辑，不同的外壳。持久化执行（第六种形态）在第 13 课用 LangGraph checkpointing 专门覆盖。

## 使用它

- **请求-响应** 用于聊天式 UX。
- **流式** 用于渐进式响应。
- **持久化执行** 用于长程任务。
- **队列** 用于 batch / 异步 / 长时间运行。
- **事件** 用于 Agent 的响应式触发。
- **Cron** 用于日常维护（记忆整合、评测、成本报告）。

## 交付它

`outputs/skill-runtime-shape.md` 为任务选择运行时形态，并接线可观测性要求。

## 练习

1. 把第 01 课的 ReAct 循环移植到你的技术栈中全部六种形态。哪种形态适合哪个产品面？
2. 为基于队列的演示添加 DLQ。模拟 10% 作业失败；呈现 DLQ 大小。
3. 编写一个 cron 触发的评测 Agent，每晚针对当天的 top 20 条轨迹运行。
4. 实现带背压的流式：如果客户端较慢，暂停 Agent。这与轮次预算如何相互作用？
5. 阅读 Claude Managed Agents 文档。何时会把自托管的长程 Agent 迁移到托管方案？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 请求-响应（Request-response） | 「同步」 | 用户等待；仅限短任务 |
| 流式（Streaming） | 「SSE / WS」 | 渐进输出；更好的 UX；按分块观测延迟 |
| 持久化执行（Durable execution） | 「失败后恢复」 | 状态做 checkpoint；从最后一步重启 |
| 基于队列（Queue-based） | 「后台作业」 | 生产者 / worker 池 / DLQ |
| 事件驱动（Event-driven） | 「基于触发器」 | Agent 响应外部事件 |
| DLQ | 「死信队列」 | 失败作业的停车场 |
| Claude Managed Agents | 「托管 harness」 | Anthropic 托管的长时异步，带缓存 + 压缩 |

## 进一步阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 持久化执行的细节
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) —— 托管的长时异步
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) —— 「每个任务数十到数百步」
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) —— actor 模型的故障隔离
