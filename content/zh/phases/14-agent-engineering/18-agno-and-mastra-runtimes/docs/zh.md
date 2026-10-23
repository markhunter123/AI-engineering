# 生产级智能体运行时——快速实例化与类型化工作流

> 生产级智能体运行时优化的是原型框架忽视的东西：实例化成本、类型化工作流表面，以及可直接用于服务的后端。2026 年的搭配：Agno（Python）瞄准微秒级智能体实例化和无状态 FastAPI 后端。Mastra 在 Vercel AI SDK 基座上提供智能体、工具、工作流、统一模型路由和复合存储。

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time:** ~45 minutes

## 学习目标

- 指出 Agno 的性能目标，以及它们何时重要。
- 说出 Mastra 的三个原语——Agents、Tools、Workflows——以及支持的服务器适配器。
- 解释为什么无状态、session 作用域的 FastAPI 后端是推荐给 Agno 的生产路径。
- 针对给定技术栈（Python 优先还是 TypeScript 优先）选择 Agno 或 Mastra。

## 问题

LangGraph、AutoGen、CrewAI 都属于重框架。想要「只要智能体循环，速度快，跑在我的运行时里」的团队会转向 Agno（Python）或 Mastra（TypeScript）。两者都放弃了一些框架自有的原语，换取原始速度和与周边技术栈更紧密的契合。

## 概念

### Agno

- Python 运行时，前身是 Phi-data。
- 「没有图、链或绕来绕去的模式——只有纯 Python。」
- 文档中的性能目标：约 2μs 智能体实例化，每个智能体约 3.75 KiB 内存，约 23 个模型提供方。
- 生产路径：无状态、session 作用域的 FastAPI 后端。每个请求启动一个新智能体；session 状态存放在数据库中。
- 原生多模态（文本、图像、音频、视频、文件）和 agentic RAG。

当你每秒有数千个短生命周期智能体时（聊天扇入、评估流水线），速度目标很重要。当一个智能体要跑 10 分钟时，它们就没那么重要了。

### Mastra

- TypeScript，构建在 Vercel AI SDK 之上。
- 三个原语：**Agents**、**Tools**（Zod 类型化）、**Workflows**。
- 统一 Model Router —— 覆盖 94 个提供方的 3,300+ 个模型（2026 年 3 月）。
- 复合存储：记忆、工作流、可观测性可分别写入不同后端；大规模可观测性推荐 ClickHouse。
- Apache 2.0，其中 `ee/` 目录采用源码可见的企业许可证。
- Express、Hono、Fastify、Koa 的服务器适配器；一流的 Next.js 和 Astro 集成。
- 附带 Mastra Studio（localhost:4111）用于调试。
- 1.0 版本（2026 年 1 月）时 22k+ GitHub 星标，每周 300k+ npm 下载。

### 定位

两者都不打算成为 LangGraph。它们的竞争点在于：

- **语言契合度。** Agno 面向 Python 优先的团队；Mastra 面向 TypeScript 优先的团队。
- **运行时体验。** Agno = 近乎零开销；Mastra = 与 Vercel 生态集成。
- **可观测性。** 两者都集成 Langfuse/Phoenix/Opik（第 24 课），但 Mastra Studio 是第一方的。

### 何时选择

- **Agno** —— Python 后端、大量短生命周期智能体、强性能要求、FastAPI 团队。
- **Mastra** —— TypeScript 后端、Next.js / Vercel 部署、统一多提供方模型路由、Zod 类型化工具。
- **LangGraph**（第 13 课）—— 当持久状态和显式图推理比原始速度更重要时。
- **OpenAI / Claude Agent SDK** —— 当你想要提供方产品化的形态时（第 16–17 课）。

### 这个模式会在哪里出错

- **为性能而性能。** 工作负载是每次请求一个慢速智能体调用，却因为「2μs」听起来不错而选 Agno。开销不是瓶颈。
- **生态锁定。** Mastra 的 Vercel 风味集成在 Vercel 上是加分项，在其他地方是减分项。
- **企业许可证混淆。** Mastra 的 `ee/` 目录是源码可见，不是 Apache 2.0。如果打算 fork，请阅读许可证。

```figure
wb-runtime-spawn
```

## 构建它

这节课主要是比较性的——没有任何单一代码产物能公平对待两个框架。见 `code/main.py` 中的并排玩具：一个最小的「运行智能体、流式输出、持久化 session」流程实现两遍（一次 Agno 风格，一次 Mastra 风格）。

运行它：

```
python3 code/main.py
```

两条结构不同但功能等价的 trace。

## 使用它

- **Agno** —— 需要速度和 FastAPI 形态的 Python 后端。
- **Mastra** —— 需要多个提供方和工作流原语的 TypeScript 后端。
- 两者都提供第一方可观测性 hook，都集成 Langfuse。

## 交付它

`outputs/skill-runtime-picker.md` 根据技术栈、延迟预算和运营形态，选择 Agno、Mastra、LangGraph 或提供方 SDK。

## 练习

1. 阅读 Agno 文档。把标准库 ReAct 循环（第 01 课）移植到 Agno。什么消失了？什么保留了下来？
2. 阅读 Mastra 文档。把同一个循环移植到 Mastra。工具类型化发生了什么变化（Zod 对比什么都没有）？
3. 基准测试：测量你技术栈上的智能体实例化延迟。Agno 的 2μs 对你的工作负载有意义吗？
4. 设计一次迁移：如果你一直用 Python 跑 CrewAI，迁到 Agno 会有什么坏掉？
5. 阅读 Mastra 的 `ee/` 许可证条款。哪些限制会影响一个开源 fork？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Agno | 「快速的 Python 智能体」 | 无状态、session 作用域的智能体运行时 |
| Mastra | 「Vercel AI SDK 上的 TypeScript 智能体」 | Agents + Tools + Workflows + Model Router |
| 统一 Model Router | 「多提供方访问」 | 一个客户端访问 94 个提供方的 3,300+ 模型 |
| 复合存储 | 「多后端」 | 记忆/工作流/可观测性分别写入不同存储 |
| Mastra Studio | 「本地调试器」 | localhost:4111 的智能体内省 UI |
| 源码可见 | 「不是开源」 | 许可证允许阅读源码但限制商业使用 |

## 延伸阅读

- [Agno Agent Framework 文档](https://www.agno.com/agent-framework) —— 性能目标、FastAPI 集成
- [Mastra 文档](https://mastra.ai/docs) —— 原语、服务器适配器、Model Router
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) —— 有状态图的替代方案
- [Comet Opik](https://www.comet.com/site/products/opik/) —— Mastra 集成所引用的可观测性对比
