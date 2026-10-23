# 把 harness 当作库——子 Agent 与 session 存储

> 一个可以 import 的 harness：内置工具、用于上下文隔离的子 Agent、hook、W3C trace 传播、session 持久化。Claude Agent SDK 是参考示例——Claude Code harness 的库形态——而 Claude Managed Agents 是面向长时间异步工作的托管替代方案。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 10 (Skill Libraries)
**Time:** ~75 minutes

## 学习目标

- 解释 Anthropic Client SDK（原始 API）与 Claude Agent SDK（harness 形态）的区别。
- 描述子 Agent——并行化与上下文隔离——以及何时使用它们。
- 说出 Python SDK 的 session 存储接口（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`），以及 `--session-mirror` 的作用。
- 用标准库实现一个 harness，包含内置工具、带隔离上下文的子 Agent 派生、生命周期 hook 和 session 存储。

## 问题

裸 LLM API 只能给你一次往返。生产级 Agent 需要工具执行、MCP 服务器、生命周期 hook、子 Agent 派生、session 持久化、trace 传播。Claude Agent SDK 以库的形式提供这套形态——与 Claude Code 使用同一个 harness，开放给自定义 Agent。

## 概念

### Client SDK 与 Agent SDK

- **Client SDK（`anthropic`）。** 裸 Messages API。循环、工具、状态都由你负责。
- **Agent SDK（`claude-agent-sdk`）。** 内置工具执行、MCP 连接、hook、子 Agent 派生、session 存储。以库形式提供的 Claude Code 循环。

### 内置工具

SDK 开箱提供 10+ 个工具：文件读写、shell、grep、glob、网页抓取等。自定义工具通过标准的工具 schema 接口注册。

### 子 Agent

Anthropic 文档记载的两个用途：

1. **并行化。** 并发运行独立工作。「为这 20 个模块分别找到测试文件」就是 20 个并行的子 Agent 任务。
2. **上下文隔离。** 子 Agent 使用自己的上下文窗口；只有结果返回给编排器。编排器的预算得以保留。

Python SDK 近期新增：`list_subagents()`、`get_subagent_messages()`，用于读取子 Agent 的对话记录。

### Session 存储

与 TypeScript 协议对等：

- `append(session_id, message)` —— 添加一轮。
- `load(session_id)` —— 恢复对话。
- `list_sessions()` —— 枚举。
- `delete(session_id)` —— 级联删除子 Agent session。
- `list_subkeys(session_id)` —— 列出子 Agent 键。

`--session-mirror`（CLI 标志）在流式输出时把对话记录镜像到外部文件，用于调试。

### Hook

可注册的生命周期 hook：

- `PreToolUse`、`PostToolUse` —— 拦截或审计 tool call。
- `SessionStart`、`SessionEnd` —— 初始化与清理。
- `UserPromptSubmit` —— 在模型看到用户输入之前处理它。
- `PreCompact` —— 在上下文压缩之前运行。
- `Stop` —— Agent 退出时清理。
- `Notification` —— 旁路告警。

Hook 是 pro-workflow（第 14 阶段课程参考）和类似系统添加横切行为的方式。

### W3C trace 上下文

调用方上活跃的 OTel span 通过 W3C trace context 头传播到 CLI 子进程。整个多进程 trace 在你的后端显示为一条 trace。

### Claude Managed Agents

托管替代方案（beta header `managed-agents-2026-04-01`）。长时间运行的异步工作、内置 prompt 缓存、内置压缩。用控制权换取托管基础设施。

### 这个模式会在哪里出错

- **子 Agent 过度派生。** 为 100 个微小任务派生 100 个子 Agent。开销占主导。改为批处理。
- **hook 蔓延。** 每个团队都加 hook；启动时间膨胀。每季度审查一次 hook。
- **session 膨胀。** session 不断累积；体积增长。使用 `list_sessions` + 过期策略。

```figure
ae-subagent-isolation
```

## 构建它

`code/main.py` 用标准库实现 SDK 的形态：

- `Tool`、`ToolRegistry`，内置 `read_file`、`write_file`、`list_dir`。
- `Subagent` —— 私有上下文、隔离运行、返回结果。
- `SessionStore` —— 追加、加载、列出、删除、列出子键。
- `Hooks` —— `pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 演示：主 Agent 并行派生 3 个子 Agent（各自隔离），汇总结果，持久化 session。

运行它：

```
python3 code/main.py
```

trace 展示了子 Agent 的上下文隔离（编排器的上下文大小保持有界）、hook 执行和 session 持久化。

## 使用它

- **Claude Agent SDK** 用于想要 Claude Code harness 形态、以 Claude 为先的产品。
- **Claude Managed Agents** 用于托管的长时间异步工作。
- **OpenAI Agents SDK**（第 16 课）用于以 OpenAI 为先的对应方案。
- **LangGraph + 自定义工具** 如果你更想要图形态的状态机。

## 交付它

`outputs/skill-claude-agent-scaffold.md` 搭建一个 Claude Agent SDK 应用，包含子 Agent、hook、session 存储、MCP 服务器挂载和 W3C trace 传播。

## 练习

1. 增加一个子 Agent 派生器，把 20 个任务分成每组 5 个并行子 Agent 的 batch。测量编排器上下文大小与每任务一个子 Agent 的对比。
2. 实现一个 `PreToolUse` hook，对 `write_file` 调用限流（每个 session 每分钟 5 次）。追踪其行为。
3. 把 `list_subkeys` 接上，渲染一棵子 Agent 树。深层嵌套看起来是什么样？
4. 把玩具移植到真正的 `claude-agent-sdk` Python 包。工具注册会发生什么变化？
5. 阅读 Claude Managed Agents 文档。什么时候你会从自托管切换到托管？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Agent SDK | 「把 Claude Code 当库」 | harness 形态：工具、MCP、hook、子 Agent、session 存储 |
| 子 Agent（subagent） | 「子 agent」 | 独立上下文、自有预算；结果向上冒泡 |
| Session 存储 | 「对话数据库」 | 持久化、加载、列出、删除轮次，并级联子 Agent |
| Hook | 「生命周期回调」 | 工具前后、session、prompt 提交、压缩、停止 |
| W3C trace 上下文 | 「跨进程 trace」 | 父 span 传播进 CLI 子进程 |
| Managed Agents | 「托管 harness」 | Anthropic 托管的长时间异步工作 |
| `--session-mirror` | 「对话记录镜像」 | 流式输出时把 session 轮次写入外部文件 |
| MCP 服务器 | 「工具面」 | 挂载到 Agent 的外部工具/资源来源 |

## 延伸阅读

- [Claude Agent SDK 概览](https://platform.claude.com/docs/en/agent-sdk/overview) —— Claude Code 的库形态
- [Anthropic，Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) —— 生产模式
- [Claude Managed Agents 概览](https://platform.claude.com/docs/en/managed-agents/overview) —— 托管替代方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) —— 对应实现
