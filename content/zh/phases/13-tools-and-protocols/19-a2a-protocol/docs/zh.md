# A2A —— Agent 到 Agent 协议

> MCP 是 Agent 到工具（agent-to-tool）。A2A（Agent2Agent）是 Agent 到 Agent——一种开放协议，让构建在不同框架上的不透明 Agent 能够协作。它于 2025 年 4 月由 Google 发布，2025 年 6 月捐赠给 Linux Foundation，2026 年 4 月达到 v1.0，拥有 150 多个支持者，包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow。它吸收了 IBM 的 ACP，并增加了 AP2 支付扩展。本课将讲解 Agent Card、Task 生命周期和三种协议绑定，并使用 A2A 1.0.1 的线上名称。

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## 学习目标

- 区分 Agent 到工具（MCP）与 Agent 到 Agent（A2A）的用例。
- 在 `/.well-known/agent-card.json` 发布包含 skills 与 `supportedInterfaces` 元数据的 Agent Card。
- 走通 Task 生命周期：`TASK_STATE_SUBMITTED`、`TASK_STATE_WORKING`、`TASK_STATE_INPUT_REQUIRED`，以及终态 `TASK_STATE_COMPLETED`、`TASK_STATE_FAILED`、`TASK_STATE_CANCELED`、`TASK_STATE_REJECTED`。
- 使用 Message，其中的每个 Part 恰好持有一个 `text`、`raw`、`url` 或 `data` 字段；并用 Artifact 作为输出。

## 问题

一个客服 Agent 需要把报告撰写委托给一个专门的写作 Agent。在 A2A 之前，可选项有：

- 自定义 REST API。可行，但每一对组合都是一次性工程。
- 共享代码库。要求两个 Agent 运行在同一框架上。
- MCP。不合适：MCP 用于调用工具，而不是让两个 Agent 在保持各自不透明内部推理的同时协作。

A2A 填补了这个空白。它把交互建模为一个 Agent 向另一个 Agent 发送 Task，并带有生命周期、消息和制品。被调用 Agent 的内部状态保持不透明——调用方只能看到任务状态转换和最终输出。

A2A 是「让跨框架的 Agent 彼此对话」的协议。它并不取代 MCP；两者是互补关系。

## 概念

### Agent Card

每个兼容 A2A 的 Agent 都在 `/.well-known/agent-card.json` 发布一份卡文档：

```json
{
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "version": "1.2.0",
  "supportedInterfaces": [
    {
      "url": "https://research.example.com/a2a",
      "protocolBinding": "JSONRPC",
      "protocolVersion": "1.0"
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true},
  "securitySchemes": {
    "bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}
  },
  "securityRequirements": [{"schemes": {"bearer": {"list": []}}}],
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/markdown"],
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "tags": ["research", "summarization"],
      "inputModes": ["text/plain", "application/pdf"],
      "outputModes": ["text/markdown"]
    }
  ]
}
```

发现基于 URL：拉取卡文档，选择 `supportedInterfaces` 中第一个 `protocolBinding` 为你的客户端所支持的条目，然后枚举 skills。输入与输出模式都是媒体类型。

### 签名 Agent Card

卡可以携带 `signatures` 数组。每个条目是一个 JWS（RFC 7515），基于该卡的 RFC 8785 规范 JSON（不含 `signatures` 字段）计算。消费方以相同方式对卡做规范化并验证。这可以防止冒充。

### Task 生命周期

```text
TASK_STATE_SUBMITTED
  -> TASK_STATE_WORKING
  -> TASK_STATE_COMPLETED | TASK_STATE_FAILED | TASK_STATE_CANCELED | TASK_STATE_REJECTED

TASK_STATE_WORKING
  -> TASK_STATE_INPUT_REQUIRED
  -> TASK_STATE_WORKING (the client sends a message with the same taskId)
```

客户端用 `SendMessage` 发起，服务器创建 Task。被调用 Agent 在状态之间转换；客户端用 `GetTask` 轮询，或通过 `SendStreamingMessage` 与 `SubscribeToTask` 以 SSE 流式接收。流携带 `statusUpdate` 与 `artifactUpdate` 事件，并在任务进入终态时关闭。没有 `final` 标志。

### Message 与 Part

一条消息包含 `messageId`、`role`（`ROLE_USER` 或 `ROLE_AGENT`）以及一个或多个 Part。每个 Part 恰好持有一个内容字段，字段名就是类型。没有 `kind` 字段。

- `text`：纯文本内容。
- `raw`：文件字节，在 JSON 中以 base64 表示，通常带有 `filename` 和 `mediaType`。
- `url`：指向文件内容的链接。
- `data`：结构化 JSON 载荷（给被调用 Agent 的结构化输入）。

示例：

```json
{
  "messageId": "msg-001",
  "role": "ROLE_USER",
  "parts": [
    {"text": "Summarize this paper."},
    {"raw": "...", "filename": "paper.pdf", "mediaType": "application/pdf"},
    {"data": {"targetLength": "3 paragraphs"}, "mediaType": "application/json"}
  ]
}
```

### Artifact

输出是 Artifact，而不是原始字符串。Artifact 是具名、带类型的输出：

```json
{
  "artifactId": "art-001",
  "name": "summary",
  "parts": [{"text": "...", "mediaType": "text/markdown"}]
}
```

Artifact 可以按块流式传输。每个 `artifactUpdate` 事件携带 artifact 以及 `append` 和 `lastChunk`。调用方负责累积。

### 三种协议绑定

1. **基于 HTTP 的 JSON-RPC 2.0**（`JSONRPC`）。请求用 POST，流式用 SSE。方法名为 PascalCase：`SendMessage`、`SendStreamingMessage`、`GetTask`、`ListTasks`、`CancelTask`、`SubscribeToTask`、`CreateTaskPushNotificationConfig`、`GetTaskPushNotificationConfig`、`ListTaskPushNotificationConfigs`、`DeleteTaskPushNotificationConfig` 和 `GetExtendedAgentCard`。
2. **gRPC**（`GRPC`）。适用于 gRPC 为原生协议的企业环境。方法名相同。
3. **HTTP+JSON/REST**（`HTTP+JSON`）。使用资源 URL，例如 `POST /message:send` 和 `GET /tasks/{id}`。

三种绑定承载相同的数据模型。每个 `supportedInterfaces` 条目指明一种绑定及其 `protocolVersion`。客户端在每个请求上都发送 `A2A-Version: 1.0` 头，因为服务器会把缺少该头的请求当作 0.3 版本读取。

```http
POST /a2a HTTP/1.1
Host: research.example.com
Content-Type: application/json
A2A-Version: 1.0

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "SendMessage",
  "params": {
    "message": {
      "messageId": "msg-001",
      "role": "ROLE_USER",
      "parts": [{"text": "Summarize this paper."}]
    }
  }
}
```

### 不透明性保持

一个关键设计原则：被调用 Agent 的内部状态是不透明的。调用方只能看到任务状态和制品。被调用 Agent 的思维链、tool call、子 Agent 委托——全部不可见。这与 MCP 不同，在 MCP 中 tool call 是透明的。

理由：A2A 让竞争对手在不必暴露内部实现的情况下协作。A2A 可以是「调用这个客服 Agent」，而调用方无需了解该 Agent 如何实现服务。

### 时间线

- **2025-04-09。** Google 宣布 A2A。
- **2025-06-23。** 捐赠给 Linux Foundation。
- **2025-08。** 吸收 IBM 的 ACP。
- **2025-09。** AP2 扩展（Agent Payments）发布。
- **2026-04。** v1.0 发布，拥有 150 多个支持组织。

### 与 MCP 的关系

| 维度 | MCP | A2A |
|-----------|-----|-----|
| 用例 | Agent 到工具 | Agent 到 Agent |
| 不透明度 | 透明的 tool call | 不透明的内部推理 |
| 典型调用方 | Agent 运行时 | 另一个 Agent |
| 状态 | Tool call 结果 | 带生命周期的 Task |
| 授权 | OAuth 2.1（第 13 阶段 · 16） | Agent Card 的 `securitySchemes` + `securityRequirements` |
| 传输 | Stdio / Streamable HTTP | JSON-RPC / gRPC / HTTP+JSON |

想调用特定工具时用 MCP。想把整个任务委托给另一个 Agent 时用 A2A。许多生产系统两者并用：Agent 在工具层用 MCP，在协作层用 A2A。

```figure
a2a-task-lifecycle
```

## 使用它

`code/main.py` 实现了一个最小 A2A 测试装置：writer Agent 发布它的卡，research Agent 向它发送一个 `SendMessage` 请求，其中包含一个 PDF part 和一段文本指令，任务依次经过 `TASK_STATE_WORKING` → `TASK_STATE_INPUT_REQUIRED` → `TASK_STATE_WORKING` → `TASK_STATE_COMPLETED`，最后返回一个文本 artifact。全部使用 stdlib；采用内存传输，以聚焦消息形态。

重点观察：

- Agent Card 的 JSON 结构。
- 服务端任务 id 分配与状态转换。
- Part 以存在哪个内容字段来定类型。
- 任务中途的 `TASK_STATE_INPUT_REQUIRED` 分支。
- 完成时返回 Artifact。

## 交付它

本课产出 `outputs/skill-a2a-agent-spec.md`。给定一个应可被其他 Agent 调用的新 Agent，该技能会生成 Agent Card JSON、skills schema 和端点蓝图。

## 练习

1. 运行 `code/main.py`。追踪完整的 Task 生命周期，包括被调用 Agent 请求澄清时的 `TASK_STATE_INPUT_REQUIRED` 暂停点。

2. 添加签名的 Agent Card。在 `signatures` 中放入一个 JWS 条目，`alg` 设为 `HS256`，对不含 `signatures` 字段的卡的规范 JSON 签名。编写验证器，并确认它在卡被篡改时失败。

3. 用 `SendStreamingMessage` 实现任务流式传输：writer Agent 发出 `task`、三个 `artifactUpdate` 块，以及一个带 `TASK_STATE_COMPLETED` 的 `statusUpdate`，然后关闭流。调用方负责累积这些块。

4. 设计一个封装 MCP 服务器的 A2A Agent。把每个 MCP 工具映射为一个 A2A skill。说明其中的权衡——损失了哪些不透明性？

5. 阅读 A2A v1.0 公告，找出截至 2026 年 4 月尚无任何框架实现的那个特性。（提示：它与多跳任务委托有关。）

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| A2A | 「Agent 到 Agent 协议」 | 用于不透明 Agent 协作的开放协议 |
| Agent Card | 「`/.well-known/agent-card.json`」 | 描述 Agent skills 与 `supportedInterfaces` 的已发布元数据 |
| Skill | 「可调用单元」 | Agent 支持的具名操作（类比 MCP 工具） |
| Task | 「委托单元」 | 带生命周期和最终 artifact 的工作项 |
| Message | 「任务输入」 | 携带 Part（`text`、`raw`、`url`、`data`） |
| Part | 「带类型的块」 | 恰好是 `text` / `raw` / `url` / `data` 之一，外加可选的 `mediaType`；没有 `kind` 字段 |
| Artifact | 「任务输出」 | 完成时返回的具名、带类型输出 |
| AP2 | 「Agent Payments Protocol」 | 构建在 A2A 之上的支付扩展；卡签名是 A2A 核心（`signatures`） |
| Opacity | 「黑盒协作」 | 被调用 Agent 的内部实现对调用方隐藏 |
| `TASK_STATE_INPUT_REQUIRED` | 「任务暂停」 | Agent 需要更多信息时的中断状态 |

## 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A 规范权威来源
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — 参考实现与 SDK
- [A2A v1.0.1 发布](https://github.com/a2aproject/A2A/tree/v1.0.1)：本课遵循的带标签 `docs/specification.md` 与规范性 `specification/a2a.proto`
- [Linux Foundation — A2A 启动新闻稿](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025 年 6 月的治理权转移
- [Google Cloud — A2A 协议升级](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — 路线图与合作伙伴势头
- [Google Dev — A2A 1.0 里程碑](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 发布说明与向后兼容指导
