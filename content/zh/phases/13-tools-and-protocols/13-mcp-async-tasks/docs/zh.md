# MCP Tasks 扩展：无状态核心上的持久化工作

> 无状态 MCP 并不意味着每个操作都必须在一个请求内完成。官方 Tasks 扩展为长时间运行的工作提供了显式的持久句柄。服务器可以从 `tools/call` 返回该句柄，任何实例都能响应 `tasks/get`，客户端输入通过 `tasks/update` 送达，无需恢复协议会话。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 11 (stateless MRTR), Phase 13 · 12 (elicitation)
**Time:** ~90 minutes

## 学习目标

- 区分无状态的协议传输与持久的应用任务状态。
- 在每请求能力和 `server/discover` 中协商 `io.modelcontextprotocol/tasks` 扩展。
- 仅在持久化创建完成后，才返回服务器主导的、带 `resultType: "task"` 的 `CreateTaskResult`。
- 用 `tasks/get` 轮询，用 `tasks/update` 提供任务输入，用 `tasks/cancel` 请求协作式取消。
- 抛弃旧的 `tasks/status`、`tasks/result` 和 `tasks/list` 假设。
- 通过 POST 响应上的 SSE 流，用 `subscriptions/listen` 订阅可选的任务通知。
- 正确建模任务过期、重启恢复、输入键去重和执行错误。

## 为什么 Tasks 是扩展

Tasks 最早在 2025-11-25 作为实验性核心特性出现。2026 年 7 月的重新设计将其移入官方 `io.modelcontextprotocol/tasks` 扩展，这样客户端和服务器可以按需选择这套额外生命周期，而不必为所有人扩展核心协议。

尽管该扩展规范已是 Tasks 当前的官方归属，它仍处于草案阶段。请固定 SDK 支持的扩展版本，运行一致性场景，并把线路适配器与你的 worker 和存储领域隔离开。

当操作具备以下一项或多项属性时，使用 task：

- 它可能超出普通请求的超时时间。
- worker 队列或外部任务系统已经负责执行。
- 客户端需要在自身重启后恢复。
- 操作在执行期间为等待用户或模型输入而暂停。
- 取消和持久的结果获取是产品需求。

不要为廉价的确定性查询创建 task。句柄、持久化、轮询、过期和取消都是实实在在的复杂度。

## 无状态核心，有状态应用

MCP 2026-07-28 移除了 `initialize`、`notifications/initialized`、协议会话和 `Mcp-Session-Id`。这并不禁止有状态产品。

任务 id 是显式的应用状态：

- 服务器在返回它之前先将其持久化。
- 客户端可以保存它，并在重启后继续轮询。
- 该 id 可以路由到共享同一持久存储的任意副本。
- 每个任务方法都会检查授权。
- 过期与删除由任务字段定义，而不是传输层生命周期。

从运维角度看，这与挂在连接上的隐藏状态完全不同。

保持四种生命周期彼此分离：

| 状态 | 生命周期 | 归属位置 |
|---|---|---|
| 协议元数据 | 单个请求 | `params._meta`，每次调用都重新校验 |
| 传输层工作 | 单个 stdio 请求或 HTTP 响应 | 带有限截止时间的工作中协调器 |
| MRTR 续行 | 一次重试序列 | 受完整性保护的 `requestState`，需要时加上重放控制 |
| 持久任务 | 跨请求、副本、重启和重连 | 以已授权 `taskId` 为键的共享应用存储 |

把任务记录搬进进程内存并不会让 MCP 变成有状态，只会让应用变得不可靠。协议仍然无状态，但后续路由到另一副本的 `tasks/get` 无法找回该记录。先持久化再返回句柄，然后让每个任务方法都在租户与主体检查下解析到同一条共享记录。

## 能力协商

客户端在每个合规请求上声明支持：

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "extensions": {
        "io.modelcontextprotocol/tasks": {}
      }
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "lesson-client",
      "version": "1.0.0"
    }
  }
}
```

服务器从 `server/discover` 返回确切的 `supportedVersions`、能力、`ttlMs` 和 `cacheScope`，并在 capabilities 下给出同一个扩展。由于它声明了 tools，它还必须实现强制的 `tools/list`。该结果返回确定性的 `generate_report` 描述符、合法的对象型 `inputSchema`、`resultType: "complete"`、服务器身份元数据和公共缓存提示。

未声明该扩展的客户端调用任务方法时，返回 `-32021`，Missing Required Client Capability，并把 `data.requiredCapabilities` 设为 `{"extensions":{"io.modelcontextprotocol/tasks":{}}}`。不支持的协议字符串返回 `-32022`，其中携带确切的 `supported` 和 `requested` 数据；版本缺失或不是字符串则返回 `-32602`。

不带 JSON-RPC `id` 的信封是通知。接收方可以处理它，但不会发出任何 JSON-RPC 结果或错误。对于已接受的通知，Streamable HTTP 适配器返回 `202 Accepted` 且无响应体。

目前只有 `tools/call` 支持任务增强执行。设计内部抽象时，应让未来的请求类型无需重写存储。

## 服务器主导的任务创建

旧的客户端标志 `params._meta.task.required` 已被移除。客户端声明扩展支持，随后由服务器决定某次 `tools/call` 是否变成任务。

请求：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "generate_report",
    "arguments": {"size": "large"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

响应：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "task",
    "taskId": "tsk_786512e29e0d",
    "status": "working",
    "statusMessage": "Preparing report outline.",
    "createdAt": "2026-08-21T10:30:00Z",
    "lastUpdatedAt": "2026-08-21T10:30:00Z",
    "ttlMs": 900000,
    "pollIntervalMs": 1000
  }
}
```

在针对该 id 的 `tasks/get` 能够解析之前，服务器不得返回这个句柄。在最终一致存储中，要等到读取可见后再作答。否则客户端可能拿到一个看似合法的 id，却立刻得到 "not found"。

任务响应之所以说“未经请求”，是因为客户端并未请求 task 模式。但它并非未经协商：当前请求仍然必须声明该扩展。

## 任务的结构

每个任务都携带：

- `taskId`：服务器生成的稳定标识符；
- `status`：`working`、`input_required`、`completed`、`cancelled` 或 `failed`；
- `createdAt` 和 `lastUpdatedAt`：ISO 8601 时间戳；
- `ttlMs`：从创建起算的过期时长，或 `null` 表示未声明上限；
- 可选的 `pollIntervalMs`：服务器当前建议的最小轮询间隔；
- 可选的 `statusMessage`：面向用户或模型的上下文。

与状态相关的字段仅在相关时出现：

- `input_required` 包含 `inputRequests`。
- `completed` 包含原始请求的 `result` 形状。
- `failed` 包含 JSON-RPC `error` 对象。

客户端应遵守 `pollIntervalMs`。服务器可以对更激进的轮询限流，并可能在任务生命周期内调整该间隔。

## 用 `tasks/get` 轮询

客户端请求当前快照：

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/get
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tasks/get",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

`tasks/get` 本身已经完成，因此其结果始终是 `resultType: "complete"`。嵌套的任务仍然可能处于 `status: "working"` 或 `status: "input_required"`。

这个区分能避免常见的解析器 bug：

```text
result.resultType = complete    means the tasks/get RPC finished
result.status = working        means the represented job is still running
```

没有 `tasks/result` 调用。任务完成时，下一次 `tasks/get` 响应会把原始的 `CallToolResult` 内联在 `result` 下：

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "completed",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:34:12Z",
  "ttlMs": 900000,
  "result": {
    "resultType": "complete",
    "content": [
      {"type": "text", "text": "Generated large report with approved outline."}
    ],
    "structuredContent": {"size": "large", "approved": true},
    "isError": false,
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "tasks-demo",
        "version": "1.0.0"
      }
    }
  },
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "tasks-demo",
      "version": "1.0.0"
    }
  }
}
```

外层 `resultType` 表示 `tasks/get` RPC 已完成。嵌套的 `result.resultType` 表示原始工具调用已完成。这个嵌套判别字段是必需的。嵌套的 `CallToolResult` 还应携带自己的 `io.modelcontextprotocol/serverInfo`；本课将其包含在内，而不是存储无类型负载。

没有 `tasks/list`。无会话的服务器无法安全推断哪些任务属于某个连接范围内的列表。需要历史记录的应用应暴露一个带显式过滤器和所有权规则的、经过授权的领域工具。

## 任务执行期间的输入

任务输入与核心 MRTR 看起来相似，但使用不同的续行机制。

### 任务创建前需要输入

从原始的 `tools/call` 返回核心的 `resultType: "input_required"`。客户端满足该请求并重试原始调用。只有这些同步 MRTR 轮次结束后，才创建任务。

### 任务创建后需要输入

把任务置为 `input_required`。`tasks/get` 暴露未完成的 `inputRequests`，客户端通过 `tasks/update` 发送响应。客户端不重试原始的 `tools/call`。

快照：

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "input_required",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:31:00Z",
  "ttlMs": 900000,
  "inputRequests": {
    "approve_outline": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Approve the generated report outline?",
        "requestedSchema": {
          "type": "object",
          "properties": {"approved": {"type": "boolean"}},
          "required": ["approved"]
        }
      }
    }
  }
}
```

更新：

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/update
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tasks/update",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "inputResponses": {
      "approve_outline": {
        "action": "accept",
        "content": {"approved": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

成功响应是一个空确认加 `resultType: "complete"`。状态变更可能是最终一致的，因此客户端应继续轮询或监听。

每个 `inputRequests` 键在整个任务生命周期内必须唯一。重复的 `tasks/get` 快照可能显示同一个未完成的键；客户端在 UI 上去重，服务器则忽略针对未知、已被取代或已满足键的响应。部分更新可能让任务保持 `input_required`，直到所有必需键都被回答。

## 取消是协作式的

`tasks/cancel` 表达意图并返回空的 complete 确认。该确认并不保证 worker 已经停止。工作可能先完成、忽略取消，或在之后转换状态。

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/cancel
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "tasks/cancel",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

对于三个任务方法，`Mcp-Name` 都镜像 `params.taskId`。它不重复 JSON-RPC 方法名。`code/main.py` 把这条规则集中在 `make_http_request` 中。

本课的 worker 会立即响应取消，使重复调用幂等。生产客户端仍须把取消视为协作式，而不要从确认推断最终任务状态。

不要用 `notifications/cancelled` 取消任务。该通知属于请求取消，而不是持久 Tasks。

这一区分在路由边界上很重要。请求取消针对一个在途的 JSON-RPC 操作或其请求范围的 HTTP 响应。如果 `tools/call` 已经返回 `resultType: "task"`，该请求就已完成，关闭其传输无法命名或停止持久作业。`tasks/cancel` 是一个新的、经过授权的 RPC。它携带 `params.taskId`，在 `Mcp-Name` 中镜像该 id，解析任务所属后端，记录协作取消意图，并返回一个不宣称 worker 已停止的确认。

因此网关必须把请求协调器和任务路由放在不同的表中。请求表可以在响应结束后消失。任务路由必须存活到终态和保留期结束。[第 29 课：MCP 可靠性、取消与流控](../../29-mcp-reliability-cancellation-and-flow-control/docs/en.md) 为两条路径构建竞态、超时、幂等、背压和重试规则。

## 可选通知

轮询是基线。想要推送更新的客户端发送带任务 id 的 `subscriptions/listen`。对 Streamable HTTP 而言，这是一个 POST，其响应是请求范围的 SSE 流。没有独立的 GET 事件流，也没有需要保活的协议会话。

服务器用 `notifications/subscriptions/acknowledged` 确认接受的 id，随后可以通过 `notifications/tasks` 发送完整快照。该确认和每一条任务通知都在 `_meta` 中携带 `io.modelcontextprotocol/subscriptionId`，其值等于 `subscriptions/listen` 的请求 id。除此之外，每条任务通知等价于当时 `tasks/get` 会返回的内容。

客户端仍必须声明 Tasks 扩展。它们应基于持久任务 id 重连并恢复，而不是依赖事件重放或 `Last-Event-ID`。

## 失败语义

正确使用两层错误。

### 协议错误

无效的方法参数或未知任务 id 返回 JSON-RPC 错误，通常是 `-32602`。缺少扩展支持返回 `-32021`，并附上所需能力对象。

### 任务执行结果

- 带 `isError: true` 的正常工具结果仍是一个 `completed` 任务，因为工具调用产生了其定义内的结果。
- 延迟执行期间发生 JSON-RPC 错误会使任务变为 `failed`，并把该 JSON-RPC 错误存到 `error` 下。
- 用户拒绝可能产生 `cancelled`、一个已完成的拒绝结果，或其他领域特定的安全结果。请记录你的选择。

## 持久性、过期与所有权

至少持久化：任务 id、状态、时间戳、ttl、轮询间隔、原始操作所有权、结果或错误、未完成的输入请求，以及所有已发出的输入键。

存储键必须包含或解析出权威的租户和主体。知道任务 id 不得因此获得访问权。在每次 `tasks/get`、`tasks/update`、`tasks/cancel` 和订阅时都检查所有权。

`ttlMs` 从创建起计量，且可能变化。当任务不再产生可观察更新时，客户端可将其视为兜底。服务器可能让任务失败，随后删除过期任务。不要把它描述成“完成后仍保留结果那么久”的承诺。

使用原子写入或事务。本课写入临时文件并原子重命名。多副本服务应使用共享持久存储，加上 worker 租约或等价的并发控制。

```figure
tp-task-lifecycle
```

## 构建它

`code/main.py` 实现了一个确定性的任务服务：

- `server/discover` 返回 `supportedVersions`、缓存提示和 Tasks 扩展。
- `tools/list` 返回确定性的、可缓存的 `generate_report` 描述符，及其合法的输入 schema。
- `tools/call` 在返回 `resultType: "task"` 之前创建并持久化任务。
- 新的服务实例会重新加载同一个任务，演示重启恢复。
- `tasks/get` 返回完整的任务快照。
- worker 从 `working` 进入 `input_required`。
- `tasks/update` 接受表单响应，并返回空的 complete 确认。
- worker 存储带自身 `resultType` 和服务器身份的嵌套 `CallToolResult`，然后转换到 `completed`。
- 在本实现中 `tasks/cancel` 是幂等的。
- HTTP 构建器为 `tasks/get`、`tasks/update` 和 `tasks/cancel` 把 `Mcp-Name` 设为 `params.taskId`。
- 通知辅助函数使用 `notifications/subscriptions/acknowledged` 和 `notifications/tasks`，两者都带上监听请求 id。
- 无 id 的通知不产生 JSON-RPC 响应。

worker 显式推进状态，而不是在后台线程里睡眠。这样每次状态转换都是确定性的，也让协议示例与队列机制保持分离。

## 使用它

在仓库根目录下：

```bash
cd phases/13-tools-and-protocols/13-mcp-async-tasks/code
python3 main.py
python3 -m unittest discover tests -v
```

预期的结果序列：

```text
id=0 resultType=complete status=ack
id=1 resultType=task status=working
id=2 resultType=complete status=working
id=3 resultType=complete status=input_required
id=4 resultType=complete status=ack
id=5 resultType=complete status=completed
```

同时验证在现代服务中，`tasks/status`、`tasks/result` 和 `tasks/list` 返回 method-not-found。
验证 `tools/list` 是确定性的，且每个当前 HTTP 任务方法都通过 `Mcp-Name` 镜像其任务 id。

## 交付它

`outputs/skill-task-store-designer.md` 现在产出感知扩展的设计：能力协商、返回前持久化、当前方法、输入更新流程、所有权、过期、取消、订阅，以及从已移除的实验性方法迁移。

## 练习

1. 增加第二个未完成的输入键。发送部分 `tasks/update`，证明任务在两个键都被回答前保持 `input_required`。
2. 在存储中加入租户所有权，并拒绝由错误认证主体出示的合法任务 id。
3. 加入带过期的 worker 租约。证明两个服务实例不能并发完成同一个任务。
4. 为 `subscriptions/listen` 实现 POST 响应 SSE 适配器。不要添加 GET、`Last-Event-ID` 或会话头。
5. 加入过期清理。区分过期任务与格式错误的任务 id，同时不泄露跨租户的存在性。

## 关键术语

| 术语 | 在当前扩展中的含义 |
|------|----------------------------------|
| Tasks extension | 用于持久异步工作的可选 `io.modelcontextprotocol/tasks` 能力 |
| `CreateTaskResult` | 对合规请求做出的、服务器主导的 `resultType: "task"` 响应 |
| `tasks/get` | 轮询完整的当前任务快照，包括终态结果或待处理输入 |
| `tasks/update` | 提交对任务未完成 `inputRequests` 的响应 |
| `tasks/cancel` | 确认协作式取消意图 |
| `input_required` | 表示客户端输入尚未完成的任务状态 |
| `pollIntervalMs` | 服务器建议的下一次轮询前最小延迟 |
| `ttlMs` | 从任务创建起算的过期时长 |
| Durable-before-return | 任务 id 必须在句柄发出前可解析的规则 |
| `notifications/tasks` | 在已订阅 SSE 响应上投递的可选完整任务快照 |

## 旧版兼容性

2025-11-25 的实验性接口使用客户端请求的任务增强、`tasks/status`、`tasks/result` 和可选的 `tasks/list`。只在锁定的旧版适配器内保留这些名称。当前客户端使用扩展能力，接受服务器主导的句柄，轮询 `tasks/get`，用 `tasks/update` 提供输入，并从任务快照读取最终结果。

## 延伸阅读

- [官方 MCP Tasks 扩展](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
- [MCP 2026-07-28 多轮往返请求](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
