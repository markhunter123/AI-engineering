# MCP 传输：stdio 与无状态 Streamable HTTP

> 传输只负责承载 MCP 消息，不负责补齐缺失的协议状态。在 `2026-07-28` 中，本地 stdio 和远程 Streamable HTTP 承载的都是自描述请求。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 07 and 08
**Time:** ~65 minutes

## 学习目标

- 本地子进程选择 stdio，网络服务选择 Streamable HTTP。
- 实现现代的单端点、仅 POST 的 Streamable HTTP 约定。
- 在 HTTP 头部中镜像 MCP 版本、方法和名称，并对照 JSON-RPC 请求体进行验证。
- 正确投递请求作用域的 SSE 和长生命周期的 `subscriptions/listen` 流。
- 迁移基于会话的部署和遗留 HTTP+SSE 部署，且不把遗留行为呈现为现代行为。

## 问题

早期 Streamable HTTP 修订版把协议协商与连接和会话行为混在一起。服务器可以签发 `Mcp-Session-Id`，暴露独立的 GET 流，接受 DELETE 来终止会话，并使用 `Last-Event-ID` 续传 SSE。

MCP `2026-07-28` 把这些机制从现代线路上移除了。每个请求都可以落到任意健康的 worker 上，因为协议版本和客户端能力都在请求体中传输。HTTP 头部镜像部分选定字段，用于路由和策略，但服务器会在执行前对照请求体验证这些头部。

结果是更容易扩展，也更容易推理。这也意味着，一个把 2025 年传输当作现行标准来讲解的服务器，教的是一套错误的故障模型和安全模型。

## 概念

### stdio

stdio 绑定用于客户端启动的子进程：

- 客户端向 stdin 每行写入一条 UTF-8 JSON-RPC 消息。
- 服务器向 stdout 每行写入一条 UTF-8 JSON-RPC 消息。
- 服务器把诊断信息写到 stderr。
- 服务器在 stdin EOF 时立即退出。
- 每个现代请求都在 `params._meta` 中携带版本和客户端能力。

该进程可能存活很多次调用，但它不是一个现代协议会话。如果它意外退出，在途请求就会丢失。重启进程、重新发现、重新列举、重新打开订阅，并用新的请求 id 重试安全的操作。

### 2026-07-28 中的 Streamable HTTP

现代服务器暴露一个 MCP 端点，例如 `/mcp`，只接受 POST。

每个 JSON-RPC 请求或通知都是一次新的 HTTP POST。请求体包含一条 JSON-RPC 消息。客户端不向服务器发送 JSON-RPC 响应。

对于请求，服务器返回以下之一：

- 带一条 JSON-RPC 响应的 `Content-Type: application/json`；或
- 带与该请求相关通知的 `Content-Type: text/event-stream`，随后是最终的 JSON-RPC 响应。

对于已接受的通知，服务器返回 `202 Accepted`，无响应体。

客户端同时声明接受两种响应类型：

```http
Accept: application/json, text/event-stream
```

### 仅 POST 就是仅 POST

现代 Streamable HTTP 没有独立的 GET 流，也没有 DELETE 会话端点。

- `GET /mcp` 返回 `405 Method Not Allowed`。
- `DELETE /mcp` 返回 `405 Method Not Allowed`。
- `Mcp-Session-Id` 被忽略，既不会被签发也不会被回显。
- `Last-Event-ID` 被忽略，因为现代流不可续传。

如果请求作用域的流在最终响应之前中断，客户端就丢失了那个在途请求。当重试是安全的时，它可以用新的 JSON-RPC id 发起新请求。它不得尝试续传流。

### Origin 验证

服务器在传入连接上验证 `Origin`，以防止 DNS 重绑定。如果该头部存在且未被显式允许，返回 `403 Forbidden`。非浏览器客户端可以省略 `Origin`，官方传输规则允许这样做。

本地服务器应绑定到 `127.0.0.1`，而不是所有网络接口。网络服务仍然需要对每个请求进行认证和授权。Origin 验证不是认证。

在规范化配置之后使用精确的 origin 匹配。像 `origin.startswith("https://trusted.example")` 这样的前缀检查是不安全的，因为它可能接受攻击者控制的后缀。

### 必需的 HTTP 元数据头部

每个现代 POST 请求都包含：

```http
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes_search
```

头部规则：

- `MCP-Protocol-Version` 是必需的，且必须等于 `params._meta.io.modelcontextprotocol/protocolVersion`。
- `Mcp-Method` 是必需的，且必须等于 JSON-RPC 的 `method`。
- `tools/call`、`resources/read` 和 `prompts/get` 要求提供 `Mcp-Name`。
- `Mcp-Name` 等于 `params.name`；对 `resources/read` 则等于 `params.uri`。
- 头部值区分大小写，尽管头部名称不区分大小写。

不安全或非 ASCII 的 `Mcp-Name` 值使用精确的 UTF-8 Base64 哨兵形式：

```text
=?base64?{Base64EncodedValue}?=
```

服务器在把该值与请求体比较之前先对其进行解码。

镜像头部缺失、畸形或不匹配时，返回 HTTP `400` 和 JSON-RPC 错误码 `-32020`。如果头部与请求体一致地声明了一个服务器不支持的版本，返回 HTTP `400` 和 `-32022`，并附带精确的错误数据，例如 `{"supported":["2026-07-28"],"requested":"2027-01-01"}`。

未知的现代方法返回 HTTP `404` 和 JSON-RPC `-32601`。这个 JSON-RPC 响应体很重要，因为双时代客户端用它来区分现代错误和遗留端点未命中。

### 请求作用域的 SSE

服务器可以为某个长时间运行的请求选择 SSE：

```text
POST tools/call id=41
  <- notifications/progress related to id=41
  <- notifications/progress related to id=41
  <- JSON-RPC response id=41
stream closes
```

服务器不得在这个流上发送独立的 JSON-RPC 请求。Sampling、elicitation 和 roots 交互使用多轮往返请求（Multi Round-Trip Request）结果。关闭响应流即取消该请求。

不要为回放而添加 SSE 事件 id。`Last-Event-ID` 续传不属于现代修订版。

### 长生命周期的变更使用 subscriptions/listen

变更通知使用客户端打开的请求，而不是独立的 GET：

```json
{
  "jsonrpc": "2.0",
  "id": "listen-1",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true,
      "resourceSubscriptions": ["notes://note-1"]
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    }
  }
}
```

POST 响应是一个长生命周期的 SSE 流。它的第一条协议消息是 `notifications/subscriptions/acknowledged`。该确认消息、每一条变更通知以及最终结果都在 `_meta` 中携带 `io.modelcontextprotocol/subscriptionId`，其值等于 listen 请求 id。服务器可以发出 SSE 注释作为保活信号。当流断开时，客户端用新的请求 id 重新发起 `subscriptions/listen`，并重新获取受影响的数据。

`resources/subscribe` 和 `resources/unsubscribe` 属于遗留时代。不要在现代连接上使用它们。

### 显式的应用状态

移除协议会话并不禁止有状态的工作流。服务器可以签发一个不透明的状态句柄，并把它作为普通工具结果返回。客户端在后续调用中把这个句柄作为显式参数传入。

把句柄绑定到已认证的主体，使其不可猜测，为其设置过期时间，并对每次使用进行授权。这样状态就在应用层可见，而不是藏在传输亲和性里。

隐藏的副本状态引发的故障是机械性的：

1. 请求 A 到达副本 1，并在该进程的内存中创建了一个草稿。
2. 响应没有返回草稿句柄，因为实现假设由连接来标识该草稿。
3. 请求 B 是一次全新的 POST，到达副本 2。
4. 副本 2 拥有有效的协议元数据，但无法命名或加载该草稿，于是工作流失败或读到了错误的本地对象。
5. 粘性路由看起来能修复症状，直到一次重启、滚动发布、重新调度或故障转移把下一个请求挪走。

正确的边界有两部分。协议上下文留在每个请求里。持久化的应用状态存放在共享存储中，由服务器签发句柄并返回给客户端。下一次调用提供该句柄，任意副本都能加载同一条记录，授权把该记录绑定到已认证主体和租户。副本内存可以缓存记录，但它不能是正确性所依赖的唯一副本。

按生命周期选择状态机制。请求局部变量可以服务一次调用。短暂的 MRTR 续接可以使用受完整性保护的 `requestState`。草稿或持久化任务则需要显式句柄，外加共享持久化、过期、并发控制和幂等性。这些对象都不是 MCP 协议会话。

### HTTP 双时代兼容

同时支持现代和遗留服务器的客户端会先尝试现代 POST。如果收到 HTTP `400`、`404` 或 `405`，它会检查响应体：

- 可识别的现代 JSON-RPC 错误证明服务器是现代服务器。修正请求，或按服务器公布的版本重试。不要降级。
- 空响应体或无法识别的响应可能意味着这是一个遗留 HTTP+SSE 服务器。只有在这种情况下，才尝试旧的 GET 端点，并期待它的遗留 `endpoint` 事件。

在迁移期间，服务器可以同时支持两个时代：把现代元数据路由到现代的仅 POST 实现，同时为老客户端保留独立的遗留端点。绝不要把遗留的 GET、DELETE、会话 id 或回放行为描述为 `2026-07-28` 的一部分。

```figure
tp-transport-handshake
```

## 使用它

`code/main.py` 用 Python 标准库实现了一个有界的现代 Streamable HTTP 服务器。它验证 Origin 和镜像头部，忽略已移除的会话头部，对普通调用返回 JSON，并演示一个有界的 `subscriptions/listen` SSE 流。

```bash
cd code
python3 main.py --probe
python3 -m unittest discover tests -v
```

该探测检查：

- 无效的 Origin 被拒绝；
- 在没有会话 id 的情况下发现成功；
- `Mcp-Session-Id` 和 `Last-Event-ID` 被忽略；
- 头部不匹配返回 `-32020`；
- 不支持的版本返回 `-32022`，并附带精确的 `supported` 和 `requested` 数据；
- 已接受的无 id 通知返回 HTTP `202`，无响应体；
- GET 和 DELETE 返回 `405`；
- `subscriptions/listen` 是一个 POST 响应流，其确认消息、通知和最终结果都携带自己的订阅 id。

## 交付它

本课交付 `outputs/skill-mcp-transport-migrator.md`。它移除现代协议会话，添加头部与请求体的校验，用 `subscriptions/listen` 替换独立的 GET，并让任何遗留桥接保持可见的分离。

## 练习

1. 从一个 POST 中移除 `Mcp-Method`。确认返回 HTTP `400` 和错误 `-32020`。
2. 发送头部与请求体一致、版本为 `2027-01-01` 的请求。确认返回 HTTP `400`、错误 `-32022`，以及精确数据 `{"supported":["2026-07-28"],"requested":"2027-01-01"}`。
3. 为一个非 ASCII 资源 URI 发送 Base64 哨兵形式的 `Mcp-Name`。确认解码后的值会与 `params.uri` 比较。
4. 在最终响应之前中断这个有界的 listen 流。用新的 JSON-RPC id 重新发起，并重新获取工具。
5. 为 ping 工具添加一个显式工作流句柄。把它绑定到授权主体，且不使用连接亲和性。

## 关键术语

| 术语 | 含义 |
|------|---------|
| stdio | 通过客户端启动的子进程、以换行分隔的 JSON-RPC |
| Streamable HTTP | 单一端点，每条现代消息都是一次新的 POST |
| 请求作用域 SSE | 包含相关通知和最终响应的 POST 响应流 |
| `subscriptions/listen` | 用于已选择加入的变更通知的长生命周期 POST 请求 |
| 头部不匹配 | 镜像头部与请求体不一致时返回 HTTP `400` 和 JSON-RPC `-32020` |
| Origin 验证 | 针对传入连接的 DNS 重绑定防护，不是认证 |
| 显式状态句柄 | 作为普通参数传递的应用令牌，取代隐藏的会话状态 |
| 遗留桥接 | 仅为兼容而保留的独立早期时代行为 |

## 延伸阅读

- [MCP Transport Overview](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP Subscriptions](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [MCP 2026-07-28 Changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
