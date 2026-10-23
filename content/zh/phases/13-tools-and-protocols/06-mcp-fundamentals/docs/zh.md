# MCP 基础：无状态请求与 JSON-RPC

> 现代 MCP 没有握手，也没有协议会话。每个请求都必须携带足够的元数据，才能被独立地理解、授权、路由和重试。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 01 through 05
**Time:** ~55 minutes

## 学习目标

- 区分 MCP 的服务器原语与其客户端侧特性。
- 为 MCP `2026-07-28` 构建合法的 JSON-RPC 2.0 请求与响应。
- 在每个请求中附带协议版本、客户端能力和客户端身份。
- 使用 `server/discover` 并在没有握手的情况下处理 `UnsupportedProtocolVersionError`。
- 追踪一个独立请求，从校验一直到完整结果。

## 问题

一个 MCP server 可能在同一进程或 HTTP worker 上，连续收到来自不同客户端、具有不同能力的两个请求。如果服务器记住了上一个请求声明了什么，它就可能施加错误的权限，或返回错误的线上形状。

MCP `2026-07-28` 移除了这种歧义。协议核心是无状态的。服务器必须根据当前请求本身来决定如何处理它，而不是根据连接历史。

这改变了心智模型。旧顺序是先连接、再握手、然后操作。现代顺序更简单：

1. 客户端发送一个自描述的请求。
2. 服务器校验该请求的版本与能力。
3. 服务器处理该方法。
4. 服务器返回类型化结果或 JSON-RPC 错误。

下一个请求从头重复同样的过程。

## 概念

### 服务器原语

MCP server 暴露三种主要原语：

1. **Tools（工具）** 是模型控制的动作，用 `tools/list` 发现，用 `tools/call` 调用。
2. **Resources（资源）** 是用 URI 寻址的数据，用 `resources/list` 发现，用 `resources/read` 获取。
3. **Prompts（提示）** 是可复用模板，用 `prompts/list` 发现，用 `prompts/get` 渲染。

Roots、sampling 和 logging 保留在 `2026-07-28` schema 中以保持兼容，但已被弃用。新实现应为 roots 使用显式的工具或资源输入，为 sampling 直接使用模型供应商 API，为 logging 使用 stderr 或 OpenTelemetry。Elicitation 仍可通过 Multi Round-Trip Requests 使用：服务器返回一个输入请求，客户端重试原操作。现代服务器永远不会主动发起独立的 JSON-RPC 请求。

### JSON-RPC 信封

MCP 使用 JSON-RPC 2.0：

- 请求：`{jsonrpc, id, method, params}`
- 响应：`{jsonrpc, id, result}` 或 `{jsonrpc, id, error}`
- 通知：`{jsonrpc, method, params}`，带 `id`

请求的 `id` 只关联一个响应。它不创建协议会话。

### 必需的请求元数据

每个现代请求都在 `params` 内携带一个 `_meta` 对象：

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/list",
  "params": {
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

协议版本和客户端能力是必需的。客户端身份是推荐项。它是自报的展示与调试数据，不是安全凭证。

服务器不得从更早的请求、stdio 进程、HTTP 连接或单独的传输头中推断这些值中的任何一个。

### 完整结果与服务器身份

每个成功的现代结果都包含 `resultType`。普通的最终结果使用 `"complete"`。服务器还应在结果元数据中标识自己：

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "resultType": "complete",
    "tools": [],
    "ttlMs": 30000,
    "cacheScope": "public",
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "notes-server",
        "version": "1.0.0"
      }
    }
  }
}
```

`tools/list`、`resources/list`、`prompts/list`、`resources/templates/list`、`resources/read` 和 `server/discover` 是可缓存结果。它们包含 `ttlMs` 和 `cacheScope`。安全的默认值是 `ttlMs: 0` 和 `cacheScope: "private"`。列表项应具有确定性排序，使等价响应产生稳定的缓存键和稳定的模型上下文。

### 无需握手的发现

每个现代服务器都必须实现 `server/discover`。客户端可以在调用其他方法之前先调用它，以获取：

- `supportedVersions`
- 服务器 `capabilities`
- 可选的用法 `instructions`
- 结果 `_meta` 中的服务器身份
- 缓存提示

发现机制很有用，但不是门禁。客户端可以先发送 `tools/list`，因为该请求本身已经携带协议版本和能力。

如果请求的版本不受支持，服务器返回 JSON-RPC 错误码 `-32022`，并附带：

```json
{
  "requested": "2027-01-01",
  "supported": ["2026-07-28"]
}
```

客户端选择一个双方都支持的现代版本，并用新的 JSON-RPC 请求 id 重试。

### 单个请求的生命周期

按以下顺序追踪一个现代请求：

1. 解析一个 JSON-RPC 信封。
2. 确认 `jsonrpc` 为 `"2.0"`、存在 `id`、`method` 是字符串、`params` 是对象。
3. 要求 `params._meta` 中包含版本字符串和能力对象；元数据格式错误或缺失返回 `-32602`。
4. 在 HTTP 边界，把版本、方法和适用的名称头与请求体做比较。即使两个版本值中有一个不受支持，只要不匹配就返回 `-32020`。
5. 在确认相等之后，以 `-32022` 拒绝匹配但不受支持的版本。
6. 检查必需的能力，然后按 `method` 路由并校验方法特有的参数。
7. 在处理器运行之前，对具体操作进行认证和授权。
8. 返回包含服务器身份的完整结果。
9. 忘掉请求作用域的协议元数据。

这个顺序防止两个组件对不同的调用作出解释。网关绝不能授权 `Mcp-Name: notes.read`，而源站却执行 `params.name: notes.delete`。它还把格式错误的输入、头部混淆、版本协商、能力失败、授权失败和处理器失败区分为不同的证据。

关闭 stdin 或 HTTP 响应会结束传输活动。它不会终止协议会话，因为现代 MCP 没有协议会话。

### 显式的旧版兼容性

截至 `2025-11-25` 的版本使用 `initialize`、`notifications/initialized`、连接作用域的能力，以及在更早的 Streamable HTTP 上可选的协议会话。当双时代客户端与旧服务器通信时，这些行为仍然相关。

请把两个时代分开。现代请求由必需的逐请求元数据来识别。旧版连接只能通过文档化的回退路径来选择。不要把 `initialize` 作为 `2026-07-28` 服务器的默认请求发送。

因此，「无状态」的含义与时代相关。在 `2026-07-28` 中，它是协议不变量：每个普通请求都可独立解释，不存在 MCP 会话。在截至 `2025-11-25` 的版本中，初始化和协商出的能力属于某个连接，因此兼容性适配器可以保留那份旧版连接状态。双时代实现不是一个宽松的状态机，而是无状态的现代核心与隔离的旧版适配器并置，并在任一解析器运行之前做出显式选择。

两种含义都不禁止持久化的应用状态。工作流、任务或草稿可以存放在共享存储中，由一个不透明句柄代表。客户端把这个句柄作为普通输入发送，每个副本都会对它的使用进行认证和授权。协议上下文不得泄漏进该存储，以充当被移除会话的替代品。

```figure
mcp-tool-call
```

## 使用它

`code/main.py` 不依赖框架，构建、校验、追踪并分发现代 MCP 消息。运行：

```bash
python3 code/main.py
python3 -m unittest discover code/tests -v
```

在输出中留意三个不变量：

- 每个请求都重复携带自己的 `_meta` 字段。
- 每个成功结果都是 `resultType: "complete"`，并包含服务器身份。
- 列表结果按确定性顺序排列，并带有显式的缓存提示。

## 交付它

本课交付 `outputs/skill-mcp-handshake-tracer.md`。历史文件名保持不变，但该产物现在是针对无状态请求的追踪器。它独立审计每条消息，只有在旧版握手流量确实存在时才加以标记。

## 练习

1. 把某个请求的协议版本改为 `2027-01-01`。确认错误码为 `-32022`，并且数据中公布了支持的版本。
2. 从第二个请求中移除 `io.modelcontextprotocol/clientCapabilities`。确认服务器没有复用第一个请求的能力。
3. 反转内存中的工具注册表。确认 `tools/list` 仍返回相同的确定性顺序。
4. 把 `cacheScope` 从 `public` 改为 `private`。解释在每种情况下哪些授权上下文可以复用该响应。
5. 增加一个可选的 `clientInfo` 省略测试。请求应保持合法，因为客户端身份是推荐项而非必需项。

## 关键术语

| 术语 | 含义 |
|------|---------|
| 无状态协议（Stateless protocol） | 每个请求都提供解释它所需的元数据 |
| 请求元数据 | `params._meta` 中的版本、客户端能力与推荐的客户端身份 |
| `server/discover` | 用于获取版本、能力、指令和身份的强制服务器方法 |
| `resultType` | 每个成功的现代结果上的判别字段 |
| 可缓存结果 | 包含必需 `ttlMs` 和 `cacheScope` 提示的结果 |
| 协议时代 | 现代的逐请求元数据，或旧版的连接作用域初始化 |
| 传输生命周期 | 进程、连接或响应流的生命周期，而非协议会话状态 |
| `-32022` | 不受支持的协议版本错误，带请求版本与支持版本 |

## 延伸阅读

- [MCP Architecture](https://modelcontextprotocol.io/specification/2026-07-28/architecture)
- [MCP Base Protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP 2026-07-28 Changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
