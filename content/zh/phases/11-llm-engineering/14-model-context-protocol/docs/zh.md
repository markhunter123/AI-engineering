# 模型上下文协议（MCP）

> MCP 为 AI 宿主提供一套协议，用来发现和调用工具、资源与 prompt。2026-07-28 修订版让该协议变成无状态的：能力与版本上下文随每个请求一起传输，而不是放在与连接绑定的握手中。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 03 (Structured Outputs)
**Time:** ~75 minutes

## 学习目标

- 区分 MCP 宿主（host）、客户端（client）、服务器（server）、传输层（transport）和服务端原语（primitive）。
- 构建一个符合 MCP 2026-07-28 元数据要求的 JSON-RPC 请求。
- 使用 `server/discover` 查看版本、身份和能力。
- 从工具、资源和 prompt 返回带类型、可感知缓存的结果。
- 解释现代无状态 MCP 如何与握手时代的服务器互操作。
- 为服务器选择安全的状态、传输和审批边界。

## 问题

你的应用需要一次数据库查询、一个日历操作和一个文件读取器。没有共享协议时，每一个 AI 宿主都得为这些相同能力单独编写发现、调用、错误、传输和授权胶水代码。

MCP 缩小了这张集成矩阵。服务器发布一套标准的 JSON-RPC 表面。符合规范的客户端可以发现这个表面，把它呈现给模型或用户，调用它，并解释结果，无需服务器专用的适配器。

有一条重要边界容易被忽略。MCP 标准化的是通信。它不决定模型该调用哪个工具，不让不可信内容变安全，也不会把无状态请求变成持久的应用状态。这些决策仍然由你的宿主和服务器负责。

## 概念

![MCP 宿主、无状态请求与服务器原语](../assets/mcp-architecture.svg)

### 三种服务端原语

1. **工具（Tools）** 是可调用的动作。每个工具都有名称、描述、JSON Schema 输入和处理函数。
2. **资源（Resources）** 是带名字、以 URI 寻址的内容，客户端可以读取。
3. **Prompt** 是可复用的模板，宿主可以将其呈现给用户。

宿主是 AI 应用。宿主内部的 MCP 客户端与一个服务器通信。传输层在两者之间传递 JSON-RPC 消息。

### 无状态请求取代握手

MCP 2026-07-28 移除了 `initialize` 和 `notifications/initialized`，也移除了协议级会话。每个请求都在 `params._meta` 中携带解释它所需的上下文：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

协议版本和客户端能力是必填项，客户端身份是建议项。缺失 `_meta`、缺少必填字段，或必填字段类型错误，都属于格式错误，返回 Invalid Params（`-32602`）。格式正确但服务器不支持的版本字符串返回 `UnsupportedProtocolVersionError`（`-32022`）。服务器可以在不恢复先前协商记录的情况下处理一个合法请求。

无状态并不意味着应用永远不能维护状态。它意味着状态不藏在 MCP 连接或 `Mcp-Session-Id` 背后。如果某个工作流需要连续性，服务器签发一个不透明句柄，客户端在后续调用中把它作为普通工具参数传入。授权仍然必须在每次请求时检查。

### 发现与版本选择

每个现代服务器都实现 `server/discover`。结果中会公布支持的版本、能力和服务器身份：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "complete",
    "supportedVersions": ["2026-07-28"],
    "capabilities": {
      "tools": {},
      "resources": {},
      "prompts": {}
    },
    "ttlMs": 3600000,
    "cacheScope": "public",
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "demo-server",
        "version": "1.0.0"
      }
    }
  }
}
```

客户端也可以直接调用其他方法并处理版本错误，但发现机制让能力展示和版本选择变得显式。不支持的版本返回 `UnsupportedProtocolVersionError`，错误码为 `-32022`。其数据包含 `supported`（服务器支持的修订版数组）和 `requested`（被拒绝的修订版）。

在 stdio 上，双时代客户端先用 `server/discover` 探测。发现结果，或可识别的现代错误（例如 `UnsupportedProtocolVersionError`），表明这是现代服务器。任何无法识别为现代的错误或超时，都允许回退到 2025-11-25 的 `initialize` 流程。旧版行为属于兼容代码，不是现代默认路径。

### 结果是显式的

2026-07-28 的每个核心结果都带有 `resultType`：

- `complete` 表示操作已完成。
- `input_required` 表示服务器需要经由 Multi Round-Trip Requests 模式再进行一轮往返。核心服务器只可能从 `tools/call`、`resources/read` 或 `prompts/get` 返回它。

客户端必须把省略 `resultType` 的旧版结果视为 complete。

服务器应在每个结果的 `_meta` 中包含 `io.modelcontextprotocol/serverInfo`。这个身份是自报的，仅用于展示、日志和调试，不用于安全决策。

列表和读取结果还携带 `ttlMs` 和 `cacheScope`。确定性的 `tools/list` 顺序加上新鲜度提示，让客户端能够安全地缓存发现结果，并提升 prompt 缓存的稳定性。`cacheScope: public` 允许共享缓存；`private` 把复用限制在调用上下文内。

### 线路格式与传输

MCP 使用 JSON-RPC 2.0，运行在 stdio 或 Streamable HTTP 之上。

- 请求包含 `jsonrpc`、`id`、`method` 和 `params`。
- 响应包含匹配的 `id`，以及 `result` 或 `error` 之一。
- 通知没有 `id`，也不期待响应。

现代 Streamable HTTP 只暴露一个接受 POST 的端点。每条 JSON-RPC 消息各占一个 POST。请求型 POST 收到的是单个 JSON 对象，或一条以最终响应结尾、作用域限定在该请求内的 Server-Sent Events 流。被接受的 notification POST 返回 HTTP 202，没有响应体；本核心修订版没有定义 Streamable HTTP 上的客户端到服务器通知。

2026-07-28 中没有独立的 MCP GET 流、DELETE 会话端点、`Mcp-Session-Id` 或 `Last-Event-ID` 重放。长生命周期的变更通知使用 `subscriptions/listen` POST，其响应保持打开，作为 SSE 流。

### 无需服务器发起请求的客户端输入

旧修订版允许服务器通过流发送 `sampling/createMessage`、`roots/list` 或 `elicitation/create` 之类的请求。当前协议改用 Multi Round-Trip Requests。符合条件的工具调用、资源读取或 prompt 获取会返回 `resultType: input_required`，并至少带有 `inputRequests` 或 `requestState` 之一。客户端收集所有被请求的输入，用新的 JSON-RPC ID 和相应的 `inputResponses` 重试原方法，并在提供了 `requestState` 时原样回传。如果没有 `inputRequests`，重试时省略 `inputResponses`。

Roots、Sampling 和 Logging 仍然可用，但已废弃，因此新实现不应采用它们。现有的 Roots 或 Sampling 请求放在 MRTR 的 `inputRequests` 中传递，绝不能作为独立的服务器到客户端 JSON-RPC 请求。优先使用显式的文件或目录参数、资源 URI、服务器配置，以及直接的模型提供商集成。stdio 诊断信息写入 stderr，生产遥测使用 OpenTelemetry。

```figure
mcp-nxm-collapse
```

## 构建它

### 步骤 1：注册服务器表面

尽管请求契约变了，注册依然简单：

```python
server = MCPServer("demo-server")

@server.tool(
    "add",
    "Add two integers.",
    {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"}
        },
        "required": ["a", "b"]
    }
)
def add(a: int, b: int) -> dict:
    return {"sum": a + b}
```

`code/main.py` 中交付的实现还注册了一个资源和一个 prompt。它刻意只使用标准库，让你看清每一个信封（envelope），而不是把协议交给 SDK 处理。

### 步骤 2：为每个请求附加元数据

```python
def request(method, params=None):
    body_params = dict(params or {})
    body_params["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {
            "name": "demo-client",
            "version": "1.0.0"
        }
    }
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": body_params
    }
```

不要把这份元数据只缓存在连接对象里。服务器会在每次请求时校验它。

### 步骤 3：可选，在列出之前先发现

先调用 `server/discover`，选择一个受支持的版本，再调用 `tools/list`。如果你已经知道版本并能处理 `-32022`，直接调用 `tools/list` 也合法。

演示程序按名称顺序返回工具列表，并附带 `ttlMs`、`cacheScope`、`resultType` 和服务器身份。工具调用返回 complete、不可缓存的结果，因为其输出可能取决于当前状态。

### 步骤 4：把同一个请求映射到 HTTP

远程 `tools/call` POST 包含与 JSON-RPC 请求体对应的头部：

```http
POST /mcp HTTP/1.1
Content-Type: application/json
Accept: application/json, text/event-stream
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: add
```

`MCP-Protocol-Version` 头部必须与 `_meta` 中的版本一致。`Mcp-Method` 在每个 JSON-RPC 请求上都是必填的，且必须与 `method` 一致。`Mcp-Name` 仅在 `tools/call`、`resources/read` 和 `prompts/get` 上必填，且必须与工具名、资源 URI 或 prompt 名匹配。缺少必填头部或不匹配时，返回 HTTP 400 和 `HeaderMismatch` 错误码 `-32020`。

### 步骤 5：在协议状态之外落实安全

- 对每个 HTTP 请求校验授权和受众（audience）。
- 把本地服务器绑定到 localhost，并在 Streamable HTTP 上校验 `Origin`。
- 用 `destructiveHint: true` 标记会修改状态的工具，并要求宿主审批。
- 显式传入目录和文件范围，不要依赖已废弃的 Roots。
- 把资源和工具输出视为不可信数据。
- stdio 下 stdout 只留给 JSON-RPC；诊断信息写入 stderr。

## 使用它

在课程目录下运行：

```bash
python3 code/main.py
cd code
python3 -m unittest discover tests -v
```

第一行应报告在协议 `2026-07-28` 上发现了 `demo-server`。然后查看 `MCPClient.request`：它会为每次调用重新构造 `_meta`。从某个请求中删掉元数据，观察服务器如何拒绝它。

## 交付它

`outputs/skill-mcp-server-designer.md` 能把一个业务领域变成无状态 MCP 设计。它的验收门槛要求包含发现结果、逐请求元数据策略、确定性的可缓存列表、显式状态句柄、传输头部、授权和审批规则。

## 继续深入 MCP

本课给你的是协议模型。Phase 13 把四个生产边界拆成独立的构建与验证课程：

1. [MCP 工具契约与内容](../../../13-tools-and-protocols/28-mcp-tool-contracts-and-content/docs/en.md) 涵盖封闭输入 schema、结构化内容、路由元数据、不透明分页、补全授权，以及协议错误与工具领域错误的区别。
2. [MCP 可靠性、取消与流控](../../../13-tools-and-protocols/29-mcp-reliability-cancellation-and-flow-control/docs/en.md) 涵盖请求取消、持久任务取消、截止时间、幂等性、背压、代理缓冲和重连行为。
3. [MCP 注册表供应链、准入、漂移与回滚](../../../13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift/docs/en.md) 涵盖命名空间证明、制品来源、不可变固定版本、线上漂移、注册表状态、准入证据和回滚。
4. [MCP 一致性工程](../../../13-tools-and-protocols/31-mcp-conformance-versioning-and-operations/docs/en.md) 涵盖金标准与负例线路记录、严格的版本时代、SDK 差分、代理证据、脱敏、健康门禁和发布回滚。

当服务器要跨越团队或信任边界时，按顺序学习这四课。它们合在一起，把「方法能跑通」推进到「契约在部署全程都保持安全、可诊断」。

## 练习

1. 增加一个 `subtract` 工具，确认 `tools/list` 仍然按字母顺序排列。
2. 删除协议版本键，验证返回 Invalid Params（`-32602`）。然后发送格式正确但不受支持的版本 `2025-11-25`，验证 `-32022`，确认 `requested` 回显了该修订版，并从 `supported` 中选择可用版本。
3. 为 create 操作增加一个服务器签发的 `draftId`，然后要求 update 把它作为参数传入。解释为什么这是应用状态，而不是协议会话。
4. 从一个需要用户确认的工具返回 `input_required`。用新的 ID、一个 `inputResponses` 条目和原样的 `requestState` 重试原调用，而不是发明一个服务器到客户端的 JSON-RPC 请求。
5. 设计一个双时代 stdio 客户端。把结果或可识别的现代错误视为现代服务器，只在遇到无法识别的错误或超时才允许回退到 `initialize`。

## 关键术语

| 术语 | 大家的说法 | 实际含义 |
|------|-----------------|------------------------|
| MCP | 「LLM 的工具协议」 | 用于服务器发现、工具、资源、prompt 和扩展的 JSON-RPC 协议 |
| Host | 「AI 应用」 | 拥有模型和 UI，并挂载一个或多个 MCP 客户端 |
| Client | 「连接器」 | 代表宿主与一个服务器进行 MCP 通信 |
| Stateless MCP | 「无会话」 | 每个请求都携带版本和能力；没有以连接为键的协议状态 |
| `server/discover` | 「能力探测」 | 必需的服务器方法，公布版本、能力和身份 |
| `resultType` | 「结果状态」 | 把结果标记为 `complete` 或 `input_required` |
| State handle | 「工作流 ID」 | 服务器签发、作为普通参数传入的应用标识符 |
| Streamable HTTP | 「远程传输」 | 单个 POST 端点，返回 JSON 或请求作用域的 SSE 响应 |
| MRTR | 「询问并重试」 | 嵌入结果中的输入请求，随后重试原操作 |

## 延伸阅读

- [MCP 2026-07-28 关键变更](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [MCP 服务器发现](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 已废弃特性](https://modelcontextprotocol.io/specification/2026-07-28/deprecated)
