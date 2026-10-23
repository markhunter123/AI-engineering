# 无状态 MCP 网关与注册表准入

> 网关应让每条路由都显式可见。2026-07-28 协议为它提供了方法、名称、版本、能力、身份、缓存与追踪边界，而无需传输会话。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13 · 15 (security), Phase 13 · 16 (authorization)
**Time:** ~75 minutes

## 学习目标

- 在没有会话亲和性的前提下，把多个 MCP 服务器聚合到一个 2026-07-28 端点之后。
- 在策略判断或转发之前，校验每个请求的元数据与路由请求头。
- 用稳定的命名空间、确定性顺序、描述符固定（descriptor pin）、RBAC 和私有缓存来合并工具。
- 把注册表记录视为发现证据，它仍然需要准入策略。
- 正确路由请求作用域的 SSE、`subscriptions/listen`、MRTR 重试以及 Tasks 扩展调用。
- 将旧式握手与会话支持同现代路径隔离。

## 问题

将一个客户端直接连接到一个服务器很简单。规模更大的部署需要对更难的问题给出统一的答案：

- 允许使用哪些服务器？
- 哪个主体（principal）可以查看并调用每个工具？
- 两个后端暴露相同名称时会发生什么？
- 描述符变更如何评审？
- 速率限制和审计事件在何处施加？
- 任意一个实例都能处理下一个请求吗？

网关位于客户端与后端 MCP 服务器之间。它对外呈现一个 MCP 端点，施加横切策略，并转发已批准的请求。

较早的网关设计常把一个客户端会话复用到多个后端会话，并重写 `Mcp-Session-Id`。那是一种旧式兼容设计。2026-07-28 核心没有协议会话。

## 概念

### 现代网关路径

对每个请求：

1. 从传输层授权信息中认证主体。
2. 校验 `MCP-Protocol-Version`、`Mcp-Method`、`Mcp-Name` 和 `params._meta`。
3. 对主体、资源、方法、工具和参数进行授权。
4. 应用描述符、注册表、速率与数据策略。
5. 为选定的后端创建新的自包含请求。
6. 校验后端结果并返回网关结果。
7. 记录审计事件，且不记录任何机密。

没有任何一步需要隐藏的协议会话。应用状态仍可存在于数据库、显式句柄、Tasks 或受完整性保护的 MRTR 状态中。

### 运行时策略是网关的首要决策

准入决定哪个后端版本可以进入网关，但并不授权一次实际调用。对每个请求，网关都会根据已认证的主体、签发方与资源、租户、匹配的方法与名称、规范化后的参数、已准入的描述符固定、当前后端健康状况、能力交集、数据分类、速率状态以及任何与操作绑定的审批，重新计算策略。

这个顺序很重要。注册表记录可能仍然有效，而用户的角色已被撤销；描述符可能仍处于固定状态，而目标参数已越过租户边界；后端可能仍处于已批准状态，而事件响应策略已隔离会改变状态的调用。因此，运行时策略是首要的允许或拒绝决策，注册表与描述符证据只是输入。

不要把允许决策缓存在连接或已移除的会话标识符之下。如果策略不可用，应按操作类别遵循已声明的失败策略。安全的默认做法是：对状态变更和敏感读取采用失败关闭（fail closed）；只有在风险模型允许时，明确批准的公开读取路径才可以使用短时效的最近已知策略。记录是哪个策略版本、经由哪条失败路径做出了决策，然后在返回前校验后端结果。

### 单一 POST 端点

现代的 Streamable HTTP 通过 POST 发送每条 JSON-RPC 消息：

```text
POST /mcp
Authorization: Bearer <gateway-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.search
Accept: application/json, text/event-stream
```

对于该 POST，网关可以返回 JSON 或请求作用域的 SSE。现代请求下，GET 与 DELETE 返回 405。`Mcp-Session-Id` 与 `Last-Event-ID` 不会产生授权、亲和性或重放行为。

请求头与请求体的值必须一致。在查找后端之前，用 `-32020` 拒绝不匹配。这样负载均衡器、网关和速率限制器无需解析完整请求体即可路由，同时保持端到端完整性。

按唯一确定的顺序校验：先校验 JSON-RPC 与元数据的类型，再校验请求头与请求体是否相等，最后校验是否支持匹配到的版本。不匹配时返回 HTTP 400 和 `-32020`。若请求头与请求体一致但版本不受支持，返回 HTTP 400 和 `-32022`，且 `data` 精确为 `{"supported":["2026-07-28"],"requested":"<actual>"}`。未知方法返回 HTTP 404 和 `-32601`。

`ProtocolError` 携带可选的 `data`，网关会把它序列化进 JSON-RPC 错误对象。通知没有 `id`，因此永远不会收到 JSON-RPC 的成功或错误响应。被接受的 HTTP 通知返回 202，响应体为空。

### 在每一层实现发现

网关为客户端实现 `server/discover`。它同时发现每个后端，以了解其协议版本、能力和扩展。

网关结果示例：

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": true}
  },
  "ttlMs": 30000,
  "cacheScope": "private",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "enterprise-gateway",
      "version": "2.0.0"
    }
  }
}
```

只对外宣告网关能够端到端兑现的能力交集。后端具备某项特性，并不意味着暴露它就是安全的。没有后端路径支撑的网关特性，宣告出来也没有用处。

`serverInfo` 是自报的展示与诊断数据，不要把它当作注册表或发布者的证明。

### 按请求携带客户端能力

每个转发的请求都需要一个最新的 `_meta` 信封：

```json
{
  "io.modelcontextprotocol/protocolVersion": "2026-07-28",
  "io.modelcontextprotocol/clientCapabilities": {},
  "io.modelcontextprotocol/clientInfo": {
    "name": "enterprise-gateway",
    "version": "1.0.0"
  }
}
```

不要盲目把外层客户端的能力复制给后端。网关本身就是后端的客户端。只宣告网关能够正确中介的特性。

### 确定性命名空间

把后端工具合并到稳定的公开名称之下：

```text
notes.search
notes.create
issues.list
issues.open
```

维护一份从公开名称到后端及原始工具名的映射。绝不要用「先到先得」或「后到覆盖」来处理冲突。公开名称是审批与审计契约的一部分，更改它等同于一次迁移。

`tools/list` 必须是确定性的。当可见性因主体而异时，返回 `cacheScope: private`。有界的 `ttlMs` 可以降低后端发现负载，同时不会让特定用户的列表泄漏到其他授权上下文。

每个对外暴露的工具描述符都包含稳定的名称、描述，以及以对象为根的 `inputSchema`。命名空间处理不能删除必需的描述符字段。完整的列表结果还包含 `resultType`、服务器身份元数据和缓存提示。

### 固定已批准的描述符

在准入时，对完整描述符做规范化（canonicalize），并把它的摘要存放在限定后的公开名称之下。在列表和调用时，将实时描述符与已批准的摘要进行比对。

如果发生变化：

- 将其从 `tools/list` 中移除。
- 拒绝直接调用。
- 发出审计事件。
- 在更新固定值之前，需要策略或人工重新批准。

网关是有用的集中执行点，但它不会让首次见到的描述符自动变成安全的。初始评审仍然必要。

### 注册表帮助发现，而非决策

注册表中的 `server.json` 提供发布元数据。一个基于包发布的记录可能如下所示：

```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
  "name": "com.example/notes",
  "description": "Example notes MCP server.",
  "version": "1.0.0",
  "packages": [
    {
      "registryType": "npm",
      "identifier": "@example/notes-mcp",
      "version": "1.0.0",
      "transport": {"type": "stdio"}
    }
  ]
}
```

发布元数据并不携带网关的安全决策。把经验证的发布者与来源证据保存在独立的准入状态中：

```json
{
  "registryName": "com.example/notes",
  "registryVersion": "1.0.0",
  "publisher": {"namespace": "com.example", "status": "verified"},
  "provenance": {
    "source": "registry.modelcontextprotocol.io",
    "recordId": "com.example/notes@1.0.0"
  },
  "admission": {"status": "approved", "reviewedBy": "gateway-policy"}
}
```

网关检查 `server.json` 的结构，并将其与该外部状态关联起来。网关仍然需要准入策略。

对每个已准入的后端，记录：

- 精确的注册表与记录标识符。
- 经验证的发布者命名空间或域名证据。
- 允许的传输方式与端点。
- 固定的版本或已批准的升级策略。
- 制品或描述符摘要。
- 授权签发方与资源。
- 评审人、批准时间与过期时间。

不要因为某个服务器的显示名称像某个熟悉的产品就接受它。不要把出现在注册表中当作一次运行时安全评审。私有服务器即使从未出现在公共注册表中，也可以通过同一套证据模式获得准入。

本课实现了网关的接缝：在后端变得可路由之前，把发布证据与本地准入结合起来。[第 30 课：MCP 注册表供应链、准入、漂移与回滚](../../30-mcp-registry-supply-chain-and-drift/docs/en.md) 构建了完整的控制平面，涵盖精确的命名空间证明、制品来源、不可变固定、实时描述符漂移、注册表状态核对、防篡改准入账本以及有证据支撑的回滚。请把该供应链状态与上面按请求的运行时决策分开保存。

### 凭据中介

网关认证它的调用方，并单独向后端进行认证。后端凭据永远不会传给客户端。

让这些绑定关系保持显式：

```text
outer principal -> gateway role and policy
backend issuer + resource -> backend registration and token
```

绝不要把外层网关 token 传给后端。绝不要在不同的签发方或资源处复用后端 token。如果某个工具代表最终用户行事，应通过设计好的交换流程或声明模型来保留这种委托关系，而不是用共享的服务凭据冒充该用户。

### 无会话的速率限制

以已认证主体、签发方、资源、公开工具、成本类别和时间窗口为键施加限制。会话 id 并不存在，即便存在也很容易被轮换。

在消耗昂贵工作之前先应用低成本的校验。明确被拒绝的调用是否计入滥用限制、业务配额，还是两者都计入。

### 审计决策链

记录足以重建一次调用的信息：

- 请求与追踪标识符。
- 已认证的主体与签发方。
- 公开工具与后端路由。
- 描述符固定版本。
- 策略决策与原因。
- 延迟与结果类别。
- 适用时的 MRTR 轮次或任务标识符。

对 bearer token、授权码、刷新 token、原始机密以及不必要的敏感参数进行脱敏。

### 请求作用域的 SSE

当工作在这一个请求期间流式进行时，普通 POST 可以返回请求作用域的 SSE。关闭响应流会取消这个进行中的现代 HTTP 请求。

不要创建单独的 GET 流，也不要承诺 Last-Event-ID 重放。那些都是较旧的传输假设。

### 长期存活的变更通知

对于列表与资源的变更通知，现行客户端通过 POST 发送 `subscriptions/listen`，并收到 SSE 响应。通知过滤器使用精确的扁平字段 `toolsListChanged`、`promptsListChanged`、`resourcesListChanged` 和 `resourceSubscriptions`：

```json
{
  "jsonrpc": "2.0",
  "id": "listen-tools",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

第一个事件确认了受支持的子集。其订阅标识符就是打开该流的请求的 JSON-RPC id：

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": "listen-tools"
    },
    "notifications": {
      "toolsListChanged": true
    }
  }
}
```

随后网关只转发已确认的变更类型。该流上的每条通知都在 `params._meta` 中携带相同的 `io.modelcontextprotocol/subscriptionId`。没有自动重放，也没有自动重新监听。重新连接时，客户端重新打开订阅，并刷新它所依赖的列表。服务器主动发起的优雅关闭会返回一个最终完整结果，并带上相同的订阅 id。

现代路径取代了 `resources/subscribe`、`resources/unsubscribe` 以及未经请求的独立 GET 流式传输。仅应把它们保留在按版本门控的旧式路径中。

### 经由网关的 MRTR

当后端返回 `resultType: input_required` 时，只有当外层客户端支持所需的输入请求，网关才能转发该结果。除非网关有意终止并重新发起该交互，否则应逐字节保留 `requestState`。

客户端用一个全新的 JSON-RPC id 和 `inputResponses` 重试原来的公开工具。网关会重新授权这次重试，检查同一条公开路由，然后转发一个新的后端请求。它不能假定早先某一轮已经授予了无限期的批准。

### Tasks 扩展路由

Tasks 是由 `io.modelcontextprotocol/tasks` 标识的官方扩展。它们并不是核心会话的替代品。

客户端在按请求的客户端能力中声明该扩展；只有当网关能够端到端保持生命周期时，才在发现结果中宣告它。对于受支持的 `tools/call`，只有后端决定是返回普通结果还是 `resultType: task`。任务结果直接在结果中携带 `taskId`、`status`、时间戳、`ttlMs` 以及可选的 `pollIntervalMs`。在该结果发送之前，任务必须已经可持久读取。

网关为不透明的任务标识符记录已认证主体与后端路由。后续的 `tasks/get`、`tasks/update` 和 `tasks/cancel` 调用把 `params.taskId` 用作 `Mcp-Name`，从而为中间层提供路由键。`tasks/get` 返回 `resultType: complete` 以及当前任务状态，并在终态时内联最终结果或协议错误。`tasks/update` 针对未完成的任务输入发送带键的 `inputResponses`，并返回一个空的 complete 确认。`tasks/cancel` 是一种协作式意图，只返回空的 complete 确认，并不保证工作停止。

不要实现新的 `tasks/list` 或 `tasks/result` 方法。它们属于较早的实验性模型。需要输入的任务通过 `tasks/get` 暴露完整的嵌入式请求；客户端通过 `tasks/update` 作答，而不是重试原始 tool call。客户端仍按建议的间隔轮询；任务的创建始终由服务器主导。

持久化的任务路由状态是以任务句柄为键的应用数据，而不是协议会话。

### 兼容性边界

如果网关必须服务较旧的客户端或后端：

- 显式检测协议时代。
- 将初始化、传输会话、GET 流、资源订阅和旧任务词汇一并限制在旧式适配器内部。
- 绝不让旧式会话 id 泄漏到现代路由或授权中。
- 优先使用有界的发现探测和显式的回退策略，而不是静默降级。

```figure
t3-gateway-funnel
```

## 构建它

`code/main.py` 实现了一个进程内协议网关和两个后端服务器。每个后端都会收到一个全新的现行协议请求。该网关提供发现、按用户过滤的确定性 `tools/list`、命名空间路由、注册表 `server.json` 加外部准入状态、描述符固定、RBAC、以主体为键的速率限制、审计决策，以及一个建模的 `subscriptions/listen` SSE 确认。

该模型接收已解析的请求体、路由请求头以及已认证的 bearer 身份。它不是完整的 HTTP 适配器，不解析 `Content-Type`，也不处理完整的 `Accept` 契约。可以把它连接到第 09 课的 Streamable HTTP 适配器；后者要求 `Content-Type: application/json`，并要求 `Accept` 值同时包含 `application/json` 和 `text/event-stream`。

运行方式：

```bash
cd phases/13-tools-and-protocols/17-mcp-gateways-and-registries
python3 code/main.py
python3 -m unittest discover code/tests -v
```

该演示会打印外层请求 id 和新的后端请求 id，让无状态的一跳清晰可见。

## 使用它

把进程内的后端对象替换为真实的现行协议客户端，并保持相同的接缝：

- 连接之前先有准入记录。
- 暴露能力之前先发现后端。
- 授权之前先有合格的公开名称。
- 列出或调用之前先固定描述符。
- 转发之前准备全新的按请求元数据。
- 返回之前校验结果。

## 交付它

本课交付 `outputs/skill-gateway-bootstrap.md`。它会产出一份现代网关设计，涵盖入口、发现、准入、命名空间、授权、缓存、流式传输、订阅、MRTR、Tasks、可观测性以及旧式隔离。

## 练习

1. 在外层请求和转发请求的元数据中加入追踪上下文，并在审计事件中记录两者的关联。
2. 增加一个支持 Tasks 的后端，并按 `Mcp-Name` 中的任务 id 路由 `tasks/get`。
3. 修改一个后端的描述符，证明发现和直接调用都被阻止。
4. 增加一项特定于主体的服务器能力，并解释为什么发现结果必须保持私有缓存。
5. 编写一个旧式适配器接口，同时不向现代 `Gateway` 类添加任何旧式状态。

## 关键术语

| 术语 | 含义 |
|------|------|
| MCP gateway | 位于客户端与后端 MCP 服务器之间的策略与路由服务器 |
| Admission record | 允许某个后端进入网关的证据与策略决策 |
| Qualified tool name | 稳定的公开路由，如 `notes.search` |
| Descriptor pin | 在发现和分发期间比对的已批准摘要 |
| Private cache scope | 仅限某个授权上下文使用的缓存结果 |
| Request-scoped SSE | 附着于单个 POST 请求的流式响应 |
| `subscriptions/listen` | 由客户端打开、用于选定长期变更通知的 SSE 流 |
| Task route | 从不透明任务 id 到其后端的应用映射 |
| Legacy adapter | 针对旧式握手与会话行为、显式按版本门控的边界 |

## 延伸阅读

- [Streamable HTTP 传输](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [服务器发现](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [官方注册表 server.json 要求](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [MCP Tasks 扩展](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
