# MCP 模型输入：Sampling 迁移与无状态 MRTR

> MCP 2026-07-28 在新设计中弃用 Sampling，并移除了服务器到客户端的请求通道。如果现有工作流仍然需要客户端侧的模型，服务器返回一个 `input_required` 结果，客户端带着模型输出重试原始请求。推理循环变得显式、有界，并在协议层保持无状态。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources and prompts)
**Time:** ~75 minutes

## 学习目标

- 解释 Sampling 在 MCP 2026-07-28 中为何被弃用，并为新服务器选择直接集成模型的默认方案。
- 实现一条兼容工作流，通过多轮往返请求（MRTR）承载 `sampling/createMessage`。
- 把协议修订版和客户端能力放进每个请求的 `_meta` 对象。
- 返回 `resultType: "input_required"`，并用新的 JSON-RPC id 重试原始方法。
- 对 `requestState` 做完整性保护，并把它绑定到主体、方法、参数和过期时间。
- 通过能力检查、审批、响应验证和轮数限制，为模型辅助的循环设定边界。

## 协议之前先做决定

像 `summarize_repo` 这样的工具需要两类工作：

1. 确定性工作：列出文件、读取允许的文件、验证路径并组装内容。
2. 模型工作：挑选代表性文件并综合出摘要。

你现在有两种有效的架构。

### 新服务器：直接与模型提供商集成

这是当前的默认方案。服务器负责模型选择、凭据、预算、重试和可观测性。它向 MCP 客户端返回一个普通的 `tools/call` 结果。

当服务器本身已经是托管服务，或者可预测的模型行为比使用宿主的模型更重要时，选择这一方案。

### 现有 Sampling 工作流：迁移到 MRTR

在弃用窗口期内，Sampling 仍然存在。面向 2026-07-28 的服务器无法向客户端发回一个实时的 `sampling/createMessage` 请求。它改为把该请求内嵌到 `InputRequiredResult` 中。

只有当使用客户端的模型和凭据确实是一项产品需求时，才选择这条兼容路径。要记录一份移除计划，因为新实现不应采用已弃用的 Sampling。

## 无状态约定

2026 年 7 月的协议没有 `initialize` 交换，没有 `notifications/initialized`，也没有 `Mcp-Session-Id`。每个请求都携带过去存在于握手环节的信息：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

服务器在每个请求上验证修订版。版本缺失或不是字符串属于无效参数，`-32602`。不支持的字符串返回 `-32022`，精确数据为 `{"supported":["2026-07-28"],"requested":"<client version>"}`。缺少 Sampling 能力返回 `-32021`，并把 `data.requiredCapabilities` 设为 `{"sampling":{}}`。

没有 JSON-RPC `id` 的信封是一条通知。接收方可以处理它，但既不发出成功响应，也不发出错误响应。对于已接受的通知，Streamable HTTP 适配器返回 `202 Accepted`，无响应体。

服务器还实现 `server/discover`，使用精确的 `supportedVersions` 键、能力、`ttlMs` 和 `cacheScope`，让客户端可以在调用工具之前了解并缓存服务器约定。由于发现结果公布了 `tools`，服务器还实现了必需的 `tools/list`。其确定性的 `summarize_repo` 描述符包含合法的对象型 `inputSchema`、`resultType: "complete"`、服务器身份元数据以及公开缓存提示。

每个成功的现代结果都有一个判别字段：

- `resultType: "complete"` 表示操作已完成。
- `resultType: "input_required"` 表示客户端必须完成内嵌请求并重试。
- 扩展可以定义额外的结果类型。Tasks 扩展在第 13 课中增加了 `"task"`。

## 一次 MRTR 轮次

服务器在处理请求时不能调用客户端。它改为返回这个结果：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "pick_files": {
        "method": "sampling/createMessage",
        "params": {
          "messages": [
            {
              "role": "user",
              "content": {
                "type": "text",
                "text": "Choose three representative files and return a JSON array."
              }
            }
          ],
          "systemPrompt": "Return only the requested value.",
          "modelPreferences": {
            "costPriority": 0.8,
            "intelligencePriority": 0.2
          },
          "maxTokens": 400
        }
      }
    },
    "requestState": "opaque-integrity-protected-value"
  }
}
```

客户端验证自己支持 Sampling，应用其审批和模型策略，并获得模型响应。然后它用不同的 JSON-RPC id 发送一个新请求：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "inputResponses": {
      "pick_files": {
        "role": "assistant",
        "content": {
          "type": "text",
          "text": "[\"README.md\", \"server.py\", \"docs/intro.md\"]"
        },
        "model": "host-model",
        "stopReason": "endTurn"
      }
    },
    "requestState": "opaque-integrity-protected-value",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}}
    }
  }
}
```

这次重试不是协议会话的延续。它是一个新请求：重复原始方法和参数，仅加上当前轮次的 `inputResponses`，并逐字节回显 `requestState`。

MRTR 只允许用于 `tools/call`、`prompts/get` 和 `resources/read`。服务器不得从无关方法返回 `input_required`。

## 多轮状态

本课需要两次模型调用：

1. `pick_files` 返回一个 JSON 数组。
2. `summary` 返回最终的摘要正文。

每次重试只携带该轮次的响应。因此服务器把阶段和已验证的中间数据放进下一个 `requestState`。

要把这个值当作受攻击者控制的数据。仅对一个原始阶段名签名是不够的。要把状态绑定到：

- 已认证的主体，而不是自报的 `clientInfo`；
- 发起请求的方法；
- 原始参数的摘要；
- 一个很短的过期时间；
- 当前阶段和已验证的中间值。

当不需要机密性时使用 HMAC。当客户端不得读取该状态时，使用认证加密。对签名错误、值已过期、主体变化或参数变化的请求，用 `-32602` 拒绝。

客户端不得解析或修改 `requestState`。它唯一的职责是在重试时原样回显该字符串。

## 模型偏好是提示

`costPriority`、`speedPriority` 和 `intelligencePriority` 是彼此独立的偏好。它们不是概率分布，也不需要加起来等于 1。客户端可以忽略它们，因为模型策略归客户端所有。

如果你维护一条遗留 Sampling 流，请把 `includeContext` 保持在 `"none"`。其他上下文模式会增加泄露风险，而且它们本身也已被弃用。在请求中只传递最少的显式上下文。

## 安全不变量

对于内嵌的 Sampling 请求，客户端是信任边界。

- 当策略要求审批时，向用户展示服务器正在要求模型做什么。
- 限制 MRTR 轮数。否则恶意服务器可以制造一个不断消耗模型花费的循环。
- 在把每个 sampling 响应用作文件名、URL 或工具输入之前先验证它。
- 限制每轮的字节数和 token 数。
- 拒绝当前客户端能力中未声明的输入请求。
- 让模型输出不参与授权决策。
- 记录发起方法和输入请求键，但不记录敏感的提示内容。

`clientInfo` 和 `serverInfo` 是展示和诊断元数据。绝不要把它们中的任何一个用作已认证身份。

```figure
t3-sampling-flip
```

## 构建它

`code/main.py` 在不使用任何第三方包的情况下实现了完整的双轮流程：

- `server/discover` 返回 `supportedVersions`，公布工具支持，并返回缓存提示。
- `tools/list` 返回一个确定性的、可缓存的 `summarize_repo` 描述符，带有对象型输入模式。
- `tools/call` 验证逐请求元数据。
- 第一个结果内嵌 `sampling/createMessage` 用于文件选择。
- 第一次重试验证模型结果，并内嵌第二个请求。
- 受 HMAC 保护的 `requestState` 在彼此独立的请求之间携带阶段信息。
- 最终结果使用 `resultType: "complete"`。

假的宿主模型让示例具有确定性。连接真实宿主时，只需替换 `fake_host_model`。服务器端的状态机应保持确定性和可测试性。

## 使用它

从仓库根目录运行：

```bash
cd phases/13-tools-and-protocols/11-mcp-sampling/code
python3 main.py
python3 -m unittest discover tests -v
```

预期的 checkpoint：

- 发现返回一个带 `ttlMs` 和 `cacheScope` 的完整结果。
- 工具发现返回同一个已排序的描述符，带 `resultType`、服务器身份和缓存提示。
- 能力缺失和不支持的版本使用精确的 `-32021` 和 `-32022` 错误数据。
- 无 id 的通知不产生 JSON-RPC 响应。
- 请求 id 是 `[1, 2, 3]`，证明每个 MRTR 轮次都是独立的。
- 前两个结果是 `input_required`。
- 最终结果是 `complete`，包含选中的文件和摘要。
- 在重试时更改原始参数会导致请求状态检查失败。

## 交付它

`outputs/skill-sampling-loop-designer.md` 现在是一份迁移规划器。它首先判断是否应移除 Sampling、改用直接集成模型。如果确实需要兼容，它会产出 MRTR 轮次、状态绑定、能力门控、预算、验证和移除计划。

## 练习

1. 把文件选择响应改成无效 JSON。确认服务器返回 `-32602`，而不是信任模型输出。
2. 在首次调用和重试之间更改 `audience`。解释为什么密封状态会阻止跨请求复用。
3. 增加第三轮，让宿主对摘要进行评审。把之前的摘要放进签名状态中携带，并把整个流程限制在三轮以内。
4. 用一个服务器自有的模型适配器替换假宿主回调，从而移除 Sampling。列出哪些审批、计费和可观测性责任转移到了服务器。
5. 添加一个过期测试，使用一个超过截止时间一秒的状态值。

## 关键术语

| 术语 | 在 2026-07-28 中的含义 |
|------|------------------------|
| Sampling | 已弃用的特性，用于向客户端的模型请求补全 |
| MRTR | 当请求进行中需要客户端输入时采用的无状态重试模式 |
| `InputRequiredResult` | 带 `resultType: "input_required"` 的结果 |
| `inputRequests` | 由服务器分配的、内嵌 elicitation、sampling 或 roots 请求的映射 |
| `inputResponses` | 当前轮次的客户端结果，键与 `inputRequests` 对应 |
| `requestState` | 由客户端原样回显、由服务器验证的不透明服务器状态 |
| `resultType` | 现代 MCP 结果必需的判别字段 |
| 直接模型集成（Direct model integration） | 对需要模型推理的新服务器推荐的替代方案 |
| 能力门控（Capability gate） | 阻止发送客户端未公布的内嵌请求的规则 |
| 循环预算（Loop budget） | 该操作允许的最大轮数、token 数、字节数、时间和花费 |

## 遗留兼容

固定在 2025-11-25 的客户端仍然可以在实时连接上使用旧的、由服务器发起的 `sampling/createMessage` 流程。只把这种保留在特定版本的适配器中。不要让有会话的路径成为 2026-07-28 服务器的架构。

官方 SDK 可以为较旧的 peer 转换现代的 `input_required` 处理器。这个垫片是一层兼容边界，而不是新增依赖会话逻辑的许可。

## 延伸阅读

- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [MCP Sampling deprecation](https://modelcontextprotocol.io/seps/2577-deprecate-roots-sampling-and-logging)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
