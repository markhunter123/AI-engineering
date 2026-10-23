# MCP 安全：被投毒的元数据、路由与 MRTR 状态

> 无状态并不意味着无信任。它意味着每个请求都暴露服务器和网关独立校验该调用所需的证据。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Time:** ~60 minutes

## 学习目标

- 把工具描述、注解、客户端信息和服务器信息视为不可信数据。
- 检测元数据投毒、描述符变更和跨服务器名称冲突。
- 校验 2026-07-28 请求元数据和 Streamable HTTP 路由头。
- 保护 MRTR `requestState` 免受篡改，并把确认绑定到确切参数。
- 对主体而非已移除的协议会话施加授权和速率限制。

## 问题

模型读取工具描述来决定调用什么。路由器读取工具名来决定把请求发往哪里。用户读取标签来决定批准什么。一个恶意描述符可以同时针对这三者。

官方 MCP 安全指南说得很直接：除非来自受信任的服务器，描述和注解都应视为不可信。即便如此，部署信任也可能变化。一次服务器更新、被入侵的包、注册表错误或网关合并，都可能改变模型看到的内容。

当前协议也改变了安全边界。2026-07-28 没有核心握手，也没有传输会话。只以 `Mcp-Session-Id` 为键来设计审批、速率限制或审计历史的安全方案，不是当前的设计。

## 概念

### 值得检查的七个攻击面

用一张具体清单代替“要小心”这类含糊指示。

1. **元数据投毒。** 描述中包含与声明工具行为无关的指令。
2. **描述符掉包（rug pull）。** 先前已批准的名称、描述、schema 或注解发生变化。
3. **跨服务器遮蔽。** 两个后端暴露同一个未限定工具名，路由静默选择其中一个。
4. **头与体混淆。** `Mcp-Method` 或 `Mcp-Name` 与 JSON-RPC 请求不一致。
5. **能力提升。** 对端声明某个扩展或客户端特性，服务器误把该声明当作授权。
6. **MRTR 状态篡改。** 客户端修改 `requestState`、回答不同的问题，或用不同参数重用确认。
7. **供应链身份混淆。** 把熟悉的显示名当作发布者或服务器身份的证明。

这些攻击面彼此重叠。哈希固定有助于描述符变更，但不能证明第一个描述符是安全的。静态扫描能抓到明显短语，却抓不到细微指令。命名空间能防住一类冲突，却防不住恶意的命名空间服务器。要把多种控制叠加使用。

### 当前请求信封是证据，不是身份

每个 2026-07-28 请求都包含：

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "elicitation": {"form": {}}
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "security-lab",
      "version": "1.0.0"
    }
  }
}
```

在每个请求上校验版本与能力形状。用能力选择兼容的响应形状。不要拿 `clientInfo` 当已认证主体。它是自报的。

同样的警告适用于结果元数据中的 `io.modelcontextprotocol/serverInfo`。它对日志和调试有用。它不是证书、注册表证明或授权决定。

### 先校验路由，再执行策略

对于 `tools/call`，Streamable HTTP 包含：

```text
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.export
```

头中的方法必须等于体中的方法。头中的名称必须等于 `params.name`。在选择后端、应用 RBAC 或消耗限流令牌之前，以 `-32020` 拒绝不一致。

这个顺序堵住了一个常见歧义：一个组件依据请求体授权，另一个组件却按头部路由。

线路校验遵循一个确切序列。校验 JSON-RPC 与元数据类型，比较头值与请求体，然后检查匹配到的版本是否受支持。头不匹配返回 HTTP 400 和 `-32020`。如果头和体一致地指向不支持的版本，返回 HTTP 400 和 `-32022`，且 `data` 恰好为 `{"supported":["2026-07-28"],"requested":"<actual>"}`。未知方法返回 HTTP 404 和 `-32601`。

当契约需要结构化恢复信息时，每个错误对象都包含可选的 `data`。通知没有 `id`，因此永远不会收到 JSON-RPC 成功或错误响应。被接受的 HTTP 通知返回 202 与空响应体。

### 固定整个描述符

仅哈希描述会漏掉 schema 和注解的变化。请规范化并哈希用户批准过的描述符字段：

```python
normalized = json.dumps(tool, sort_keys=True, separators=(",", ":"))
digest = hashlib.sha256(normalized.encode()).hexdigest()
```

把摘要存在 `notes.export` 这样的限定键下，并在这个玩具示例之外一并保存发布者证据和批准时间。

每次刷新时：

- 未知键：隔离，等待审查。
- 同键不同摘要：作为掉包隔离，等待重新批准。
- 重复的未限定名称：要求确定性命名空间。
- 扫描器命中：阻断并审查完整描述符。

哈希相等证明的是稳定性，不是安全性。被投毒的描述符即使被完美固定，仍然是投毒的。

### 静态扫描是绊线

简单模式可以标记角色标签、指令覆盖、隐藏意图、秘密访问和混淆的网络目标。它们足够便宜，适合安装时和 CI 使用。

它们不是语义证明。安全描述可能因合法的警告语包含被标记的短语。恶意描述也可能避开所有短语。把扫描器输出当作审查证据，而不是自动的“清白分”。

### 合并之前先命名空间

假设两个服务器都暴露 `search`。绝不要让发现顺序决定谁胜出。

```text
notes.search
issues.search
```

限定名是公开的网关名。单独记录后端映射。稳定的名称让审批、审计、哈希固定和 `Mcp-Name` 路由都指向同一个对象。

### 能力是兼容性声明

每请求的 `clientCapabilities` 告诉服务器客户端能处理哪些协议特性。它不授予客户端访问工具、数据或操作的权限。

授权仍来自已认证主体和资源策略。顺序是：

1. 认证传输凭据。
2. 校验版本、头和请求形状。
3. 检查能力兼容性。
4. 授权主体、工具、资源和参数。
5. 执行或请求用户输入。

### 保护无状态 MRTR 确认

有后果的工具可能需要用户确认。当前 MCP 使用多轮往返请求（MRTR），而不是服务器到客户端的回调。

首次响应：

```json
{
  "resultType": "input_required",
  "inputRequests": {
    "confirm": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Export notes to archive?",
        "requestedSchema": {
          "type": "object",
          "properties": {
            "confirm": {"type": "boolean"}
          },
          "required": ["confirm"]
        }
      }
    }
  },
  "requestState": "opaque-integrity-protected-value"
}
```

客户端获取输入，并用新的 JSON-RPC id 重试原始方法：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes.export",
    "arguments": {"query": "private", "destination": "archive"},
    "requestState": "opaque-integrity-protected-value",
    "inputResponses": {
      "confirm": {
        "action": "accept",
        "content": {"confirm": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

每个 `inputRequests` 值都是带 `method` 和 `params` 的完整内嵌请求。其键必须与 `inputResponses` 中对应条目匹配。表单 elicitation 使用对象根 `requestedSchema`，且客户端必须先声明表单 elicitation 能力，服务器才能请求它。

当前能力有两种合法的表单声明。`{"elicitation":{}}` 隐式支持表单 elicitation，而 `{"elicitation":{"form":{}}}` 显式说明。像 `{"elicitation":{"url":{}}}` 这样的仅 URL 声明不支持表单请求。服务器返回 HTTP 400 和 `-32021`，且 `data.requiredCapabilities` 等于 `{"elicitation":{"form":{}}}`。

把 `requestState` 当作敌意输入。对它签名或加密、校验，并绑定到方法、工具、确切参数、用途、过期时间、主体，以及（在重放重要时）一次性 nonce。本课代码使用 HMAC 和精确参数匹配，让这一边界清晰可见。

nonce 账本不能只活在一个网关对象里。可运行模型注入一个有界、按 TTL 剪枝的重放存储，可被多个网关实例共享。其原子认领就是执行边界：只有经过校验的接受或显式终态拒绝才消耗状态。格式错误的响应或 `cancel` 不执行任何东西，并在过期前保持可重试。生产集群需要在共享持久存储中使用同样的条件认领。

不要把隐藏的确认上下文存进协议会话。任何服务器实例都应能校验这次重试。

### 高风险调用的 Rule of Two

沿三个轴分类一次调用：

- 它消费不可信输入。
- 它能访问敏感数据。
- 它会造成有后果的外部操作。

单个自动步骤不应同时结合三者。拆分它、降低权限，或通过 MRTR 请求明确的用户输入。这是一条设计启发式，不是协议能力。

### 执行之前先削减权限

无状态本身不是安全。它移除了隐藏的协议历史，但自包含请求仍可能要求一个权限过大的处理器泄露数据或做出不可逆变更。安全来自在每个边界削减权限：

1. **类型化动词。** 暴露 `archive_note` 这样一个有界操作，而不是能表达无关权限的通用 `run` 或 `request` 工具。
2. **校验参数。** 尽量使用封闭 schema，拒绝未知字段，统一规范化标识符，限制大小，并在策略评估前校验目标、租户和资源所有权。
3. **当前授权。** 把已认证主体绑定到确切的动词、资源、环境和规范化参数。工具注解和客户端能力不授予这种权限。
4. **动作绑定审批。** 对有后果的调用，把审批绑定到类型化动词与规范化参数的摘要，加上主体、过期时间和一次性策略。任何字段变化都需要新的决定。
5. **一等拒绝。** 把拒绝、过期审批、用户拒绝和不安全目标建模为不产生任何副作用的普通结果。不要把拒绝翻译成更弱的回退工具。
6. **脱敏审计证据。** 记录谁发起、使用了哪个被接纳的描述符和策略版本、授权了哪个规范化目标、为何允许或拒绝，以及执行是否开始。存摘要或脱敏值，不存秘密。

每一步都收窄下一个组件的可做范围。最终处理器应收到已经校验的领域命令，而不是原始模型文本加宽泛凭据。在 MRTR 重试、任务更新或网关转发调用时，重复整条链。早先的批准不会把后续请求变成可信的会话流量。

### 当前与旧版交互路径

Roots、Sampling 和 Logging 在 2026-07-28 新实现中已弃用。网关只能把旧的请求通道代码保留为按版本门控的兼容路径。

不要围绕每会话 sampling 限流器构建新防御。把配额施加到已认证主体、签发者、资源、工具和时间窗口。对当前的交互工作，检查 MRTR 输入请求和响应。

### 无状态传输检查

- 在单一 POST 端点接受现代 MCP 消息。
- 对现代 GET 和 DELETE 返回 405。
- 不要铸造或依赖 `Mcp-Session-Id`。
- 把旧版会话和重放头当作非权威输入忽略。
- 为该 POST 返回 JSON 或请求范围的 SSE。
- 仅对选择加入的长生命周期变更通知使用 `subscriptions/listen`。

```figure
tp-tool-poisoning
```

## 构建它

`code/main.py` 实现一个小型进程内安全网关模型。它规范化并固定完整工具描述符，报告元数据投毒和遮蔽，校验现代请求信封与路由值，并执行两轮确认导出，带签名 `requestState` 和注入的共享重放存储。

模型在 HTTP 适配器解析完 JSON 体和路由头之后启动。它不校验 `Content-Type` 或 `Accept`。把同一个分派器接到第 09 课的完整 Streamable HTTP 适配器，后者要求 `Content-Type: application/json`，且 `Accept` 值同时包含 `application/json` 和 `text/event-stream`。

运行它：

```bash
cd phases/13-tools-and-protocols/15-mcp-security-tool-poisoning
python3 code/main.py
python3 -m unittest discover code/tests -v
```

示例故意改动一个描述符。扫描器和摘要比较会产生独立发现。随后导出演示 `input_required` 响应和无状态重试。

## 使用它

用你自己已批准服务器的规范化快照替换 `SAFE_TOOLS`。让凭据和秘密远离该快照。在更新摘要之前审查每个新增或变更的描述符。

在网关上，发现期间运行这些检查，分派前再运行一次。缓存可以减少发现工作，但缓存中的批准必须在描述符变化时过期或失效。

## 交付它

本课交付 `outputs/skill-mcp-threat-model.md`。它生成覆盖元数据、路由、能力、授权、MRTR、缓存、注册表和兼容性边界的当前协议威胁模型。

## 练习

1. 把已认证主体和当前授权决定绑定到密封的 MRTR 状态，然后在不同主体下拒绝重试。
2. 用持久条件插入替换内存重放存储，证明两个进程不能同时认领同一个 nonce。
3. 在重放认领之后、模拟导出之前注入失败。定义并测试使恢复安全的事务或幂等规则。
4. 修改工具的 `inputSchema` 而不改描述。确认整描述符固定能抓住它。
5. 增加一条策略：当 `tools/list` 因主体而异时拒绝公共缓存。
6. 在网关后建模一个旧版服务器。把所有握手和会话行为放进显式的 `2025-11-25` 兼容分支。

## 关键术语

| 术语 | 含义 |
|------|---------|
| Metadata poisoning | 嵌入工具描述符的指令或欺骗性声明 |
| Rug pull | 对先前已批准描述符的变更 |
| Tool shadowing | 重复未限定名称造成的歧义路由 |
| Header mismatch | 路由头与 JSON-RPC 体不一致，错误 `-32020` |
| Hash pin | 完整已批准描述符的摘要 |
| MRTR | 用于服务器请求输入的无状态响应与重试模式 |
| `requestState` | 必须视为不可信输入的不透明往返值 |
| Capability declaration | 协议兼容性声明，不是授权 |
| Implicit form support | 空 `elicitation` 能力对象，等价于表单支持 |
| Qualified tool name | 如 `notes.search` 的稳定网关名称 |

## 延伸阅读

- [MCP 安全与信任指南](https://modelcontextprotocol.io/specification/2026-07-28#security-and-trust--safety)
- [多轮往返请求](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [Streamable HTTP 传输](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [已弃用特性](https://modelcontextprotocol.io/specification/2026-07-28/deprecated)
