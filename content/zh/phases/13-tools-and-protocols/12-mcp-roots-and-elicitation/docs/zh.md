# 显式作用域与无状态 Elicitation

> Roots 在 MCP 2026-07-28 中已被弃用，而且它从来都不是安全沙箱。把作用域放进可见的工具参数或资源 URI，在服务器上对其授权，并在工具确实需要用户输入时使用 MRTR。用户看到的是决策，模型看到的是句柄，任何服务器实例都能处理重试。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 11 (stateless MRTR)
**Time:** ~60 minutes

## 学习目标

- 用显式工作区参数、资源 URI 或服务器配置取代已弃用的 Roots。
- 把作用域提示与授权、路径包含检查和操作系统沙箱区分开。
- 通过 MRTR 的 `input_required` 结果投递 form 模式的 `elicitation/create`。
- 在逐请求的客户端能力中公布 elicitation 支持，并拒绝不支持的模式。
- 把 `accept`、`decline` 和 `cancel` 作为不同结果分别验证。
- 把破坏性确认绑定到已认证主体、原始参数、候选集和过期时间。

## 两个看起来相似的问题

一个笔记工具收到这个请求：「删除旧的 TPS 报告。」

服务器必须回答两个不同的问题。

1. 这个操作可以触碰哪个工作区？
2. 三条匹配的笔记中，用户指的是哪一条？

第一个是作用域和授权。第二个是交互式消歧。把二者混在一起会导致危险的设计，例如把客户端提供的文件夹当作「调用方可以删除其中所有内容」的证明。

## Roots 是一个迁移接口面

早期的 MCP 修订版允许客户端公布 Roots，并在列表变化时通知服务器。Roots 只是信息性指引。它们并不约束服务器进程能读什么，不授予调用方权限，也不创建操作系统沙箱。

MCP 2026-07-28 在新设计中弃用 `roots/list` 和 `notifications/roots/list_changed`。请优先选择以下显式替代方案之一：

- 当作用域按调用变化时，使用 `workspaceUri` 或 `directory` 工具参数。
- 当操作本身已经针对某个资源时，使用资源 URI。
- 当一个部署拥有一个固定工作区时，使用服务器配置。
- 当代码必须在技术上无法逃逸时，使用进程沙箱或受限文件系统。

如果现有的 2026-07-28 集成在弃用窗口期内仍然需要 `roots/list`，服务器应把它内嵌到 MRTR 的 `inputRequests` 中。它不得发送实时的反向请求。那只是一个迁移适配器；新的处理器应改为接受显式作用域。

模型可以看到并重复一个显式句柄。隐藏的传输会话作用域更难检查、重放、审计和路由。

### 三层规则

显式 URI 本身仍然不自带授权。请强制执行全部三层：

1. **授权：** 这个已认证主体是否被允许使用该工作区？
2. **包含：** 规范化后的目标 URI 是否仍在已授权工作区的边界之内？
3. **沙箱：** 操作系统能否在服务器被攻陷时阻止它逃逸？

可运行的服务器维护一份已授权工作区 URI 的允许列表，规范化百分号编码的路径，检查真正的路径分量边界，并在删除前一刻重新检查包含性。

朴素的字符串前缀检查是错误的：

```text
allowed:   file:///work/notes
attacker:  file:///work/notes-evil/secret.md
traversal: file:///work/notes/%2e%2e/private.md
```

两条恶意路径都以一个具有误导性的字符串开头。先规范化，再比较路径分量。生产环境的文件系统服务器还必须防范符号链接竞态和平台特定的路径语义。

## Elicitation 仍然存在，但投递方式变了

Elicitation 是当前的客户端特性，用于在 `tools/call`、`prompts/get` 或 `resources/read` 期间收集用户输入。方法名仍然是 `elicitation/create`。变化的是线上流的方向。

2026-07-28 的服务器不发送反向 JSON-RPC 请求。它返回一个 `InputRequiredResult`：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "delete_choice": {
        "method": "elicitation/create",
        "params": {
          "mode": "form",
          "message": "Choose one matching note and confirm deletion.",
          "requestedSchema": {
            "type": "object",
            "properties": {
              "note_id": {
                "type": "string",
                "enum": ["note-3", "note-7", "note-14"]
              },
              "confirm": {"type": "boolean"}
            },
            "required": ["note_id", "confirm"]
          }
        }
      }
    },
    "requestState": "integrity-protected-delete-state"
  }
}
```

宿主渲染该表单。用户可以接受、显式拒绝或关闭它。然后客户端用新的 id 重试原始 `tools/call`：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes_delete",
    "arguments": {
      "workspaceUri": "file:///Users/alice/Documents/Notes",
      "title": "TPS report"
    },
    "inputResponses": {
      "delete_choice": {
        "action": "accept",
        "content": {"note_id": "note-14", "confirm": true}
      }
    },
    "requestState": "integrity-protected-delete-state",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

两次调用之间没有协议会话。服务器验证回显的状态，按预期的模式验证响应，检查所选的笔记是否在已签名的候选集中，重新授权工作区，重新检查包含性，然后执行删除。

## 能力协商是逐请求的

支持 form 模式 elicitation 的客户端会声明：

```json
{
  "io.modelcontextprotocol/clientCapabilities": {
    "elicitation": {"form": {}}
  }
}
```

为兼容性起见，空的 elicitation 能力 `"elicitation": {}` 仍然等同于仅支持 form。显式的 `"elicitation": {"form": {}}` 也支持 form 模式。仅声明 URL 的 `"elicitation": {"url": {}}` 则不支持。服务器不得内嵌当前请求能力中未出现的模式，即使更早的请求曾公布过它。

每个请求还携带 `io.modelcontextprotocol/protocolVersion`。版本缺失或不是字符串返回 `-32602`。不支持的字符串返回 `-32022`，附带精确的 `supported` 和 `requested` 数据。缺少 elicitation 支持或仅支持 URL 时，返回 `-32021`，并把 `data.requiredCapabilities` 设为 `{"elicitation":{"form":{}}}`。

没有 JSON-RPC `id` 的信封是一条通知。处理它，但不发出 JSON-RPC 成功或错误响应。在 Streamable HTTP 上，已接受的通知会收到 `202 Accepted`，无响应体。

为了诊断应包含 `clientInfo`，但它是自报的，不能用于为授权识别用户。

服务器实现 `server/discover`，并在 `resultType: "complete"` 下返回 `supportedVersions`、能力、`ttlMs` 和 `cacheScope`。它不为这种现代设计公布 Roots。由于它公布了工具，因此还实现必需的 `tools/list`。该结果返回确定性的 `notes_delete` 描述符、合法的对象型 `inputSchema`、服务器身份元数据和公开缓存提示。

## Form 模式

Form 模式使用一种为可用对话框而设计的受限 JSON Schema。根是一个对象，其属性是扁平的原始类型字段或受支持的枚举数组。深度嵌套的对象和通用文档模式不应出现在确认对话框中。

以下情况使用 form 模式：

- 从若干候选中选择一个；
- 确认一次破坏性操作；
- 收集非敏感的偏好；
- 收集少量必须由用户（而非模型）决定的值。

不要用 form 模式收集密码、API key、访问令牌或支付凭据。这些机密会经过 MCP 客户端，可能进入日志或模型上下文。

服务器会再次验证返回的内容。客户端侧的表单验证改善 UX，但不会建立信任。

## URL 模式

URL 模式发送一个安全的 Web URL，用于带外交互：

```json
{
  "method": "elicitation/create",
  "params": {
    "mode": "url",
    "message": "Connect the report service to continue.",
    "url": "https://mcp.example.com/connect/report-service"
  }
}
```

当敏感信息必须直接进入由服务器控制的 Web 流程（例如第三方授权）时，使用 URL 模式。客户端显示完整的目标地址，并在打开前取得同意。它不得预取该 URL。

`accept` 响应意味着用户同意打开该 URL。它并不证明外部流程已经完成。重试时，服务器检查自己的状态，要么完成操作，要么返回另一个 `input_required` 结果。

URL elicitation 不能取代 MCP 客户端与 MCP 服务器之间的授权。它用于 MCP 服务器需要代表用户执行的外部交互。服务器必须把浏览器用户绑定到发起该 MCP 操作的同一个已认证主体。

## 响应分支

把这些动作当作不同的产品决策，而不是别名：

| 动作 | 含义 | 安全的服务器行为 |
|--------|---------|----------------------|
| `accept` | 用户提交了交互 | 验证内容并继续 |
| `decline` | 用户显式拒绝 | 返回一个完整、非错误的拒绝结果 |
| `cancel` | 用户关闭或未能完成 | 安全停止，并允许稍后重试 |

绝不要把缺失的内容解释为同意。绝不要把 decline 变成重复的提示循环。

## 保护破坏性 MRTR 状态

候选列表不能只存在于提示或未签名的 Base64 值中。客户端控制它回传的一切内容。

本课对一个状态载荷签名，其中包含：

- 已认证主体；
- 发起方法；
- `workspaceUri` 和 `title` 的摘要；
- 表单中展示的允许笔记 id；
- 操作阶段；
- 短暂的过期时间。

在执行变更之前，服务器还会检查实时的笔记记录。这能捕获删除竞态，以及表单展示之后目标被移出工作区的情况。

对于一次性的金融操作或不可逆操作，仅有 HMAC 并不能阻止一个有效状态在其有效期内被重放。要在所有处理器实例共享的重放存储中精确地存储并消费一次 nonce。本课注入一个有界、按 TTL 裁剪的存储，并在执行内存删除期间持有其原子声明。生产数据库应把 nonce 声明与变更耦合在同一个事务或等效的条件写入边界中。

在声明 nonce 之前先验证交互。畸形响应或 `cancel` 不执行任何变更，并让状态在过期前保持可重试。显式的 `decline` 是终态的，因此本课会消费 nonce，但不删除任何内容。

```figure
t3-roots-boundary
```

## 构建它

`code/main.py` 演示了一个现代 `notes_delete` 工具：

- `tools/list` 返回一个确定性的、可缓存的描述符，带有所需的工作区和标题模式。
- 作用域是一个显式的 `workspaceUri` 参数。
- 服务器配置为该课程主体授权该工作区。
- URI 规范化拒绝前缀混淆和编码遍历。
- 每次破坏性删除都要求 form 模式 elicitation。
- elicitation 在 `resultType: "input_required"` 内部传输。
- 已签名的 `requestState` 绑定确切的候选列表和原始参数。
- 注入的重放存储跨服务器实例拒绝同一个已被接受或拒绝的状态。
- 重试使用新的请求 id，并返回 `resultType: "complete"`。

数据存储放在内存中，以便于检查协议行为。换成数据库后，安全规则仍然相同。

## 使用它

从仓库根目录运行：

```bash
cd phases/13-tools-and-protocols/12-mcp-roots-and-elicitation/code
python3 main.py
python3 -m unittest discover tests -v
```

预期的检查点：

- 发现结果公布工具，但不公布 Roots。
- 工具发现返回 `notes_delete`，带 `resultType`、服务器身份和缓存提示。
- 请求 id `1` 在 `inputRequests.delete_choice` 中返回表单。
- 请求 id `2` 回显签名状态并完成删除。
- 前缀路径和编码遍历路径都无法通过包含性检查。
- 更改过的标题无法复用原始的确认状态。
- decline 保持笔记不变。
- 两个共享笔记和重放状态的服务器对象不能都执行同一次确认。
- 空的和显式的 form 声明都能工作，而仅支持 URL 时会返回精确的 `-32021` form 要求。
- 不支持的版本失败使用精确的 `-32022` 数据形状。
- 无 id 的通知不产生 JSON-RPC 响应。

## 交付它

`outputs/skill-elicitation-form-designer.md` 设计显式作用域、授权检查、MRTR 表单、响应分支和状态绑定。它拒绝把已弃用的 Roots 当作沙箱，也拒绝通过 form 模式收集机密。

## 练习

1. 把内存重放存储替换为 SQLite。用一个事务声明 nonce 并删除笔记，然后证明两个进程不能同时提交。
2. 添加 `url` 能力协商和一个带外设置流程。让第三方凭据不出现在 `inputResponses` 中。
3. 把内存笔记映射替换为临时 SQLite 数据库。在变更事务内部重新检查授权和包含性。
4. 为真实的文件系统实现添加符号链接策略。解释为什么仅靠 URI 的词法包含性无法阻止符号链接逃逸。
5. 设计一个 2025-11-25 适配器，把现代 MRTR 处理器的输出映射到遗留的服务器发起 elicitation。让它与当前处理器保持隔离。

## 关键术语

| 术语 | 在 2026-07-28 中的含义 |
|------|------------------------|
| Roots | 已弃用的信息性工作区提示，不是授权或沙箱 |
| 显式作用域（Explicit scope） | 在请求参数中可见的工作区、目录或资源句柄 |
| 包含性（Containment） | 规范化路径分量检查，确保目标位于边界之内 |
| Elicitation | 在 MCP 操作期间获取用户输入的客户端特性 |
| Form 模式 | 使用受限扁平模式的带内结构化用户输入 |
| URL 模式 | 用于敏感或外部工作流的带外交互 |
| MRTR | 无状态的 input-required 结果，随后是一次全新的重试 |
| `requestState` | 由客户端原样回显、由服务器做完整性检查的不透明状态 |
| Decline | 用户显式拒绝 |
| Cancel | 未经批准的关闭或不完整交互 |

## 遗留兼容

对于固定在 2025-11-25 的 peer，`roots/list`、`notifications/roots/list_changed` 以及实时由服务器发起的 `elicitation/create` 可能仍然存在。把这个适配器标注为遗留。不要让遗留的 Root 列表绕过服务器授权，也不要把协议会话的假设带进现代处理器。

## 延伸阅读

- [MCP 2026-07-28 Elicitation](https://modelcontextprotocol.io/specification/2026-07-28/client/elicitation)
- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Roots deprecation](https://modelcontextprotocol.io/specification/2026-07-28/client/roots)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
