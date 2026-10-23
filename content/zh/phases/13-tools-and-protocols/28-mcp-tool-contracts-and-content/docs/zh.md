# MCP 工具契约与内容

> 只有当发现、参数、结果、分页和传输元数据都认同同一份契约时，一个工具才适合被自动化。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 07, 09, and 10
**Time:** ~120 minutes

## 学习目标

- 用 JSON Schema 2020-12 定义工具的输入和输出。
- 校验结构化结果，而不假设它们一定是 JSON 对象。
- 在文本、图像、音频、资源链接和嵌入资源之间做出选择。
- 在工具到达模型之前，拒绝不安全的 `x-mcp-header` 定义。
- 编码参数-头部的值，并验证头部与正文的精确一致。
- 遍历游标分页，而不解释游标值。
- 对 `completion/complete` 的建议做有界和授权控制。

## 问题

调用一个 Python 函数很简单。通过 AI 宿主调用一个远程能力，则是一个契约问题。

服务器发布一个描述符。客户端把该描述符转成模型上下文和用户界面。模型生成参数。网关可能根据镜像的头部来路由请求。服务器执行工具。随后客户端判断结果是否足够安全和有效，可以返回给模型。

任何一个薄弱的边界都会污染整条链。

考虑五种失败：

- 描述符说结果是对象，但服务器返回数组。
- 客户端在 `nextCursor` 为空字符串时就停止分页。
- token 参数被镜像到 HTTP 头部，从而对中间人可见。
- 一个 Unicode 路由值作为原始头部发送，随后网关和源站解释出不同的字节。
- completion 端点向一个无权的调用者建议生产环境。

这些失败没有一个是靠更好的 prompt 就能修好的。它们需要明确的协议契约和应用契约。

## 契约管道

把每次工具调用看作五道门：

1. **发现。** 读取一个确定性的、可分页的工具列表。
2. **准入。** 校验每个描述符，并应用本地安全策略。
3. **调用。** 校验参数并构建传输元数据。
4. **执行。** 运行处理器并正确分类失败。
5. **消费。** 在模型使用之前校验内容块和结构化输出。

```figure
mcp-contract-pipeline
```

宿主拥有准入和消费两道门。服务器无法强迫客户端信任它的注解、schema 或输出。

## JSON Schema 是运行时边界

在 MCP `2026-07-28` 中，`inputSchema` 和 `outputSchema` 使用 JSON Schema。当 `$schema` 缺失时，默认方言是 2020-12。

输入 schema 必须是一个 schema 对象。没有参数的工具也应准确说明它接受什么：

```json
{
  "type": "object",
  "additionalProperties": false
}
```

这比 `{ "type": "object" }` 更严格，后者会接受任意属性。

输出 schema 是可选的。一旦服务器发布了它，每个完整的工具结果就承诺返回符合它的 `structuredContent`，包括 `isError: true` 的结果。错误标志是对执行结果的分类；它不豁免已发布的输出契约。客户端应校验结果，而不是信任描述符。

### 结构化内容可以是任意 JSON 值

不要把 `structuredContent` 硬编码为字典。它可以是：

- 对象；
- 数组；
- 字符串；
- 数字；
- 布尔值；
- `null`。

这个工具返回数组：

```json
{
  "name": "tag_catalog",
  "inputSchema": {
    "type": "object",
    "additionalProperties": false
  },
  "outputSchema": {
    "type": "array",
    "items": {"type": "string"}
  }
}
```

它的成功结果是合法的：

```json
{
  "resultType": "complete",
  "content": [
    {
      "type": "text",
      "text": "[\"contracts\", \"mcp\", \"stateless\"]"
    }
  ],
  "structuredContent": ["contracts", "mcp", "stateless"],
  "isError": false
}
```

为兼容性起见，结构化结果还应在一个文本块中包含序列化后的 JSON。文本不是校验来源，`structuredContent` 才是。

### 一个小校验器仍能教会边界

本课使用一个刻意收窄的 JSON Schema 子集，因为这样它能保持在 Python 标准库之内。它检查示例工具用到的机制：

- object、array、string、integer、number、boolean 和 null 类型；
- 必需属性；
- `additionalProperties: false`；
- 数组 items；
- enum 值；
- 字符串最小长度。

这不是完整生产校验器的替代品。可复用的教训是校验发生的位置：描述符在发现之后校验，参数在执行之前校验，结构化结果在消费之前校验。

## 内容块有不同的成本

`content` 数组可以组合多种内容类型。

| 类型 | 用于 | 主要边界 |
|------|------------|---------------|
| `text` | 人和模型可读的摘要 | 把文本视为不可信输出 |
| `image` | 以 base64 编码的视觉证据 | 校验媒体类型和大小 |
| `audio` | 以 base64 编码的语音或录音输出 | 校验媒体类型和时长限制 |
| `resource_link` | 客户端之后可能获取的 URI | 重新授权后续的资源读取 |
| `resource` | 直接嵌入结果中的数据 | 立即执行载荷和内容限制 |

资源链接并不证明该资源出现在 `resources/list` 中。它是本次工具调用返回的一个引用。客户端在跟随该 URI 时仍需应用自己的资源策略。

嵌入资源避免了一次往返，但会增加当前响应的大小。对于大型或独立变化的产物，使用链接。对于必须与结果原子地一起传输的小型证据，使用嵌入资源。

本课的 `evidence_bundle` 结果包含全部五种类型。客户端在接受结果之前会校验每个块。

## `x-mcp-header` 是路由元数据

`inputSchema` 中的属性可以声明 `x-mcp-header`。在 Streamable HTTP 上，客户端会把该参数镜像到 `Mcp-Param-{name}`。

```json
{
  "region": {
    "type": "string",
    "x-mcp-header": "Region"
  }
}
```

当 `region: "eu-west"` 时，传输层可以发出：

```http
Mcp-Param-Region: eu-west
```

这个注解存在的目的是让负载均衡器、网关或策略引擎无需解析 JSON 正文即可路由。它不是存放凭证的地方。

协议对这些注解的约束：

- 头部名称非空，并遵循 HTTP 字段名 token 语法；
- 头部名称在忽略大小写后唯一；
- 属性类型是 string、integer 或 boolean；
- 不允许 `number`；
- 注解只出现在 `inputSchema.properties` 的直接成员上；
- 整数取值保持在 `-9007199254740991` 到 `9007199254740991` 之间。

位置规则是语法性的，并且失败关闭。遍历整棵 schema 树，而不只是你的校验器恰好理解的属性。拒绝嵌套对象 `properties` 之下、`oneOf` 分支、`items`、通过 `$ref` 到达的定义，以及任何输出 schema 中的注解。解析引用并不会把被引用的节点变成直接的顶层属性。

本课增加一条部署策略：拒绝镜像 `password`、`secret`、`token`、`api_key` 或 `authorization` 等名称的描述符。官方规范建议服务器作者不要镜像敏感参数。客户端可以把这条建议变成硬性的准入规则。

审查头部名称，而不是它的值。示例代码记录 `Mcp-Param-Region`，同时让 `eu-west` 不进入审计事件。

### 构建 HTTP 头部之前先编码值

参数值只有在它是非空字符串、由 `!` 到 `~` 的可见 ASCII 字符组成，并且不像编码哨兵时，才能以纯文本传输。其他所有情况都使用下面这种精确形式：

```text
=?base64?{Base64UTF8}?=
```

`Base64UTF8` 是对精确 UTF-8 字节做的标准 base64。不要先裁剪、规范化或替换值。Unicode、空字符串、空格、制表符、控制字符、CR 或 LF、前导或尾随空白，以及任何以 `=?base64?` 开头的值，都要编码。对看起来像哨兵的值再次编码，正是让接收方恢复原始字面文本、而不是把它当作传输语法解码的方式。

布尔值渲染为小写 `true` 或 `false`。整数以十进制渲染，且必须落在 JavaScript 安全整数范围内。超出该范围的值会被拒绝，而不是被中间层四舍五入。

### 服务器检查镜像副本

头部生成只是客户端的一半。在 Streamable HTTP 边界上，服务器必须：

1. 在忽略头部名称大小写的情况下找到可识别的 `Mcp-Param-*` 名称；
2. 在存在时解码精确的 base64 哨兵形式；
3. 将解码后的文本与对应的 JSON 正文参数精确比较；
4. 在分发之前拒绝缺失、重复、意外、格式错误或不匹配的可识别头部。

拒绝方式是 HTTP `400` 加上 JSON-RPC 错误码 `-32020`。正文值和它的编码头部形式都不应进入审计记录。只记录可识别的头部名称和拒绝类别。

`code/main.py` 直接建模这个边界。[第 09 课](../../09-mcp-transports/) 涵盖更广的 Streamable HTTP 校验顺序，包括方法和协议版本的一致性。

## 分页游标是不透明的

MCP 列表操作使用游标分页。服务器选择页大小和游标格式。客户端只有一个决定：

```python
if result.get("nextCursor") is None:
    break
cursor = result["nextCursor"]
```

不要写成：

```python
if not result.get("nextCursor"):
    break
```

空字符串是合法的游标。真值判断会过早停止。

客户端不得解码游标、对游标做递增、与先前游标比较顺序，或推断页码。服务器可以给游标签名、把它绑定到目录版本，或映射到私有状态。那是服务器的实现细节。

示例服务器故意在第一页之后返回 `""`。客户端必须在第二个请求中发送这个确切的值。它的 trace 是：

```text
<first request with no cursor>
<second request with cursor "">
```

无效游标产生 JSON-RPC invalid params，错误码 `-32602`。

## Completion 是授权面

`completion/complete` 为 prompt 参数和资源模板参数提供建议。它对交互式表单很有用，但也可能泄露普通列表方法所保护的名称。

一个 completion 请求会指定一个引用和正在补全的参数：

```json
{
  "method": "completion/complete",
  "params": {
    "ref": {
      "type": "ref/prompt",
      "name": "deployment_review"
    },
    "argument": {
      "name": "environment",
      "value": "st"
    }
  }
}
```

结果最多返回 100 个值，并可能报告 `total` 和 `hasMore`。

应用与被引用 prompt 或资源相同的授权边界。示例中的分析员只能收到 `development` 和 `staging`。只有操作员才能收到 `production`。

生产环境的 completion 还需要：

- 输入校验；
- 调用者感知的过滤；
- 客户端请求去抖；
- 服务器速率限制；
- 有界的结果数量；
- 不暴露敏感建议值的日志。

Completion 是辅助，不是绕过发现的手段。

## 两层错误

把协议错误与工具执行错误分开。

当 MCP 请求无法被正确分发时，使用 JSON-RPC 错误：

- 未知工具名；
- 请求形状格式错误；
- 缺少请求元数据；
- 无效游标。

当调用已经到达工具、且工具报告一个可操作的失败时，使用带 `isError: true` 的完整工具结果：

- 报告来源不可用；
- 日期超出支持范围；
- 业务规则拒绝所请求的操作。

模型通常能修复工具执行错误。它们无法修复一个违反自身输出 schema 的服务器。

如果工具声明了输出 schema，就在该 schema 内建模可操作的失败。示例 `route_report` 的失败会返回所请求的区域以及 `accepted: false`，同时附带人类可读的错误文本和 `isError: true`。

## 构建它

`code/main.py` 用 Python 标准库构建边界的双方。

服务器实现：

- 逐请求的 MCP 元数据校验；
- 带 tools 和 completions 能力的 `server/discover`；
- 确定性的 `tools/list` 分页；
- 四个工具描述符，其中一个必须被拒绝；
- 数组结构化输出；
- 当前全部工具内容块类型；
- 一个 Streamable HTTP 一致性门：解码可识别的参数头部，并在不匹配时返回 HTTP `400` 加 JSON-RPC `-32020`；
- 带授权和速率限制的 completion。

客户端实现：

- 描述符准入；
- 全树 `x-mcp-header` 位置校验和敏感字段策略；
- 精确的纯可见 ASCII 或 base64 UTF-8 值编码；
- 跟随空字符串的不透明游标循环；
- 参数与结果校验；
- 内容块校验；
- 只包含名称、不包含值的头部审计事件。

那个故意不安全的描述符是教学数据。它证明一个被拒绝的工具不会阻止合法工具加载。

## 使用它

从仓库根目录：

```bash
cd phases/13-tools-and-protocols/28-mcp-tool-contracts-and-content/code
python3 main.py
python3 -m unittest discover tests -v
```

演示会打印被准入的工具、被拒绝的描述符、两次分页请求、结构化数组内容、内容块类型、镜像的头部名称、值是否需要编码、HTTP 一致性状态，以及按调用者过滤后的 completion 值。

## 交互实验

打开 `code/main.py`，找到 `TOOLS`。

1. 把 `tag_catalog.outputSchema.type` 从 `array` 改为 `object`。
2. 运行演示。客户端应拒绝返回的数组。
3. 恢复 schema。
4. 让第一页的 `nextCursor` 保持为 `""`，然后让最后一页返回 `nextCursor: None`，而不是省略该字段。
5. 运行测试，比较游标 trace。
6. 给一个字符串属性加上 `x-mcp-header: "Authorization"`。
7. 确认描述符准入在调用之前就拒绝它。
8. 尝试包含 Unicode、换行、前后空格，以及字面文本 `=?base64?SGVsbG8=?=` 的 `region` 值。解码每个发出的头部，证明原始值被完整保留。
9. 把注解移到 `oneOf`、`items` 或 `$ref` 定义之下。确认每个描述符都被拒绝，即使该分支从未被演示使用。
10. 移除可识别的头部，或修改它的解码值。确认 HTTP 边界返回状态 `400` 和 JSON-RPC 错误码 `-32020`。

重点不是记住某个 JSON 形状，而是观察每道门在拥有它的边界上失败。

## 实践练习

用 `search_evidence` 工具扩展这个契约实验。

要求：

1. 它的输入 schema 接受 `query`、`limit` 和一个安全的 `region` 路由字段。
2. 它的输出 schema 是一个对象数组，包含 `uri`、`title` 和 `score`。
3. 结果包含兼容性文本，以及每一项的资源链接。
4. 参数拒绝未知属性。
5. `limit` 受应用层校验约束。
6. 无法访问某个 URI 的调用者，绝不会通过 completion 或工具输出看到该 URI。
7. 测试包括一个不符合规范的 score、一个无效的头部注解，以及一个两页列表。
8. 头部值测试覆盖可见 ASCII、Unicode、控制字符、空白、看起来像哨兵的文本，以及 JavaScript 安全整数的两端边界。
9. HTTP fixture 接受大小写不敏感的头部名称，但对缺失或不匹配的可识别值返回状态 `400` 和错误码 `-32020`。

## 交付产物

`outputs/skill-mcp-contract-reviewer.md` 是一个扁平、可复用的审查技能。给它一个工具描述符、示例结果、分页行为和 completion 策略，它会返回准入决定、结果校验计划、头部策略和具体的失败测试。

## 验证它

当下列陈述全部为真时，本课才算完成：

- `tools/list` 在重复调用时返回相同的逻辑顺序。
- 当 `nextCursor` 为 `""` 时，客户端执行第二次请求。
- 不安全的敏感头部描述符被排除，而其他工具仍然可用。
- 一个数组能通过其数组输出 schema。
- 一个对象无法通过同一个数组 schema。
- 错误结果不能省略或违反已发布的输出 schema。
- 文本、图像、音频、资源链接和嵌入资源块都能通过校验。
- 头部审计事件包含名称，不包含值。
- 纯可见 ASCII 保持纯文本；Unicode、控制、带填充、空字符串和看起来像哨兵的值，经过精确的 base64 UTF-8 编码往返后保持一致。
- 超出 JavaScript 安全范围的镜像整数被拒绝。
- `oneOf`、`items`、嵌套对象、`$ref` 定义或输出 schema 下的注解在准入阶段被拒绝。
- 大小写不敏感的可识别头部名称，只有在解码值与正文精确匹配时才通过；缺失或不匹配的副本产生 HTTP `400` 和 JSON-RPC `-32020`。
- 分析员的 completion 永远不会返回 `production`。
- 工具失败使用 `isError: true`；格式错误的协议调用使用 JSON-RPC `error`。

## 生产环境故障模式

| 故障 | 学习者看到的现象 | 正确的响应 |
|---------|-----------------------|------------------|
| 客户端假设输出是对象 | 合法的数组失败或被静默包装 | 按已发布的 schema 校验，不要只支持对象类型 |
| 空游标被当作假值 | 最后一页消失 | 只要 `nextCursor` 存在且非 null 就继续 |
| 敏感值被镜像 | 机密出现在代理、WAF 或 trace 数据中 | 拒绝描述符，把机密保存在受保护的请求数据中 |
| 原始 Unicode 或空白被镜像 | 网关和源站不一致，或值被规范化 | 使用精确的 base64 UTF-8 哨兵编码，解码后再比较 |
| 注解藏在 schema 分支中 | 客户端在准入时漏掉路由元数据 | 遍历整棵 schema 树，只允许直接的顶层属性 |
| 大整数被镜像 | JavaScript 中间层对路由值做了舍入 | 拒绝超出 JavaScript 安全整数范围的值 |
| 头部与正文不一致 | 网关路由到一个目标，源站执行另一个 | 在分发前以 HTTP `400` 和 JSON-RPC `-32020` 拒绝 |
| 输出 schema 被忽略 | 下游代码消费损坏的结构 | 在模型或应用使用之前校验 |
| 资源链接被自动信任 | 调用者跟随了未授权的 URI | 每次资源读取都重新授权 |
| Completion 共享全局建议 | 隐藏的租户名称泄露 | 按调用者、引用和授权过滤 |
| 工具注解被当作策略 | 破坏性操作绕过确认 | 在注解之外强制执行授权和审批 |
| 一个格式错误的工具破坏了发现 | 整个服务器不可用 | 拒绝坏描述符，独立准入合法工具 |

## 项目关联

Phase 13 的项目需要一个能合并多个服务器工具的网关。本课提供了它的准入核心。

用该产物为项目的四类证据评分：

- 确定性且完整的分页发现；
- 模型暴露之前的描述符校验；
- 经过校验的结构化输出，加上有界的内容块；
- 保留授权边界的 completion 和路由元数据。

不要仅凭一次成功的 `tools/call` 就宣称网关兼容。要捕获描述符、分页 trace、准入工具集、拒绝工具集和一份经过校验的结果。

## 关键术语

| 术语 | 含义 |
|------|---------|
| `inputSchema` | 定义可接受工具参数的 JSON Schema 对象 |
| `outputSchema` | 定义 `structuredContent` 的可选 JSON Schema |
| `structuredContent` | 工具结果产生的任意 JSON 值 |
| 内容块 | 类型化的文本、图像、音频、资源链接或嵌入资源 |
| `x-mcp-header` | 把原始类型参数镜像到 Streamable HTTP 元数据的 schema 注解 |
| 不透明游标 | 服务器发出的分页 token，客户端不解释其值 |
| Completion 引用 | 正在补全参数所对应的 prompt 名称或资源 URI/模板 |
| 准入 | 客户端决定暴露还是拒绝一个已发现的描述符 |

## 延伸阅读

- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Completion](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/completion)
- [MCP Pagination](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/pagination)
- [MCP Streamable HTTP Parameter Headers](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http#custom-headers-from-tool-parameters)
