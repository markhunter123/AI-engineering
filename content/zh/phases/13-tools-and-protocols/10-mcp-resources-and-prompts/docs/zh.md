# MCP 资源与提示：为无状态服务器提供可寻址的上下文

> 工具执行操作。资源暴露可寻址的内容。提示打包用户选定的消息模板。一个良好的 MCP 服务器会让这些约定彼此分离且可预测。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lesson 07 (Building an MCP Server), Phase 13, Lesson 09 (MCP Transports)
**Time:** ~60 minutes

## 学习目标

- 根据使用者的意图在工具、资源和提示之间做选择。
- 通过必需的 `server/discover` 公布资源和提示接口。
- 构建确定性的 `resources/list` 和 `prompts/list` 结果。
- 应用 `ttlMs` 和 `cacheScope`，且不泄露用户专属数据。
- 对无效或未知的资源 URI 返回 JSON-RPC 错误 `-32602`。
- 打开 `subscriptions/listen` POST 响应流，并按订阅 ID 关联每个事件。
- 把资源内容和提示模板视为不可信的服务器输出。

## 从使用者出发

误用 MCP 最简单的方式就是从实现代码出发。数据库查询被做成工具，因为函数用起来熟悉。可复用的工作流被做成资源，因为它存在文件里。提示被做成隐藏策略，因为宿主可以注入它。

应当从「谁来做选择」以及「他们期待什么」出发。

| 原语 | 主要意图 | 选择方 | 典型结果 |
|---|---|---|---|
| 工具 | 执行一个操作 | 模型或应用 | 结构化的动作结果 |
| 资源 | 读取某个 URI 上的内容 | 宿主、应用或用户 | 文本或二进制内容 |
| 提示 | 启动一个可复用的消息工作流 | 用户通过宿主 UI | 一条或多条提示消息 |

`notes://note-1` 上的笔记是资源，因为它是可寻址的内容。`delete_note` 是工具，因为它改变状态。`review_note` 是提示，因为由用户选择一个预先准备好的评审工作流。

不要为了让接口看起来完整，就把同一个操作同时暴露成三种原语。每个额外的接口面都需要发现、授权、缓存、错误处理、测试和文档。

## 2026-07-28 的无状态信封

本课面向 MCP 协议修订版 `2026-07-28`。此配置文件中没有初始化握手，也没有协议会话。每个请求都在预留的 `_meta` 键中携带自己的协议版本和客户端能力。

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "resources/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      },
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

服务器必须实现 `server/discover`。它的结果会公布受支持的版本、资源和提示能力、实现身份以及缓存提示。客户端也可以直接调用其他方法，但发现能为它在构建 UI 之前提供一个稳定的快照。

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "resources": {"listChanged": true, "subscribe": true},
    "prompts": {"listChanged": true}
  },
  "ttlMs": 3600000,
  "cacheScope": "public"
}
```

普通结果声明 `"resultType": "complete"`。响应的 `_meta` 用 `io.modelcontextprotocol/serverInfo` 标识提供服务的实现。这些信息对诊断有用，但它不是认证身份。携带不支持修订版的请求会返回 `-32022`，同时附带请求的修订版和服务器支持的修订版。

无状态约定会改变你的设计直觉。列表不能依赖同一连接上的先前调用。授权可能改变可见集合，因为凭据是请求输入的一部分，但连接历史绝不能影响可见集合。

## 资源是稳定的 URI 约定

资源是由 URI 标识的内容。请在编写处理器之前先设计 URI。

良好 URI 的属性：

- 足够稳定，可以被收藏或在请求之间传递。
- 命名空间归属于服务器的领域。
- 独立于进程 ID 或连接。
- 在访问存储之前完成验证。
- 每次读取都经过授权。

`notes://note-1` 比 `note-1` 更好，因为它的命名空间是显式的。文件服务器可以使用 `file://` URI，但它仍然必须在解析符号链接和相对路径段之后检查配置的目录边界。

`resources/list` 返回当前对调用方可见的资源。按 URI 之类的稳定键排序。确定性顺序可以避免嘈杂的缓存未命中、快照变化，以及刷新之间跳动的宿主 UI。

```json
{
  "resultType": "complete",
  "resources": [
    {
      "uri": "notes://note-1",
      "name": "Architecture decision",
      "description": "Why the service uses a stateless boundary",
      "mimeType": "text/markdown"
    }
  ],
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

`resources/read` 返回一个或多个内容项。未知 URI 不是一次成功的空读取。当前的 Resources 规范把无效或未知的资源 URI 归为 JSON-RPC 无效参数，错误码为 `-32602`。

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "error": {
    "code": -32602,
    "message": "Unknown or invalid resource URI",
    "data": {
      "uri": "notes://missing"
    }
  }
}
```

这一区分让客户端能把「不存在」与「合法的空文档」分开。它还能防止意外回退到范围更宽的查找。

### 资源模板

资源模板描述一族参数化的 URI。当列举每一个具体条目代价高昂或数量无界时，就使用资源模板。例如，`notes://projects/{project}/decisions/{decision}` 告诉客户端如何构造一个有效地址，而无需返回每一条决策。

模板不会削弱验证。要解析变量、应用授权、强制长度和字符限制，并用带类型的参数构建存储查询。绝不要把任意 URI 尾部拼接进文件系统路径或数据库语句。

### 内容不是可信的指令

资源文本可能包含提示注入、机密信息、误导性指令或畸形标记。宿主应保留来源信息，并始终把资源内容当作数据。服务器应限制内容大小、返回准确的 MIME 类型、对调用方无权访问的字段进行脱敏，并避免返回无关记录。

## 提示是用户控制的模板

MCP 提示是为用户显式选择而设计的。宿主可以把它们渲染成斜杠命令、菜单项或工作流按钮。协议并不规定必须使用某一种 UI。

在请求授权相同的情况下，`prompts/list` 应当是确定性的。每个提示都需要稳定的名称、有用的描述，以及参数声明，让宿主能在调用 `prompts/get` 之前收集输入。

```json
{
  "resultType": "complete",
  "prompts": [
    {
      "name": "review_note",
      "title": "Review a note",
      "description": "Review one note for a named concern",
      "arguments": [
        {
          "name": "uri",
          "description": "The note resource URI",
          "required": true
        }
      ]
    }
  ],
  "ttlMs": 600000,
  "cacheScope": "public"
}
```

`prompts/get` 把参数解析成消息。它不会替换宿主的系统指令。宿主决定返回的消息如何进入模型上下文，并让自己的可信策略保持在更高优先级。

在服务器边界验证提示参数。提示 URI 应通过与直接资源读取相同的授权检查。不要让提示成为绕过资源访问的旁路通道。

## 缓存提示是正确性的一部分

`ttlMs` 告诉客户端一个结果可以被复用多久。`cacheScope` 描述谁可以共享这个缓存值。

| 作用域 | 含义 | 典型用途 |
|---|---|---|
| `public` | 在授权允许时可在用户之间复用 | 公开的提示目录 |
| `private` | 绑定到请求用户或凭据上下文 | 用户拥有的笔记内容 |

根据数据的变化速率和过时造成的损害来选 TTL。公开的提示目录用五分钟可能合适。私有笔记的读取可以用一分钟。

MCP 只把 `public` 和 `private` 定义为 `cacheScope` 的取值。对于含机密或快速变化的结果，返回 `cacheScope: "private"` 和 `ttlMs: 0`，然后在宿主缓存策略中应用任何更严格的 no-store 规则。`no-store` 本身并不是 MCP 的 `cacheScope` 取值。

缓存提示永远不会取代授权。缓存键必须包含所有会改变可见性的请求维度，包括租户、用户、作用域、语言区域和分页游标。如果共享缓存无法安全地表达这些维度，就使用 `private` 加零 TTL，并配合宿主级 no-store 策略。

## 订阅使用客户端打开的响应流

现代订阅模式取代了过去的 `resources/subscribe` RPC 和旧的 HTTP GET 事件端点。

客户端把 `subscriptions/listen` 作为普通 JSON-RPC 请求发送。在 Streamable HTTP 上，这是一次 POST，其响应保持打开，成为 SSE 流。`notifications` 对象是一份允许列表。服务器不得投递未被请求的通知类型。

```json
{
  "jsonrpc": "2.0",
  "id": 17,
  "method": "subscriptions/listen",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    },
    "notifications": {
      "resourcesListChanged": true,
      "promptsListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

请求 ID 就是订阅 ID。在发送任何被请求的事件之前，服务器先发送 `notifications/subscriptions/acknowledged`。它的过滤器只包含服务器实际接受的那部分子集。

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "notifications": {
      "resourcesListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

该流上后续的每一个事件都携带相同的元数据。

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "uri": "notes://note-1"
  }
}
```

该通知只是说资源发生了变化。客户端通过 `resources/read` 重新读取它，并受当前授权约束。客户端不假设事件本身包含新文档。

多个订阅可以共享同一条 stdio 通道。订阅 ID 让客户端可以把它们解复用。在 HTTP 上，关闭响应流即取消订阅。优雅结束流的服务器会返回一个与原始请求关联的最终 `resultType: "complete"` 响应。

不要把一个订阅流当作协议会话。后续的读取仍然是一个完整请求，可能到达任意健康的服务器实例。

```figure
t3-primitive-sort
```

## 交互实验

用这张图对项目跟踪系统中的五项能力进行分类：issue 详情、创建 issue、sprint 评审模板、项目策略和关闭 issue。然后判断哪些列表可以公开缓存、哪些读取必须保持私有，以及哪些资源值得发送更新通知。

对每一种分类，都要说清楚选择方是谁。如果由模型执行动作，就用工具。如果由宿主读取以 URI 寻址的内容，就用资源。如果由用户启动一个准备好的消息工作流，就用提示。

## 实践练习

从仓库根目录运行模拟器：

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

按以下顺序检查运行记录：

1. 确认 `server/discover` 公布了当前修订版和两种能力。
2. 确认两个列表结果都已排序，并使用 `resultType: "complete"`。
3. 确认列表和读取结果都带有有意设置的缓存提示。
4. 把读取 URI 改成 `notes://missing`，观察 `-32602`。
5. 确认订阅确认先于资源事件。
6. 确认事件和优雅关闭都携带订阅 ID `5`。

这个 Python 模型不打开真实的 HTTP 连接。它表示的是 SDK 必须放到请求作用域响应流上的消息。生产环境中请使用官方 SDK 处理帧格式和传输。

## 交付产物

`outputs/skill-primitive-splitter.md` 是一份可复用的 MCP 原语选择设计评审。它现在会检查确定性发现、缓存作用域、无效 URI 行为和现代订阅过滤器。

本课还交付 `assets/primitive-split.svg`，它是原语与订阅边界的静态版本，供离线学习使用。

## 验证它

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

预期结果：主程序打印一份 JSON 运行记录，测试命令报告至少十二个通过的测试。

## 项目关联

当你的项目服务器在动作之外还要暴露可寻址的知识时，请使用这套约定。其中应包含一份确定性目录快照、一次经过授权的资源读取、一次提示解析、一个无效 URI 案例，以及一份订阅运行记录。

你的证据应表明：没有任何列表依赖连接历史，订阅事件也绝不会授予对底层资源的访问权。

## 练习

1. 添加 `notes://projects/{project}/notes/{id}` 资源模板，并验证两个变量。
2. 为 `resources/list` 添加分页，同时保持确定性顺序。
3. 把一个资源改为 `cacheScope: "private"` 和 `ttlMs: 0`，添加宿主级 no-store 策略，并解释需要这两项控制措施的威胁是什么。
4. 添加提示列表变更订阅，并证明当过滤器省略 `promptsListChanged` 时不会发送任何事件。
5. 创建两个同时存在的订阅，并证明每个事件都携带正确的请求 ID。
6. 为读取处理器添加授权主体，并证明缓存条目不能跨越主体。

## 关键术语

- **资源（Resource）：** 由 MCP 服务器暴露的、以 URI 寻址的内容。
- **提示（Prompt）：** 由 MCP 服务器暴露的、由用户控制的消息模板。
- **确定性列表（Deterministic list）：** 在相同请求输入下，成员和顺序都稳定的发现结果。
- **`ttlMs`：** 以毫秒计的缓存新鲜时长。
- **`cacheScope`：** 缓存结果的共享边界。
- **`subscriptions/listen`：** 一个长生命周期请求，其响应流投递经过显式过滤的通知。
- **订阅 ID（Subscription ID）：** 原始的 listen 请求 ID，在通知元数据中重复出现。
- **无效参数（Invalid parameters）：** JSON-RPC 错误 `-32602`，用于无效或未知的资源 URI。
- **不支持的协议版本（Unsupported protocol version）：** JSON-RPC 错误 `-32022`，包含 `supported` 和 `requested` 修订版。
- **`server/discover`：** 必需的服务器方法，返回受支持的修订版、能力、身份以及可选的缓存提示。

## 延伸阅读

- [MCP 2026-07-28 Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP 2026-07-28 Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP 2026-07-28 Subscriptions](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [MCP 2026-07-28 Caching](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/caching)
