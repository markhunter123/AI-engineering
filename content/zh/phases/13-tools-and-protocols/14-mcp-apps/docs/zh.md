# 无状态协议上的 MCP Apps

> 交互式结果仍然是一次 MCP 工具与资源交换。2026-07-28 核心让这次交换自包含，而 Apps 扩展增加了沙箱化的浏览器界面。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources)
**Time:** ~75 minutes

## 学习目标

- 通过 `server/discover` 和每请求扩展能力声明 MCP Apps 支持。
- 在工具被调用之前，为该工具声明一个 `ui://` 资源。
- 在 2026-07-28 的无状态线路上返回完整的工具与资源结果。
- 将 Apps 的 `ui/initialize` 桥接消息与已移除的 MCP 核心握手区分开。
- 应用来源校验、沙箱化、CSP 和最小权限。

## 问题

文本结果可以描述一条时间线，却无法给用户一条可以筛选、检查或操作的时间线。

MCP Apps 用一个可选扩展解决展示问题。工具定义指向一个 `ui://` 资源。宿主可以在工具运行前获取并审查该资源，在沙箱 iframe 中渲染它，并通过 JSON-RPC 桥接来中介应用的所有操作。

核心协议在 2026-07-28 发生了变化。不要把 App 包进旧的连接生命周期：

- 没有核心 `initialize` 请求或 `notifications/initialized` 通知。
- 没有 `Mcp-Session-Id` 头。
- 每个请求都在 `params._meta` 中携带协议版本和客户端能力。
- 服务器实现 `server/discover`，以便客户端检查版本、核心能力和扩展。
- 每个成功结果都有 `resultType` 判别字段。
- Streamable HTTP 每个请求使用一个 POST。现代 GET 和 DELETE 入口返回 405。

Apps 桥接仍有一个名为 `ui/initialize` 的方法。它属于 iframe postMessage 方言，并不会重建核心 MCP 会话。

## 概念

### 两个协议，一个功能

保持层次清晰：

1. MCP 核心承载 `server/discover`、`tools/list`、`tools/call`、`resources/list` 和 `resources/read`。
2. MCP Apps 扩展声明 UI，并定义 iframe 到宿主的桥接。
3. 浏览器沙箱规则限制 UI 能到达的范围。

扩展标识符是 `io.modelcontextprotocol/ui`。双方都要选择加入。客户端在每个请求的能力对象里发送扩展支持：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "server/discover",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/ui": {}
        }
      },
      "io.modelcontextprotocol/clientInfo": {
        "name": "timeline-host",
        "version": "1.0.0"
      }
    }
  }
}
```

`clientInfo` 建议用于诊断。它是自报数据，不是授权身份。

### 渲染之前先发现

服务器的发现结果会声明该扩展：

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {},
    "resources": {},
    "extensions": {
      "io.modelcontextprotocol/ui": {}
    }
  },
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "timeline-app-server",
      "version": "2.0.0"
    }
  }
}
```

服务器必须支持发现。客户端并非每次操作前都必须调用发现，因为每个操作都自带能力声明。

### 在工具定义上声明 UI

现代 Apps 契约在 `tools/list` 中把 UI 绑定到工具：

```json
{
  "name": "notes_timeline",
  "description": "Render a timeline of notes.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  },
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline.html"
    }
  }
}
```

这是刻意放在调用之前的元数据。宿主可以在某个结果要求展示之前，预加载、缓存并安全审查 HTML。兼容代码可能接受旧的扁平元数据键，但新服务器应输出嵌套的 `_meta.ui.resourceUri` 形式。

`tools/list` 在当前核心中可缓存。请包含确定性排序、`ttlMs` 和 `cacheScope`。当可见工具因用户或令牌而异时，使用 `private`。

### 返回数据，让宿主绑定视图

工具调用返回普通内容加结构化数据：

```json
{
  "resultType": "complete",
  "content": [
    {"type": "text", "text": "Timeline ready."}
  ],
  "structuredContent": {
    "notes": [
      {"id": "note-1", "title": "Discover", "created": "2026-07-28"}
    ]
  },
  "isError": false
}
```

宿主已经知道哪个视图属于该工具。不要为了重复 URI 而发明新的内容块。

### 把应用作为资源提供

服务器在发现结果中声明 `resources`，因此它还必须实现强制的 `resources/list` 操作。其确定性列表条目包含规范 URI、稳定的名称、描述和 MIME 类型。列表结果包含 `resultType`、服务器身份元数据、`ttlMs` 和 `cacheScope`，与确定性工具列表一致。

宿主发送 `resources/read`。在 Streamable HTTP 上，请求如下：

```text
POST /mcp
MCP-Protocol-Version: 2026-07-28
Mcp-Method: resources/read
Mcp-Name: ui://notes/timeline.html
```

头部的值与 JSON-RPC 体必须匹配。不匹配是协议错误 `-32020`。

结果包含 HTML 资源和缓存提示：

```json
{
  "resultType": "complete",
  "contents": [
    {
      "uri": "ui://notes/timeline.html",
      "mimeType": "text/html;profile=mcp-app",
      "text": "<!doctype html>...",
      "_meta": {
        "ui": {
          "csp": {
            "connectDomains": [],
            "resourceDomains": [],
            "frameDomains": [],
            "baseUriDomains": []
          },
          "permissions": {}
        }
      }
    }
  ],
  "ttlMs": 60000,
  "cacheScope": "public"
}
```

### 把 UI 资源当作可执行内容缓存

App 资源不能与普通散文互换。它的缓存条目可能执行桥接代码、渲染工具数据，并请求宿主中介的操作。请以规范的 `ui://` URI、被接纳的服务器身份与版本、资源内容摘要，以及 `cacheScope` 为 private 时的授权上下文作为键。绝不要跨主体复用私有 App 资源，因为即使 URI 相同，HTML 或其策略元数据也可能不同。

当 `ttlMs` 过期、工具的 `_meta.ui.resourceUri` 绑定变化、服务器版本或被接纳描述符的固定值变化，或已确认的资源变更订阅点名该 URI 时，使条目失效。重新挂载前要重新获取，并重新应用 CSP 与权限审查。过期的 iframe 不能仅仅因为新资源版本尚未加载就保留更宽的权限。

### 先拒绝线路歧义，再谈功能策略

校验有刻意的顺序。先校验 JSON-RPC 形状，要求字符串协议元数据和对象型客户端能力映射。再比较路由头与请求体。最后才决定匹配到的协议版本是否受支持。这个顺序防止代理和服务器解释出不同的请求。

| 条件 | HTTP | JSON-RPC 错误 |
|-----------|------|----------------|
| 头与体的版本、方法或名称不一致 | 400 | `-32020` |
| 头与体一致，但版本不受支持 | 400 | `-32022`，`data` 恰为 `{"supported":["2026-07-28"],"requested":"<actual>"}` |
| `resources/read` 缺少 Apps 扩展能力 | 400 | `-32021`，带 `data.requiredCapabilities.extensions.io.modelcontextprotocol/ui` |
| 方法未知 | 404 | `-32601` |

JSON-RPC 通知没有 `id`，因此服务器从不为它发出 JSON-RPC 响应。被接受的 HTTP 通知返回 202 与空响应体。错误可以改变 HTTP 状态，但仍不能为通知创建 JSON-RPC 错误体。

### 沙箱是边界，不是信任裁决

宿主控制 iframe。App 无法直接读取宿主 cookie、本地存储或页面 DOM。所有特权工作都必须经过桥接。

使用以下默认值：

- 所有 CSP 域列表保持为空，然后只添加 App 所需的来源。用 `connectDomains` 处理 fetch、XHR 和 WebSocket；用 `resourceDomains` 处理脚本、样式、图片和字体。
- 在可行时打包代码和数据。
- 除非某个可见功能需要，否则不要请求摄像头、麦克风或位置权限。
- 把 `postMessage` 固定到确切的对方来源，拒绝来自其他任何来源的事件。
- 把工具参数、工具结果、资源文本和桥接消息都视为不可信输入。
- 用户同意保留在宿主。iframe 不能批准自身的重大操作。

不要把教程里的固定 `sandbox` 属性照搬到每个宿主。宿主必须根据 App 的来源模型和自身隔离设计选择标志。

被允许的域仍然是一条外泄路径。`connectDomains: ["https://api.example.com"]` 意味着 App 内执行的任何脚本都可以向那里发送被允许的数据。精确来源匹配能防止目标混淆，但不能判断负载是否合适。默认保持连接访问为空，避免把 bearer token 放进 iframe，尽量通过宿主代理窄操作，限制响应和请求大小，并审计每个出站请求由哪个用户操作引起。把 `resourceDomains` 与 `connectDomains` 分开对待；加载字体或脚本的权限不应授予任意数据上传。

### Apps 桥接有自己的生命周期

Apps 桥接是 `postMessage` 上的 JSON-RPC 方言。它可以交换 `ui/initialize` 与 `ui/*` 通知，也可以代理 `tools/call` 这类看起来像核心方法的方法。

View 发送带 `appInfo` 和 `appCapabilities` 对象的 `ui/initialize`。宿主返回自己的能力与宿主上下文。只有在该响应之后，View 才发送 `ui/notifications/initialized`。宿主在向 View 发送消息前必须等待这个 Apps 通知。

这个本地握手在一个 iframe 与一个宿主框架之间建立桥接。它不协商 MCP 协议版本，不创建服务器状态，也不铸造传输会话。注意确切的前缀：核心的 `notifications/initialized` 已被移除，而 Apps 的 `ui/notifications/initialized` 仍然保留。经桥接的工具调用所生成的核心请求，是一个带新 JSON-RPC id 和完整请求元数据的、自包含的新请求。

### 宿主上下文、操作与撤销

桥接初始化之后，宿主仍然是权威。View 只能通过宿主声明过的能力请求工具操作、导航、剪贴板使用或其他特权效果。宿主校验类型化请求、当前用户、目标和参数，执行审批策略，并可以拒绝。按钮点击和合法桥接消息表达意图，二者都不授予权限。

把主题、尺寸和无障碍当作会变化的宿主上下文，而不是一次性渲染输入：

- 应用宿主提供的颜色与排版 token，并在主题或对比度偏好变化时作出反应。
- 允许 View 报告期望尺寸，但由宿主限制并应用 iframe 大小，使内容无法逃出布局或制造欺骗性覆盖层。
- 在 iframe 内保持键盘顺序、可见焦点、无障碍名称、屏幕阅读器状态、足够对比度、缩放和减少动态效果的行为。
- 在缩放和重新渲染后，重新测试宿主控件与 View 控件之间的焦点转移。

能力可能在 App 打开期间被撤销，因为用户切换账号、策略变化、服务器被隔离，或宿主收窄同意。要在操作执行时检查能力与授权，而不只是在 `ui/initialize` 时。撤销时，拒绝待处理的特权调用，停止不再符合策略的网络活动，清除敏感的渲染状态，并在 UI 资源本身不再被接纳时重新挂载或回退到文本。View 必须把拒绝当作普通结果，而不是反复重试直到宿主让步。

### 回退也是契约的一部分

感知 Apps 的服务器仍可服务未声明 UI 扩展的宿主：

- 在 `tools/list` 中返回不带 `_meta.ui` 的同一工具。
- 为 `tools/call` 保留有用的文本结果。
- 对 UI 的 `resources/read` 以缺少能力错误拒绝。
- 决定工具是否完成时，绝不假定 iframe 存在。

```figure
t3-ui-sandbox
```

## 构建它

`code/main.py` 构建了一个不依赖 SDK 的小型进程内协议模型。它校验当前请求信封和 Streamable HTTP 路由值，通过 `server/discover` 声明 Apps，列出工具和资源，执行工具，并提供自包含的 HTML 资源。

该模型接收已解析的请求体和路由头。它不是完整的 HTTP 适配器，也不解析 `Content-Type` 或 `Accept`。完整的 Streamable HTTP 适配器见第 09 课，它要求 `Content-Type: application/json`，且 `Accept` 值同时包含 `application/json` 和 `text/event-stream`。

运行它：

```bash
cd phases/13-tools-and-protocols/14-mcp-apps
python3 code/main.py
python3 -m unittest discover code/tests -v
```

从输出中检查四件事：

1. 每次调用都独立。
2. 每个请求都有 `_meta` 能力。
3. `resources/list` 在任何资源读取之前返回稳定描述符。
4. 每个结果都有 `resultType` 和服务器身份元数据。
5. 不出现任何核心会话标识符。

## 使用它

从 `server/discover` 开始。确认 `io.modelcontextprotocol/ui` 出现在服务器扩展映射中。然后调用两次 `tools/list`，一次带 Apps 能力，一次不带。第一次响应声明该资源。第二次仍是可用的纯文本工具。

读取 `ui://notes/timeline.html`。在 HTML 中搜索 `hostOrigin` 和 `event.origin` 守卫。这两行是最低限度的可见证据，表明桥接没有使用通配目标。

## 交付它

本课交付 `outputs/skill-mcp-apps-spec.md`。写框架代码之前，用它审查 App 契约。它迫使作者说明当前核心信封、扩展协商、回退、UI 资源、缓存策略、CSP、权限、桥接方法和同意边界。

## 练习

1. 把客户端能力改成空扩展映射。确认 `tools/list` 保留工具但移除 UI 绑定。
2. 发送 `Mcp-Name: ui://notes/other.html`，请求体却读取时间线。确认错误 `-32020`。
3. 把资源改成 `cacheScope: private`。描述使其合理的用户特定条件。
4. 把脚本移到 `https://static.example.com/app.js`。把该来源加入 `resourceDomains`，并解释新的供应链风险。
5. 添加 `notes_open` 工具，让按钮点击经由宿主路由。把用户批准保留在宿主。

## 关键术语

| 术语 | 含义 |
|------|---------|
| MCP Apps | 由 MCP 宿主渲染交互式 HTML 的可选扩展 |
| `io.modelcontextprotocol/ui` | 双方共同声明的扩展标识符 |
| `ui://` | App UI 模板的资源方案 |
| `text/html;profile=mcp-app` | MCP App HTML 的 MIME 类型 |
| `server/discover` | 当前用于协议与能力发现的 RPC |
| `resources/list` | 服务器声明资源时强制的资源列表方法 |
| `resultType` | 现代成功结果所需的判别字段 |
| `ui/initialize` | 第一个 Apps 桥接请求，独立于已移除的核心初始化 |
| `ui/notifications/initialized` | 宿主响应后由 Apps View 发送的就绪通知 |
| CSP | 限制脚本、样式、图片和网络来源的浏览器策略 |
| Text fallback | 为不支持 Apps 的宿主保留的工具行为 |

## 延伸阅读

- [MCP 2026-07-28 基础协议](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP Apps 概览](https://modelcontextprotocol.io/extensions/apps/overview)
- [MCP Apps 构建指南](https://modelcontextprotocol.io/extensions/apps/build)
- [官方扩展支持矩阵](https://modelcontextprotocol.io/extensions/client-matrix)
