# 构建 MCP 服务器：无状态的 Python 与 TypeScript

> 现代 MCP 服务器不记得握手。它对每个请求验证元数据，运行一个处理器，并返回一个带类型的结果。

**Type:** Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 13, Lesson 06
**Time:** ~85 minutes

## 学习目标

- 为 MCP `2026-07-28` 实现必需的 `server/discover`。
- 在每个请求上验证协议版本和客户端能力。
- 以确定性的列表顺序暴露工具、资源和提示。
- 在正确的结果上返回 `resultType`、服务器身份和缓存提示。
- 在 Python 和 TypeScript 中通过按行分隔的 stdio 提供同一套无状态约定。

## 问题

一个在首条消息之后保存客户端能力的服务器很容易构建，却很难运维。同一个进程可能依次服务多个客户端。一个远程请求可能落到不同的 worker 上。一份过时的能力声明可能让行为跨越授权边界泄漏。

MCP `2026-07-28` 通过让每个请求自描述，解决了该问题中协议层面的部分。你的应用仍然可以保存持久化的笔记、任务或显式状态句柄。它不能保存的是隐藏的协议状态——那种会改变后续请求解码方式的协议状态。

本课把同一个笔记服务器构建两遍。Python 和 TypeScript 版本在协议核心上都只使用各自的标准库。两者暴露相同的方法，并执行相同的线上约定（wire contract）。

## 概念

### 现代分发循环

```text
read one JSON-RPC line
parse the envelope
if it is a notification, do not respond
validate params._meta for this request
route by method
wrap success with resultType and serverInfo
write one JSON-RPC response line
forget request-scoped metadata
```

三条 stdio 规则仍然重要：

- 只向 stdout 写 JSON-RPC 消息。把诊断信息写到 stderr。
- 用换行分隔消息，并刷新每个响应。
- 当 stdin 到达 EOF 时立即退出。

进程的生命周期就是传输的生命周期。它不是一个现代 MCP 会话。

### 请求验证

每个请求都必须包含：

```json
{
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "notes-client",
        "version": "1.0.0"
      }
    }
  }
}
```

前两个字段是必需的。`clientInfo` 是推荐项。要验证出现的身份结构，但不要把它当作认证。

如果版本不受支持，返回错误码 `-32022`，并附带 `requested` 和 `supported`。请求元数据缺失属于无效参数，错误码为 `-32602`。绝不要用上一次调用的字段填补缺失字段。

### 必需的发现

现代服务器必须实现 `server/discover`。完整的发现结果包括受支持的现代版本、能力、可选的说明（instructions）、缓存提示，以及结果 `_meta` 中的服务器身份：

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": false},
    "resources": {"listChanged": false, "subscribe": false},
    "prompts": {"listChanged": false}
  },
  "ttlMs": 3600000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

发现并不会「解锁」服务器。客户端可以不调用发现就直接调用 `tools/list`，因为 `tools/list` 本身就携带了相同的请求元数据。

### 工具

`tools/list` 返回确定性的工具描述符列表。稳定的顺序有助于响应缓存，并让模型上下文保持稳定。结果还要求包含 `ttlMs` 和 `cacheScope`。

`tools/call` 返回内容块和 `isError`。当协议信封或方法参数无效时，使用 JSON-RPC 错误。当工具调用本身有效、但工具自身执行失败时，使用 `isError: true`。

工具注解始终只是提示，不是强制执行：

- `readOnlyHint`
- `destructiveHint`
- `idempotentHint`
- `openWorldHint`

宿主应当用它们来做确认和呈现。服务器仍然必须执行真正的授权。

### 资源

`resources/list` 返回稳定的 URI 描述符。`resources/read` 返回带类型的内容。在 `2026-07-28` 中两者都可缓存，因此都包含 `ttlMs` 和 `cacheScope`。

对用户专属的笔记数据使用 `cacheScope: "private"`。共享缓存不得跨授权上下文复用私有响应。

现代的变更投递不使用 `resources/subscribe`。客户端打开 `subscriptions/listen`，请求 `resourceSubscriptions` 或列表变更类别。第 10 课会构建这一流程。

### 提示

`prompts/list` 可缓存且是确定性的。`prompts/get` 用参数渲染一个具名提示。渲染出的提示结果是完整的，但它不属于需要缓存提示的可缓存列表或读取结果。

### 每个成功结果都是带类型的

示例对每个成功结果使用同一个包装器：

```python
def complete(payload):
    return {
        "resultType": "complete",
        **payload,
        "_meta": {SERVER_INFO_KEY: SERVER_INFO},
    }
```

列表、读取和发现处理器会额外加上 `ttlMs` 和 `cacheScope`。集中使用这个包装器，可以防止某个处理器悄悄漏掉现代结果字段。

### 没有服务器发起的请求

现代服务器可以发送与客户端请求相关的通知，或者在客户端打开的 `subscriptions/listen` 流上发送通知。它不得发送自己的 JSON-RPC 请求。

当处理器需要 sampling、elicitation 或 roots 输入时，它返回一个 `input_required` 结果。客户端完成内嵌的输入请求，然后用新的请求 id 重试原始方法。第 11 课会讲解这种多轮往返请求（Multi Round-Trip Request，MRTR）模式。

### 显式的遗留兼容

双时代（dual-era）服务器也可以在明确分离的遗留分支上实现 `2025-11-25` 握手。当必需的现代 `_meta` 字段存在时，它选择现代行为；当收到 `initialize` 时，它选择遗留行为。

不要把 `2026-07-28` 请求送进遗留握手路径。不要把现代 `resultType` 字段盖到遗留初始化结果上。本课的代码刻意只支持现代版本，以便让它的不变量保持可见。

```figure
t3-dispatch-loop
```

## 使用它

运行 Python 服务器的有限演示和测试：

```bash
cd code
python3 main.py --demo
python3 -m unittest discover tests -v
```

用 TypeScript 运行器运行 TypeScript 版本：

```bash
npx tsx main.ts --demo
```

该演示会发送 `server/discover`，列出每种原语（primitive），调用工具，并展示一个版本不受支持的错误。每个现代请求都会重复元数据。每个成功结果都包含服务器身份。

## 交付它

本课交付 `outputs/skill-mcp-server-scaffolder.md`。它会产出一份现代服务器方案，包含发现约定、逐请求验证、确定性可缓存列表，以及可选的隔离遗留适配器。

## 练习

1. 从一个请求中移除能力，证明服务器不会复用上一个请求的声明。
2. 反转 `TOOLS`、`PROMPTS` 和笔记的插入顺序。确认所有列表结果仍然稳定。
3. 添加一个具有破坏性的 `notes_delete` 工具，并在执行器内部要求授权检查。让 `destructiveHint` 仅作为 UX 提示。
4. 添加 `resources/templates/list`，带上 `ttlMs`、`cacheScope` 和确定性排序。
5. 为 `2025-11-25` 构建一个独立的遗留适配器。添加测试，证明现代请求永远不会进入该适配器。

## 关键术语

| 术语 | 含义 |
|------|---------|
| 无状态服务器（Stateless server） | 仅凭每个请求自身的元数据处理请求，不依赖协议会话记忆 |
| `server/discover` | 必需的现代方法，用于公布版本和能力 |
| 完整结果（Complete result） | 带 `resultType: "complete"` 的现代成功结果 |
| 可缓存结果（Cacheable result） | 带 `ttlMs` 和 `cacheScope` 的发现、列表或资源读取结果 |
| 确定性列表（Deterministic list） | 相同的逻辑注册表产生相同的条目顺序 |
| 服务器身份（Server identity） | 结果 `_meta` 中推荐的 `io.modelcontextprotocol/serverInfo` |
| 工具错误（Tool error） | 有效的工具调用返回带 `isError: true` 的内容 |
| 协议错误（Protocol error） | 无效的 JSON-RPC 或 MCP 请求，通过 `error` 返回 |

## 延伸阅读

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
