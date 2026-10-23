# 构建 MCP 客户端：发现、路由与双时代回退

> 现代 MCP 客户端在每个请求上重复自己的约定。它最难的兼容性决定是：判断一个旧服务器是否真的旧，以及一个现代服务器是否在报告一个可纠正的错误。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lesson 07
**Time:** ~85 minutes

## 学习目标

- 用当前的元数据构建每个 MCP `2026-07-28` 请求。
- 用 `server/discover` 探测 stdio 服务器，并选择一个双方都支持的版本。
- 仅对显式加入允许列表的 peer 授权一次有界的遗留探测。
- 只有在为一个受支持的修订版验证到正向的 `initialize` 结果之后，才接受遗留时代。
- 合并确定性的工具列表，且不悄悄覆盖名称冲突。
- 把调用路由到拥有该工具的 peer，而不虚构协议会话。

## 问题

一个智能体宿主通常要与不止一个 MCP 服务器通信。它必须发现每个服务器、合并工具目录、解决重名、路由调用，并从传输故障中恢复。

`2026-07-28` 修订版让稳态更简单，因为每个请求都是自包含的。兼容性则让启动阶段更微妙。客户端可能遇到：

- 支持首选版本的现代服务器；
- 返回可识别的版本错误或头部错误的现代服务器；
- 从未听说过 `server/discover` 的遗留服务器；
- 在收到 `initialize` 之前保持沉默的遗留服务器。

把每次探测错误都当成遗留是危险的。一个畸形的现代请求、一个过载的服务器、一个死掉的进程，以及一个旧服务器，都可能产生相同的超时或连接关闭。这些信号是模糊的。客户端必须在选择遗留时代之前，把运维者的显式意图与正向的协议证据结合起来。

## 概念

### 是 peer，不是协议会话

为每个服务器进程或端点保留一条传输 peer 记录：

- 传输句柄或发送函数；
- 选定的协议时代和版本；
- 最近发现的服务器能力；
- 最近的确定性工具列表；
- 用于关联的待处理请求 id；
- 传输健康状态。

这是客户端的记账。它不是协议会话状态。在现代 MCP 中，服务器仍然会在每个请求上收到当前的版本和能力。

### 从零构建每个现代请求

```python
def modern_request(request_id, method, params, version, capabilities):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {
            **params,
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": version,
                "io.modelcontextprotocol/clientCapabilities": capabilities,
                "io.modelcontextprotocol/clientInfo": CLIENT_INFO,
            },
        },
    }
```

不要只把元数据附加到连接对象上一次，就假设它到达了线上。要在最终序列化的请求上打戳并检查。

### 现代发现

`server/discover` 返回受支持的版本、服务器能力、说明、缓存提示以及推荐的服务器身份。客户端选择双方都支持的最高现代版本。

对于只支持现代的客户端，发现是可选的，但在 stdio 上推荐使用。一些遗留服务器在初始化之前就接受操作，所以先发送 `tools/list` 可能产生一个含义模糊的成功响应。`server/discover` 会划出一条清晰的时代边界。

### stdio 兼容性探测

双时代 stdio 客户端会在任何其他请求之前，用自己的首选现代元数据发送 `server/discover`。结果分为三类：

1. **DiscoverResult。** 服务器是现代服务器。选择一个双方都支持的版本，并继续使用逐请求元数据。
2. **可识别的现代错误。** 服务器是现代服务器。对于 `-32022`，从 `data.supported` 中选择版本并用新的请求 id 重试。对于头部或能力错误，修正请求。不要发送 `initialize`。
3. **模糊信号。** 无法识别的 JSON-RPC 错误、超时、连接关闭或空响应都不能标识时代。除非该 peer 被明确配置为支持遗留兼容，否则按失败关闭（fail closed）。

可识别的现代协议错误包括：

- `-32020` HeaderMismatch
- `-32021` MissingRequiredClientCapability
- `-32022` UnsupportedProtocolVersion

即使该 peer 在遗留允许列表中，可识别的现代错误仍然表明它是现代服务器。一旦服务器证明它理解现代的出错词汇，再发送 `initialize` 就是一种降级。

不要把 `-32601` 当作正向的遗留证据。它只是让一个显式加入允许列表的 peer 有资格接受一次遗留探测。超时、连接关闭或空响应也适用同样的规则。

### 允许列表是运维意图，不是证据

遗留兼容必须是某个固定 peer 配置上的显式属性：

```python
client.add_server("archive", archive_transport, allow_legacy=True)
```

把这一选择绑定到已配置的命令或端点。不要使用通配符，让任意服务器把自己选入更弱的语义。没有 `allow_legacy=True` 的 peer 在发现结果模糊之后就会失败，也永远不会收到 `initialize`。

允许列表授予的是探测许可。它并不选择时代。客户端在传输强制的截止时间内发送一次 `initialize`，然后要求以下全部条件成立：

- 带有匹配请求 id 的 JSON-RPC `2.0` 响应；
- 恰好一个 `result`，且没有 `error`；
- `protocolVersion` 位于客户端配置的遗留修订版集合中；
- `capabilities` 字段为对象值；
- `serverInfo` 对象中的 `name` 和 `version` 字段均为非空字符串。

超时、连接关闭、错误响应、畸形结果、id 不匹配或不支持的修订版都会按失败关闭处理。只有结构上有效的正向结果才会选定遗留时代。代码把 `legacy_probe_timeout_ms` 传给传输适配器；真实的 stdio 或 HTTP 适配器必须强制执行该截止时间，而不只是记录它。

为传输 peer 缓存选定的时代。不要在每次调用之前重新探测。

### 遗留是一条兼容分支

一旦有界探测返回有效的正向遗留证据，客户端就完全按该修订版的定义使用选定的遗留版本：

1. 验证响应信封和关联 id。
2. 验证协商出的修订版在配置的遗留集合中。
3. 记录已验证的能力和服务器身份。
4. 只有在所有检查通过之后才发送 `notifications/initialized`。
5. 在该传输生命周期内使用遗留请求形态。

这条分支是为了与已知 peer 互操作而存在。它不是新服务器或新请求的默认设计。如果传输重启或其端点变化，丢弃 peer 时代缓存并重新协商。

### 发现并缓存工具

对每个活跃的 peer 调用 `tools/list`。现代结果包含 `resultType`、`ttlMs` 和 `cacheScope`。在正确的授权上下文中遵守新鲜度提示。过期之后或收到已订阅的列表变更事件之后，重新获取。

客户端必须把遗留服务器缺失的 `resultType` 视为 `"complete"`。不要对来自更早协商时代的响应要求现代缓存字段。

服务器应返回确定性的顺序。客户端也应在合并之前排序，这样本地注册表的顺序就不会依赖进程启动时机。

### 冲突安全的命名空间合并

两个服务器可能都暴露 `search`。请选择一条明确声明的策略：

1. **冲突时加前缀。** 保留第一个规范名，把后续冲突暴露为 `<server>/<tool>`。
2. **冲突时拒绝。** 不加载重复项，并给出清晰的配置错误。
3. **静默覆盖。** 永远不要使用。它掩盖了模型选定的动作最终由哪个服务器接收。

同时保存规范名和本地名。模型看到的是规范名。发出的 `tools/call` 使用拥有该工具的服务器所声明的本地名。

### 路由一次调用

路由是一次纯查找：

```text
canonical tool name
  -> peer name + local tool name
  -> new JSON-RPC request id
  -> modern request metadata or explicit legacy shape
  -> matching response id
```

当拥有某个工具的传输不可用时，不要发送该调用。重连或重启传输，然后重新执行发现和 `tools/list`。在传输中断时丢失的现代在途请求，可以在操作的安全策略允许时用新的 JSON-RPC id 重试。

### 通知与订阅

现代的列表变更和资源变更只通过客户端打开的 `subscriptions/listen` 流到达。客户端发送通知过滤器，等待 `notifications/subscriptions/acknowledged`，并用通知元数据中的 listen 请求 id 关联事件。

断开连接时，打开一个新的 listen 请求，并重新获取相关列表或资源。现代流不会用 `Last-Event-ID` 续传。

### 没有服务器发起的请求

现代服务器不会用独立的 JSON-RPC 请求来调用客户端进行 sampling、elicitation 或 roots 交互。它们返回 `input_required`，客户端在完成内嵌的输入请求之后重试原始请求。

在完成输入的过程中，不要阻塞该 peer 的响应读取器。保持关联性，并为重试创建新的 JSON-RPC id。

```figure
tp-client-merge
```

## 使用它

`code/main.py` 使用进程内的 peer 函数，以便让协议决策保持可见。它连接两个现代 peer 和一个刻意加入允许列表的遗留 peer，然后合并并路由它们的工具。传输可调用对象会接收一个超时预算，因此兼容分支无法隐藏一次无界探测。

```bash
cd code
python3 main.py
python3 -m unittest discover tests -v
```

这些测试证明了普通演示会遗漏的边界：

- 现代请求重复元数据；
- `-32022` 在不初始化的情况下重试现代发现；
- 可识别的现代错误永不降级，即使对允许列表中的 peer 也是如此；
- 没有允许列表时，超时、连接关闭、空响应和无法识别的错误都不会触发 `initialize`；
- 允许列表中的 peer 只有在得到有效且受支持的 `initialize` 结果之后才会成为遗留 peer；
- 畸形和不支持的遗留结果会让该 peer 保持不可用；
- 成功选定的时代会在该传输生命周期内被缓存。

## 交付它

本课交付 `outputs/skill-mcp-client-harness.md`。它为现代请求打戳、stdio 时代协商、确定性命名空间合并、路由，以及按失败关闭的遗留兼容分支搭建骨架。

## 练习

1. 让一个假服务器返回 `-32022`，且没有双方都支持的版本。确认客户端失败，而不是发送 `initialize`。
2. 把一个假的遗留服务器加入允许列表，让它的有界 `initialize` 探测超时，并证明该 peer 保持 `unknown` 且不可用。
3. 为两个授权上下文添加 `cacheScope: "private"` 的工具列表。确认客户端从不把一个上下文的缓存结果共享给另一个。
4. 把冲突策略改为拒绝，并让启动失败，且错误中包含两个 peer 的名字。
5. 添加一个有界的 `subscriptions/listen` 模拟器。流丢失时，用新的请求 id 重新监听并重新获取工具。

## 关键术语

| 术语 | 含义 |
|------|---------|
| Peer | 针对一个服务器传输及其已发现数据的客户端记录 |
| 协议时代（Protocol era） | 现代的逐请求元数据，或遗留的初始化语义 |
| 发现探测（Discovery probe） | 用于识别 stdio 时代的初始 `server/discover` |
| 可识别的现代错误 | 证明现代行为并禁止遗留回退的错误 |
| 遗留允许列表 | 运维者配置，允许对一个固定的 peer 进行一次有界的兼容性探测 |
| 正向遗留证据 | 对一个显式支持的遗留修订版得到的有效且已关联的 `initialize` 结果 |
| 合并命名空间 | 所有活跃 peer 的规范工具名 |
| 冲突策略 | 针对重复工具名的加前缀或拒绝规则 |
| 时代缓存 | 为一个传输 peer 保存的选定现代或遗留行为 |
| 传输恢复 | 重启或重连、重新发现、重新列举，并用新 id 安全重试 |

## 延伸阅读

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP Versioning](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
