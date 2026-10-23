# MCP 授权：CIMD、签发者绑定、PKCE 与增量授权

> 远程 MCP 请求是无状态的，但它的授权不是匿名的。把每份凭据绑定到签发它的签发者，把每个 token 绑定到接收它的资源。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 15 (security)
**Time:** ~90 minutes

## 学习目标

- 通过受保护资源元数据发现授权服务器。
- 优先使用 Client ID Metadata Documents，而不是已弃用的 Dynamic Client Registration。
- 当 DCR 兼容路径不可避免时，声明正确的 `application_type`。
- 校验授权响应中的 `iss`，并按签发者隔离凭据。
- 使用 PKCE、资源指示符、受众校验和增量 scope。
- 在不使用协议会话的情况下发送经过授权的 MCP 2026-07-28 请求。

## 问题

远程 MCP 服务器可能读取私有记录、写入外部系统，或触发昂贵工作。认证告诉它是谁出示了凭据。授权还必须回答：

- 哪台授权服务器签发了这份凭据？
- 这个 token 是给哪个 MCP 资源的？
- 哪个客户端和重定向 URI 完成了流程？
- 用户批准了哪些操作？
- 这个确切的请求是否仍然符合那次批准？

2026-07-28 授权配置强化了客户端注册和签发者处理。它优先使用 Client ID Metadata Documents，弃用 Dynamic Client Registration，要求在 DCR 上使用正确的 `application_type`，校验 RFC 9207 签发者响应，并禁止跨签发者复用凭据。

这些规则补充无状态核心，并不恢复核心握手或 `Mcp-Session-Id`。

## 概念

### 认识三个角色

- **MCP 客户端：** 代表资源所有者发送请求。
- **MCP 资源服务器：** 接受访问令牌并提供 MCP 端点。
- **授权服务器：** 认证资源所有者、收集同意并签发 token。

资源服务器和授权服务器可以一起运营，但要把它们的标识符和校验职责分开。

### 授权适用于 HTTP

MCP 授权规范适用于基于 HTTP 的传输。本地 stdio 服务器运行在进程和操作系统信任边界之内。不要仅仅为了对称，就给 stdio 加一个虚假的浏览器 OAuth 流程。

对远程 Streamable HTTP，在每个请求的 `Authorization` 头里发送 bearer token。绝不要把它放进 URL。

### 从受保护资源元数据开始

资源服务器发布 RFC 9728 元数据：

```json
{
  "resource": "https://notes.example.com/mcp",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:delete", "notes:read", "notes:write"]
}
```

客户端从 MCP 资源 URL 开始，获取该文档，选择一个声明的授权服务器，然后获取该服务器的 OAuth 或 OpenID Connect 元数据。

构造 RFC 9728 well-known URL 时保留资源路径。对资源 `https://notes.example.com/mcp`，本课使用 `https://notes.example.com/.well-known/oauth-protected-resource/mcp`。丢掉 `/mcp` 后缀可能选中同一来源上另一个受保护资源的元数据。

不要从主机名猜授权服务器。不要跟随从未经验证错误体里发现的签发者。维护一份客户端愿意信任哪些签发者的策略。

### 校验授权服务器元数据

元数据应暴露端点与支持的控制项：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "code_challenge_methods_supported": ["S256"],
  "authorization_response_iss_parameter_supported": true,
  "client_id_metadata_document_supported": true
}
```

PKCE 要求 S256。记录确切的签发者字符串。该确切值成为注册和令牌存储的键。

### 遵循注册优先级

当客户端已与所选签发者存在明确关系时，使用预注册的客户端信息。否则，在授权服务器声明支持时优先使用 Client ID Metadata Documents。仅把 DCR 当作已弃用的兼容回退；如果这些机制都不可用，再提示输入客户端信息。

### 优先使用 Client ID Metadata Documents

Client ID Metadata Document 给授权服务器一个 HTTPS URL，它既是客户端标识符，也是其元数据的位置：

```json
{
  "client_id": "https://client.example.com/oauth/metadata.json",
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

授权服务器获取并校验该文档。`client_id` 必须是带路径的 HTTPS URL，且文档内的值必须与该 URL 完全相等。必需文档字段是 `client_id`、`client_name` 和 `redirect_uris`。`application_type` 出现在本例中，但不是 CIMD 的要求。它新的强制用途特指 DCR 路径。

把获取该文档视为对 SSRF 敏感的操作。解析并校验目标，拒绝回环、私有、链路本地和其他不允许的地址，在重定向和 DNS 变化后重新检查，限制重定向次数、字节数和时间，要求 JSON，并且只依据经过校验的 HTTP 缓存控制进行缓存。把 `client_name` 和其他显示字段视为不可信文本。

CIMD 免去了为每次首次接触铸造新动态标识符的需要。它并没有移除重定向 URI 校验、签发者策略或用户同意。

### DCR 是兼容路径

Dynamic Client Registration 仍可供较旧的授权服务器使用，但对新的 MCP 实现已弃用。

使用 DCR 时，声明 `application_type`：

```json
{
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

- 桌面、移动、命令行和回环客户端使用 `native`。
- 远程托管的浏览器应用使用 `web` 和远程 HTTPS 重定向。

省略该字段可能在某个 OpenID Connect 注册实现中默认为 `web`，使合法的回环重定向失败。

把 DCR 代码放在显式回退决定之后。不要在任意 CIMD 校验失败后静默回退。那可能把安全失败变成更弱的注册路径。

### 把凭据绑定到签发者

按确切签发者存储由签发者铸造的注册材料：

```text
issuer_credentials[issuer] = pre_registered_or_dcr_client
tokens[(issuer, resource)] = access_token
```

如果受保护资源发现从 `https://auth-one.example` 变为 `https://auth-two.example`，重新评估信任。绝不把第一个签发者的客户端密钥、DCR 客户端 id、注册访问令牌、刷新令牌或访问令牌发给第二个。预注册客户端和 DCR 客户端必须使用为新签发者签发的凭据。

CIMD 客户端 id 不同，因为它是自托管的 HTTPS URL，而不是授权服务器铸造的凭据。同一个 CIMD URL 是可移植的：新的受信任签发者会获取并校验该文档，无需 DCR 重新注册。授权响应和 token 仍要在新签发者下校验和存储。

### 带 PKCE 的授权码

交互流程是：

1. 生成高熵 `code_verifier`。
2. 推导 S256 `code_challenge`。
3. 用确切的 `client_id`、`redirect_uri`、`scope`、`code_challenge` 和 `resource` 发送授权请求。
4. 接收包含 `code` 的授权响应，以及（若提供）`iss`。
5. 在使用任何响应字段之前，把 `iss` 与记录的确切签发者比较。
6. 用 `code_verifier`、相同的重定向 URI 和相同的 `resource` 交换授权码。
7. 把得到的 token 存到 `(issuer, resource)` 下。

RFC 8707 的 `resource` 参数同时出现在授权请求和令牌请求中。它标识规范的 MCP 服务器 URI。

### 精确校验 `iss`

RFC 9207 防止来自一个签发者的授权响应与另一个签发者的响应混淆。

当 `iss` 存在时，把它与记录的签发者比较，不做大小写折叠、尾部斜杠变更、默认端口移除或百分号编码规范化。不匹配时，不要对授权码采取行动，甚至不要显示来自该响应的、由攻击者控制的错误细节。

包含 `iss` 的授权服务器会声明 `authorization_response_iss_parameter_supported: true`。即使缺少该声明，当前客户端仍要校验存在的 `iss`。

### 在 MCP 服务器校验受众

资源服务器只接受为自身签发的 token：

```text
token.issuer == configured_authorization_server
token.audience == canonical_mcp_resource
```

无效、过期、签发者错误或受众错误的 token 收到 401。MCP 服务器不得接受或传递面向其他服务的 token。

### 请求最小的当前 scope

从当前需要的 scope 开始。如果后续工具需要更多，服务器返回 403，并附带权威的 scope 质询：

```text
WWW-Authenticate: Bearer error="insufficient_scope",
  scope="notes:delete",
  resource_metadata="https://notes.example.com/.well-known/oauth-protected-resource/mcp"
```

客户端解释新权限，取得同意，用合并后的 scope 集合执行新的授权流程，并用新的 JSON-RPC id 重试 MCP 请求。

不要假定质询中的 scope 是 `scopes_supported` 的子集。该质询对当前操作具有权威性。

### 授权与无状态 MCP 线路

经过授权的工具调用仍然携带完整的当前请求信封：

```text
POST /mcp
Authorization: Bearer <access-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.delete
```

```json
{
  "jsonrpc": "2.0",
  "id": 12,
  "method": "tools/call",
  "params": {
    "name": "notes.delete",
    "arguments": {"id": "note-7"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "oauth-lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

token 授权主体。请求元数据协商协议行为。两者不能互相替代。

按固定顺序校验线路：JSON-RPC 与元数据类型，头与体相等性，然后协议支持。路由或版本头不匹配返回 HTTP 400 和 `-32020`。如果头和体一致地指向不支持的版本，返回 HTTP 400 和 `-32022`，且 `data` 恰好为 `{"supported":["2026-07-28"],"requested":"<actual>"}`。未知方法返回 HTTP 404 和 `-32601`。

每个请求错误，包括 401 无效 token 和 403 scope 不足，都是带原始请求 `id` 的 JSON-RPC 错误信封。结构化恢复信息放在可选的错误 `data` 中；`WWW-Authenticate` 仍是 HTTP 响应头。通知没有 `id`，因此不接收 JSON-RPC 体。被接受的 HTTP 通知返回 202 与空响应体。

服务器实现 `server/discover` 并声明 tools，因此它还必须实现强制的 `tools/list` 方法。其工具描述符具有稳定名称、描述和对象根 `inputSchema`。列表是确定性的，并返回 `resultType`、服务器身份元数据、有界的 `ttlMs` 和 `cacheScope`。发现结果和与用户无关的工具列表可以在授权之前提供。如果其中任一因主体而异，应用常规策略和私有缓存。

### 不做令牌透传

MCP 服务器不得把客户端的 MCP 访问令牌转发给下游 API。用正确的受众单独获取下游 token，或使用显式的令牌交换设计。只有当服务拒绝为他人铸造的 token 时，受众校验才有效。

### 刷新令牌

刷新令牌是可选的。签发时，要保密存储，并以签发者和资源为键。不要假定它们一定存在。当授权服务器支持轮换时轮换它们，并检测已失效值的重用。

```figure
t3-scope-stepup
```

## 构建它

`code/main.py` 是一个进程内协议与授权模拟器。它实现受保护资源发现、授权服务器元数据、CIMD 注册、按版本门控的 DCR 回退、应用类型检查、PKCE、签发者校验、资源绑定 token、scope 增量、`server/discover`、`tools/list` 以及无状态工具请求。

模型接收已解析的请求体和路由头。它不是完整的 HTTP 适配器，也不解析 `Content-Type` 或 `Accept`。把它接到第 09 课的 Streamable HTTP 适配器，后者要求 `Content-Type: application/json`，且 `Accept` 值同时包含 `application/json` 和 `text/event-stream`。

运行它：

```bash
cd phases/13-tools-and-protocols/16-mcp-security-oauth-2-1
python3 code/main.py
python3 -m unittest discover code/tests -v
```

输出依次展示发现、CIMD 注册、一次普通读取、两次独立的 scope 增量，以及按签发者键控的凭据存储。

## 使用它

把模拟器对象映射到生产组件：

- `ResourceServer.protected_resource_metadata` 变为 RFC 9728 端点。
- `AuthorizationServer.metadata` 变为 RFC 8414 或 OpenID Connect 发现。
- `Client.enroll` 变为 CIMD 解析加一条显式 DCR 兼容分支。
- 由签发者铸造的客户端凭据和 `tokens_by_issuer_resource` 变为加密记录。CIMD URL 可以保持可移植，而其授权结果仍然绑定签发者。
- `ResourceServer.handle` 变为中间件，在分派前校验当前 MCP 头、token 和工具 scope，同时让每个请求错误都落在匹配的 JSON-RPC 信封中。

## 交付它

本课交付 `outputs/skill-oauth-scope-planner.md`。它现在设计注册优先级、按签发者绑定的凭据存储、应用类型、PKCE、资源指示符、scope 质询和当前无状态请求边界。

## 练习

1. 增加刷新令牌轮换，并拒绝重用上一个刷新令牌。
2. 增加签发者允许列表。签发者变化时，只复用可移植的 CIMD URL；拒绝所有先前由签发者铸造的凭据和 token。
3. 给授权码加过期时间，确认迟到的交换会失败。
4. 构建一个带远程 HTTPS 重定向的 Web 客户端变体，并将其 DCR 元数据与原生客户端比较。
5. 在同一签发者下增加第二个资源。确认其访问令牌不能在第一个资源使用。

## 关键术语

| 术语 | 含义 |
|------|---------|
| Protected-resource metadata | 标识资源与授权服务器的 RFC 9728 文档 |
| CIMD | URL 即 OAuth 客户端标识符的 HTTPS 元数据文档 |
| DCR | 为兼容性保留的、已弃用的动态客户端注册 |
| `application_type` | `native` 或 `web`，用于校验重定向 URI 规则 |
| PKCE | 保护被截获授权码的 verifier 与 S256 challenge |
| `iss` | RFC 9207 授权响应签发者标识符 |
| Resource indicator | RFC 8707 参数，把令牌请求绑定到 MCP 资源 |
| Audience | token 对其有效的资源 |
| Step-up | 为额外的当前操作 scope 进行新的同意和令牌签发 |
| Issuer-bound credentials | 按确切授权服务器签发者隔离的注册与令牌记录 |

## 延伸阅读

- [MCP 2026-07-28 授权规范](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)
- [RFC 9728：OAuth 2.0 受保护资源元数据](https://www.rfc-editor.org/rfc/rfc9728)
- [RFC 8707：OAuth 2.0 资源指示符](https://www.rfc-editor.org/rfc/rfc8707)
- [RFC 9207：OAuth 2.0 授权服务器签发者标识](https://www.rfc-editor.org/rfc/rfc9207)
- [OAuth Client ID Metadata Document 草案](https://datatracker.ietf.org/doc/draft-ietf-oauth-client-id-metadata-document/)
