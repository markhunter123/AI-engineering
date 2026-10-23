# 生产环境中的 MCP 认证：按签发方绑定注册与 token

> 第 16 课构建了 OAuth 2.1 状态机。本课为 MCP 2026-07-28 强化其生产边界：优先使用 Client ID Metadata Documents，动态注册已弃用、仅用于兼容；校验授权响应中的签发方；客户端凭据按签发方分键；刷新 JWKS；并在每个无状态请求上使用按受众固定的 token。
>
> **规范说明（2026-07-28）：** 动态客户端注册（Dynamic Client Registration）已被弃用，取而代之的是 Client ID Metadata Documents。DCR 仍作为一种兼容机制保留。使用它时，客户端必须声明正确的 `application_type`。客户端会校验存在的 RFC 9207 `iss` 值，并且绝不在不同授权服务器签发方之间复用凭据。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 16 (OAuth 2.1 state machine), Phase 13 · 17 (gateways)
**Time:** ~90 minutes

## 学习目标

- 通过 RFC 8414 元数据发现授权服务器，并核验其契约。
- 通过 Client ID Metadata Document 完成注册，并把已弃用的 DCR 隔离为回退方案。
- 校验 RFC 9207 `iss`，按授权服务器签发方为注册分键，并按「签发方 + 资源」为绑定资源的 token 分键。
- 按计划缓存并刷新 JWKS 密钥，使签名验证能够在密钥轮换后继续工作。
- 使用 RFC 8707 资源指示符把 token 固定到单个 MCP 资源，并拒绝「混淆代理」式的复用。
- 选择 JWT 校验或 token 内省，定义撤销新鲜度，并在身份依赖不可用时安全失败。
- 把授权服务器、资源服务器和客户端分离，使各自只执行自己负责的检查。
- 按部署清单审计授权服务器，并拒绝不安全的注册或 token 复用。

## 问题

第 16 课的模拟器在内存中运行 OAuth 2.1。生产环境有三个纯内存模拟器看不到的运维缺口。

第一个缺口是注册与凭据隔离。真实组织可能运行数百个 MCP 服务器和数千个 MCP 客户端。2026-07-28 修订版优先采用 **Client ID Metadata Document**：客户端使用一个由自己掌控的、带路径的 HTTPS URL 作为标识符，授权服务器主动拉取该元数据。RFC 7591 动态注册仅作为已弃用的兼容路径保留。当 DCR 无法避免时，请求必须声明正确的 `application_type`。客户端按授权服务器签发方存储注册信息，按 `(issuer, resource)` 对存储访问 token。签发方改变意味着需要重新注册；资源不同则意味着需要一个单独绑定受众的 token。

第二个缺口是密钥轮换。JWT 校验依赖授权服务器的签名密钥，这些密钥以 JSON Web Key Set（JWKS）形式发布。授权服务器按计划轮换密钥（通常是每小时，事件响应期间有时更快）。一个只在启动时拉取一次 JWKS 的 MCP 服务器在轮换窗口到来之前一切正常——之后每个请求都会失败，直到重启。生产环境把 JWKS 接线为带刷新任务的缓存值：刷新任务在旧密钥过期之前覆盖缓存；同时在缓存未命中时回退拉取一次，以应对「token 由比缓存更新的密钥签名」的情况。

第三个缺口是受众绑定。第 16 课引入了 RFC 8707 资源指示符。在生产环境中，该指示符会变成每个请求上的一项硬性声明检查。MCP 服务器把 `token.aud` 与自身的规范资源 URL 比对，不匹配时以 HTTP 401 拒绝。这是防范上游 MCP 服务器（或持有本应发给某一服务器的 token 的恶意客户端）在同一信任网格中的另一服务器上重放该 token 的唯一手段。

本课把每个缺口映射到接口面上的一个具体部件。元数据文档是一个 HTTP 端点。JWKS 缓存刷新是一个定时任务加一个键值缓存。JWT 校验是资源服务器在分发任何工具之前运行的例程。让三个角色保持分离，每个角色只执行自己负责的检查：授权服务器签发并轮换密钥，资源服务器缓存并校验，客户端发现并注册。

## 范围：第 16 课之后的生产执行

[第 16 课：基于 OAuth 2.1 的 MCP 安全](../../16-mcp-security-oauth-2-1/docs/en.md) 负责授权码状态机、PKCE、受保护资源发现、资源指示符和 scope 决策。本课不再定义第二套 OAuth 流程。它从这些契约已经存在之处开始，追问一个已部署的资源服务器如何在密钥轮换、不透明 token 校验、撤销、依赖故障、发布上线和事件响应期间持续执行它们。

生产边界更窄，也更偏运维：

- JWT 路径在每个请求上校验固定的签发方、算法、签名密钥、受众、时间声明和 scope，同时安全地刷新 JWKS。
- 不透明 token 路径调用签发方经过认证的内省端点，并校验返回的 active 状态、受众或资源、过期时间、主体和 scope。
- 撤销策略定义凭据必须在多快内停止生效，以及哪个缓存可能推迟这一事实。
- 失败策略决定当发现、JWKS、内省或撤销基础设施不可用时该怎么办。
- 证据记录是哪些签发方元数据、密钥集或内省响应、token 声明、策略版本和拒绝原因驱动了结果，且不存储 token 本身。

这种区分让各课保持可组合。第 16 课证明流程正确；第 18 课证明 token 到达真实 MCP 请求路径之后仍然可信，或者会被拒绝。

## 概念

### RFC 8414 —— OAuth 授权服务器元数据

位于 `/.well-known/oauth-authorization-server` 的文档描述了客户端所需的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "client_id_metadata_document_supported": true,
  "registration_endpoint": "https://auth.example.com/register",
  "authorization_response_iss_parameter_supported": true,
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

拿到 MCP 资源 URL 的客户端会串联发现过程：先由 RFC 9728 的 `oauth-protected-resource`（资源服务器的文档）给出签发方，再由 `oauth-authorization-server`（本 RFC）给出每个端点。客户端永远不会硬编码授权 URL。

对于带路径的资源标识符，要把 well-known 段插入到该路径之前。例如，`https://mcp.example.com/team/server` 的受保护资源元数据位于 `https://mcp.example.com/.well-known/oauth-protected-resource/team/server`。把 `/.well-known/...` 追加在资源路径之后是错误的。

在为 MCP 信任某个 IdP 之前，需要核验的契约：

- `code_challenge_methods_supported` 包含 `S256`（按 RFC 7636 的 PKCE）。规范明确指出：如果该字段**缺失**，授权服务器就不支持 PKCE，客户端**必须**拒绝继续。
- `grant_types_supported` 包含 `authorization_code`，且拒绝 `password` 和 `implicit`。
- 至少有一条可用注册路径：`client_id_metadata_document_supported: true`（CIMD，首选）、预先注册的客户端，或 `registration_endpoint`（已弃用的 RFC 7591 兼容路径）。
- 如果 `authorization_response_iss_parameter_supported` 为 true，客户端就要求返回 RFC 9207 `iss`，并将其与重定向之前记录的签发方精确比较。
- 对 OAuth 2.1 而言，`response_types_supported` 必须恰好是 `["code"]`。

如果缺少 `S256`，MCP 服务器拒绝针对该 IdP 部署——PKCE 没有降级模式。如果*两种*注册路径都没有宣告，且你也没有预先注册的 `client_id`，同样无法注册；此时错的是部署清单，而不是代码。

### RFC 9728（回顾）——受保护资源元数据

第 16 课讲过 RFC 9728。生产环境中的增量在于：客户端只在这一份文档中查找*这个* MCP 服务器所信任的授权服务器。单个 MCP 服务器可以接受来自多个 IdP 的 token（一个面向员工，一个面向合作伙伴）。RFC 9728 声明这个集合；RFC 8414 描述每个 IdP 支持什么。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Client ID Metadata Documents（推荐的默认方式）

CIMD 把注册从*推送*反转为*拉取*。客户端不再请求授权服务器生成 `client_id`，而是把自己掌控的一个 HTTPS URL **作为** `client_id`。该 URL 解析到一份 JSON 元数据文档；授权服务器在 OAuth 流程中按需拉取它。信任根植于 DNS：如果服务器运维方信任 `app.example.com`，它也就信任由 `https://app.example.com/client.json` 提供的客户端。没有注册往返，没有会耗尽的 `client_id` 命名空间，也没有需要保持同步的每服务器状态。

客户端托管的元数据文档：

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

文档中的 `client_id` 值**必须**等于它的托管 URL（授权服务器会验证这一点；不一致会被拒绝）。授权服务器在其 RFC 8414 元数据中用 `client_id_metadata_document_supported: true` 宣告支持。

按当前 CIMD 契约，`client_id`、`client_name` 和非空的 `redirect_uris` 数组是必填项。客户端标识符是一个带路径的绝对 HTTPS URL。`application_type` 可以包含，但它不是 CIMD 的必填字段。不要把 DCR 对 `application_type` 的要求照搬到首选的 CIMD 路径中。

规范直言不讳地给出两个安全事实：

- **SSRF。** 授权服务器会拉取攻击者提供的 URL，因此必须防范服务端请求伪造（不得拉取内部或管理端点）。
- **localhost 冒充。** 仅靠 CIMD 无法阻止本地攻击者声称拥有某个合法客户端的元数据 URL，并绑定任意 `localhost` 重定向。授权服务器**必须**在同意页面清晰显示重定向 URI 的主机名，并**应该**对仅使用 `localhost` 的重定向发出警告。

由于 CIMD 不需要服务器端状态，也就不需要像 DCR 那样搭建注册器。客户端一侧是只读的：从一个静态 HTTPS 端点提供元数据文档，让授权服务器来拉取即可。

如果授权服务器运维方已经预置了客户端标识符，就先使用这个按签发方限定的注册，而不是尝试自动注册。否则优先使用 CIMD。只有当签发方既不能用预注册也不能用 CIMD 时，才使用已弃用的 DCR。

### RFC 7591：已弃用的兼容性注册

DCR 在 2026-07-28 修订版中已弃用。仅对无法消费 CIMD 且预注册不切实际的授权服务器保留它。兼容客户端会发送：

```json
POST /register
Content-Type: application/json

{
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

服务器返回 `client_id` 以及用于后续更新的 `registration_access_token`：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`application_type` 不是装饰性的。回环地址上的桌面客户端声明 `native`；服务器托管的客户端声明 `web` 并使用 HTTPS 重定向 URI。对公共原生客户端而言，`token_endpoint_auth_method: none` 是正确的默认值。它只获得 `client_id`，由 PKCE 提供持有证明。

三个生产陷阱：

- 注册端点必须按源 IP 限速。否则，恶意行为者可以脚本化地制造数百万次虚假注册，耗尽 `client_id` 命名空间。要在注册器处理请求之前先做限速检查。
- 某些企业 IdP 要求 `software_statement`（一个为客户端背书的签名 JWT）。本课的模拟实现跳过了它；生产环境会接线一个校验步骤，拒绝除 localhost 重定向 URI 之外所有未签名的注册。
- `registration_access_token` 必须以哈希形式存储，而不是明文。该 token 被盗意味着攻击者可以改写客户端的重定向 URI。

### RFC 8707（回顾）——资源指示符

第 16 课已经确立了形态。生产规则是：每个 token 请求都包含 `resource=<canonical-mcp-url>`，且 MCP 服务器在每次调用时都验证 `token.aud` 与自身的资源 URL 匹配。规范 URI 是服务器*最具体*的标识符：使用小写的 scheme 和主机名，不带 fragment，按惯例也不带末尾斜杠。路径部分**不会**被规则剥离——当需要标识单个 MCP 服务器时，规范会保留它。`https://mcp.example.com`、`https://mcp.example.com/mcp`、`https://mcp.example.com:8443` 和 `https://mcp.example.com/server/mcp` 都是合法的规范 URI。为每个服务器选定一个，并把 `aud` 精确固定到它。（本课的模拟实现为简洁起见使用 `https://notes.example.com` 这类裸主机受众；若一次部署在同一源下共置多个 MCP 服务器，则用路径区分它们。）

### RFC 7636（回顾）——PKCE

PKCE 在 OAuth 2.1 中是强制的。本课的授权码流程总是携带 `code_challenge` 和 `code_verifier`。服务器会拒绝任何没有 verifier、或 verifier 哈希后与已存储 challenge 不一致的 token 请求。

### MCP 2026-07-28 授权配置

当前 MCP 修订版保留了 OAuth 资源服务器边界，同时让 MCP 传输变为无状态。没有可用于缓存身份决策的协议会话。因此，授权层会独立校验每个请求：

- 实现 RFC 9728 受保护资源元数据，并通过 401 响应上的 `WWW-Authenticate: Bearer resource_metadata="..."` 头**或** well-known URI `/.well-known/oauth-protected-resource` 提供其位置（SEP-985 让该头变为可选，并提供了 well-known 回退）。元数据的 `authorization_servers` 字段**必须**至少给出一个服务器。
- 仅在**每个**请求上通过 `Authorization: Bearer ...` 接受 token——绝不在查询字符串中传递，也绝不只在会话开始时校验。
- 每个请求都校验 `aud`、`iss`、`exp` 和所需 scope。服务器**必须**验证 token 是专门签发给它的（受众）；`aud` 缺失或不匹配一律拒绝，绝不当作通配符处理。
- 在 401/403 时返回 `WWW-Authenticate: Bearer`，携带 `error=...`、`resource_metadata="<PRM-URL>"` 参数（元数据文档的 URL，*不是*裸资源），以及在 `insufficient_scope`（403）时携带 `scope="..."`。注意：参数是 `resource_metadata`，一个发现指针——质询中不存在 `resource` 参数。
- 授权服务器发现**既可以**接受 RFC 8414 OAuth 元数据，**也可以**接受 OpenID Connect Discovery 1.0；客户端必须按优先级顺序尝试两种 well-known 后缀。
- 由客户端（而非服务器）防御**混淆攻击（mix-up attack）**：客户端在重定向之前记录预期的 `issuer`，并在兑换授权码之前校验实际授权响应中返回的 `iss` 值（RFC 9207）。仅有 PKCE 并不能阻止混淆攻击，因为客户端会把 `code_verifier` 交给它被引导到的任何 token 端点。
- 一个客户端凭据只属于一个授权服务器签发方。如果发现过程解析到不同的签发方，客户端应重新注册，而不是出示旧的 `client_id`、注册 token 或访问 token。
- CIMD 是首选的注册机制。DCR 已弃用；兼容性 DCR 请求仍须声明正确的 `application_type`。

OAuth 2.1 草案是基底；RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD 是接口面；MCP 规范则是配置（profile）。

### 部署能力清单

厂商的功能表很快过时。应改为检查你实际要部署的授权服务器所返回的元数据。这道门禁是机械化的：

| 检查项 | 必须满足的决策 |
|---|---|
| 发现的签发方 | 与策略期望的 HTTPS 签发方完全一致 |
| PKCE | 宣告 `S256`；否则停止 |
| 注册 | 优先 CIMD，接受预注册，DCR 仅作为已弃用的兼容路径 |
| 授权响应 | 存在或已宣告时校验 RFC 9207 `iss` |
| 资源绑定 | token 请求携带 `resource`；资源服务器要求匹配的 `aud` |
| 凭据存储 | 客户端 ID 与注册凭据按签发方分键；访问 token 按「签发方 + 资源」分键 |
| DCR 兼容 | 声明 `native` 或 `web`；拒绝不符合所声明应用类型的重定向 URI |

不要从产品名称或定价层级推断支持能力。把发现的文档写入部署证据，并在必填字段缺失时失败关闭。

### JWKS 刷新模式（在 AS 轮换，在资源服务器刷新）

把两个动词分开，因为混淆它们是一个真实的生产 bug：

- **轮换（rotate）** 是*授权服务器*做的事：生成新的签名密钥，在 JWKS 中发布，稍后再让旧密钥退役。资源服务器完全不参与，也无法执行——它并不持有 IdP 的私钥。
- **刷新（refresh）** 是*资源服务器*做的事：重新 `GET` 已发布的 JWKS 到自己的缓存中。这是资源服务器唯一会执行的 JWKS 操作。

生产的失败模式是缓存过期。用定时刷新任务加键值缓存来解决。资源服务器运行一个任务（cron、定时器，或运行时提供的任何机制），按固定间隔拉取 `<issuer>/.well-known/jwks.json`，并覆盖 `cache[issuer] = {keys, fetched_at}`。校验器从该缓存读取。若某个 token 的 `kid` 在缓存中缺失，就触发**一次**同步刷新作为回退，然后重新检查。这样同时处理两种情况：定时刷新，以及密钥重叠窗口——即由全新密钥签名的 token 在下次定时刷新之前到达。

回退**必须是重新拉取，绝不能是轮换**。如果把缓存未命中路径接到「轮换并生成」上，会有两个问题：(1) 生成新密钥产生的 `kid` *仍然*与 token 不匹配，查找照样失败；(2) 攻击者用随机 `kid` 值喷洒 token，就会迫使系统无限地创建密钥——这是一种自伤式 DoS。重新拉取是幂等的，因此一个伪造的 `kid` 最多浪费一次拉取。

缓存结构：

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同时存在两个密钥是稳态。授权服务器轮换时先引入下一个密钥（`k_2026_04`），再让前一个（`k_2026_03`）退役，因此用旧密钥签发的 token 在过期前仍然有效。缓存保存两者的并集；校验器按 `kid` 选取。

### 校验例程

MCP 服务器在分发任何工具之前先运行校验。`code/main.py` 使用的形态是：

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` 解码 JWT，从 JWKS 缓存解析签名密钥（未命中时刷新一次），验证签名，然后对照允许列表检查 `iss`、对照本服务器的规范资源检查 `aud`，再检查 `exp` 和所需 scope——在第一个失败点返回 `WWW-Authenticate` 质询。把它保持在资源服务器上的单一例程中，意味着每个入口（每次工具调用、每种传输）都经过同样的检查；没有任何路径可以在未先校验的情况下到达工具。

### 不透明 token 使用内省，而不是猜测

并非每个访问 token 都是 JWT。如果签发方文档说明是不透明 token，资源服务器就无法把它解码成可信的声明。它通过经过认证的后端通道，把 token 发送到签发方的 RFC 7662 内省端点，并要求 `active: true`、符合预期的签发方上下文、精确的 MCP 受众或资源、未过期的时间声明，以及具体工具所需的 scope。

按签发方、单向 token 摘要和 MCP 资源缓存内省结果。绝不要用明文 token 作为日志或缓存标签。正向缓存条目的有效期应取以下三者中最早者：token 过期时间、签发方的缓存指导、部署的撤销新鲜度目标。负向缓存要足够短，以免新签发的 token 被错误地保持为 inactive。即使不透明 token 字符串完全相同，针对一个资源的结果也不能授权另一个资源。

不要根据攻击者控制的 token 内容来选择校验模式。把 JWT 与内省行为固定到经过校验的签发方元数据和部署配置上。在 JWT 路径上，固定可接受的算法和受信任的 `jwks_uri`；绝不要跟随仅由 token 头选定的密钥 URL 或算法。

### 撤销是一份新鲜度契约

RFC 7009 允许客户端请求授权服务器撤销一个 token。该请求并不会抹掉每个资源服务器已经缓存的所有副本。要定义可接受的最大撤销延迟，并让每个缓存都遵守它。

不透明 token 部署可以通过对每次高风险调用做内省、或使用短小的正向缓存，来实现更严格的撤销。自包含 JWT 部署通常把以下措施结合起来：较短的访问 token 生命周期、刷新 token 撤销、针对签发方范围事件的密钥退役，以及可选的按主体、会话或 token id 的拒绝列表，用于紧急的本地拒绝。除非资源服务器掌握当前的外部撤销证据，否则已签名的 JWT 在过期前始终保持密码学有效性。

登出、账户停用、撤销同意和事件响应是不同的触发条件，但都必须收敛到同一个可度量的陈述：至多经过声明的撤销窗口之后，每个副本都拒绝该凭据。要通过负载均衡器来测试这一陈述，而不只是针对一个已预热的进程。

### 依赖故障需要已声明的决策

绝不要在异常处理器里即兴决定可用性策略。

| 故障 | 安全的生产行为 |
|---|---|
| 定时 JWKS 刷新失败，已知 `kid` 仍在仍然有效的有界缓存中 | 只在声明的「出错时允许陈旧」窗口内继续，并发出降级健康证据 |
| token 的 `kid` 未知，且唯一允许的一次刷新也失败 | 拒绝；绝不接受无法验证的签名 |
| 内省不可用 | 对受保护调用失败关闭；不要把网络故障转成 `active: true` |
| 受保护资源或签发方元数据意外变更 | 停止新的注册与 token 获取；只保留显式固定、未过期的配置，并受有界事件策略约束 |
| 撤销端点不可用 | 将登出或撤销报告为未完成，尽可能在本地把该凭据标记为不可用，并且不要声称全局撤销已成功 |
| 时钟源或声明类型无效 | 拒绝，而不是不断放宽时钟偏移直到 token 通过 |

把故障与无效凭据分开归类。依赖中断是运维错误，对应健康与重试策略。签名、签发方、受众、过期时间或 scope 不合格是授权拒绝。两者都不会到达工具处理器，也都不应把 token 内容泄漏进审计证据。

### 受众重放演练（访问 token 权限限制）

服务器 A（`notes.example.com`）和服务器 B（`tasks.example.com`）都注册在同一个授权服务器上。服务器 A 被攻陷。攻击者拿到用户的 notes token，并把它重放到服务器 B。

服务器 B 的校验器：

1. 解码 JWT，按 `kid` 获取 JWKS，验证签名。
2. 对照其受保护资源元数据的 `authorization_servers` 检查 `iss`。（通过——同一个 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败——token 的 `aud` 是 `https://notes.example.com`。）
4. 返回 401，并带上 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`。

在协议层，受众声明是防御这种攻击的唯一手段。为性能而跳过它是最常见的生产错误；校验器必须在每个请求上运行，而不只是在会话开始时。规范称之为**访问 token 权限限制**：MCP 服务器 `MUST` 拒绝任何未在受众中指名它的 token。

> **命名说明。** 规范把 *confused deputy*（混淆代理）一词保留给一个相关但不同的问题：MCP 服务器作为第三方 API 的 OAuth **代理**，使用静态客户端 ID，在未获得逐客户端用户同意的情况下转发 token。受众绑定修复的是上面的重放问题；混淆代理的修复方式是逐客户端同意，**外加**绝不把入站 token 透传到上游 API（MCP 服务器 `MUST` 获取自己独立的上游 token）。

### 混淆攻击（服务器无法提供的客户端侧防御）

一个客户端在其生命周期中会与许多授权服务器打交道。恶意 AS 可能试图让客户端在攻击者的 token 端点兑换诚实 AS 的授权码。受众绑定在这里帮不上忙——攻击发生在任何 token 存在之前。防御位于客户端（RFC 9207）：

1. 重定向之前，客户端从经过校验的 AS 元数据中记录预期的 `issuer`。
2. 收到授权响应时，客户端在把授权码发送到任何地方之前，将返回的 `iss` 参数与记录下来的签发方比较（简单字符串比较，不做规范化）。
3. 不匹配（或当 AS 宣告了 `authorization_response_iss_parameter_supported` 但 `iss` 缺失）→ 拒绝，甚至不要显示 `error` 字段。

仅有 PKCE 并不能阻止混淆攻击，因为客户端会把 `code_verifier` 交给它被引导到的任何 token 端点。这就是规范要求按请求把签发方与 PKCE verifier、`state` 一起记录的原因。

### 失败模式

- **JWKS 过期。** 在 AS 轮换密钥后，校验器会拒绝有效 token。修复方法是上面的「定时刷新 + 缓存未命中重新拉取」模式。绝不要在缺少刷新任务的情况下缓存 JWKS。
- **把轮换当作回退。** 把缓存未命中路径接到「轮换并生成」而不是重新拉取，是一个真实 bug：它永远不会产生缺失的 `kid`，还会把攻击者控制的 `kid` 值变成密钥创建型 DoS。回退必须是幂等的 `refresh-jwks`。
- **缺少 `aud` 声明。** 某些 IdP 默认会省略 `aud`，除非 token 请求中带上 `resource`。校验器必须拒绝缺少 `aud` 的 token，不能把缺失当作通配符。
- **因缺少 `iss` 校验而遭受混淆攻击。** 如果客户端不把 RFC 9207 的 `iss` 授权响应参数与重定向前记录的签发方做校验，就可能被引导到攻击者的 token 端点兑换诚实 AS 的授权码。这是客户端侧的失效，资源服务器无法弥补。
- **scope 升级竞态。** 同一用户的两个并发增量授权（step-up）流程都可能成功，并产生两个 scope 不同的访问 token。校验器必须使用请求中出示的 token，而不是去查「用户当前的 scope」——那样会造成 TOCTOU 窗口。
- **注册 token 被盗。** 泄漏的 `registration_access_token` 会让攻击者改写重定向 URI。静态存储时对其做哈希；要求客户端在每次更新时出示明文；一旦怀疑即轮换。
- **`iss` 未固定。** 接受任意 `iss` 的校验器会让攻击者搭起自己的授权服务器，为目标受众注册客户端并签发 token。受保护资源元数据的 `authorization_servers` 列表就是允许列表；必须强制执行。
- **凭据或 token 缓存碰撞。** 只按资源为注册分键的客户端，可能把一个授权服务器的身份出示给另一个。只按签发方为访问 token 分键的客户端，可能把 token 重放到错误的受众。注册按经验证的签发方分键，访问 token 按 `(issuer, resource)` 分键，签发方一旦变化就重新注册。

```figure
t3-jwks-rotate
```

## 使用它

`code/main.py` 用 stdlib Python 演示完整生产流程，包含三个角色：`AuthorizationServer`、`ResourceServer` 和 `Client`。流程如下：

在仓库根目录运行：

```bash
cd phases/13-tools-and-protocols/18-mcp-auth-production
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

第一条命令打印按签发方绑定的注册与 token 校验记录。第二条报告 18 项通过的检查。两条命令都不会打开网络监听，也不会写任何凭据。

1. 授权服务器在 `/.well-known/oauth-authorization-server` 发布 RFC 8414 元数据。
2. MCP 客户端调用元数据端点，检查其注册选项（CIMD 看 `client_id_metadata_document_supported`，DCR 看 `registration_endpoint`）以及 `S256` PKCE 支持。
3. 客户端先检查是否存在按签发方限定的预注册，否则用其 HTTPS Client ID Metadata Document 注册。已弃用的 DCR 仍作为可单独测试的兼容方法保留。
4. 客户端记录经验证的签发方，创建 S256 challenge，收到一次性授权码和 `iss`，校验返回的签发方，然后用原始 verifier 和 RFC 8707 `resource` 指示符兑换授权码。
5. MCP 客户端用 `Authorization: Bearer ...` 调用 MCP 服务器上的工具。
6. MCP 服务器运行 `validate`，从 JWKS 缓存解析签名密钥。
7. IdP 轮换密钥；定时刷新把 JWKS 重新拉取到缓存中。
8. 下一次调用在无需重启的情况下使用刷新后的密钥通过校验；在重叠窗口内，先前的 token 仍然有效。
9. 针对不同 MCP 资源的受众重放尝试会得到 401，并带有 `audience mismatch` 和 `resource_metadata` 指针。

这里的 JWT 使用 HS256 和共享密钥（以便本课只依赖 stdlib）。生产环境使用 RS256 或 EdDSA，并配合上面的 JWKS 模式；除此之外校验逻辑完全相同。由于 IdP 和资源服务器位于同一个进程，`refresh_jwks` 直接读取授权服务器的密钥列表；在真实网络上，这是对 `jwks_uri` 的一次 HTTP `GET`。

## 交付它

本课产出 `outputs/skill-mcp-auth.md`。给定 MCP 服务器配置和 IdP 能力集合，该技能会输出需要搭建的认证接口面——受保护资源元数据、应使用的注册路径（CIMD、预注册或 DCR 回退）、JWKS 刷新计划、scope 映射，以及当 IdP 不支持完整 RFC 配置时应采用的拒绝规则。

## 练习

1. 运行 `code/main.py`，追踪流程。注意 IdP 如何在第 6 步轮换密钥，定时执行的 `refresh_jwks` 如何重新拉取已发布的密钥集，以及旧 token（重叠窗口）和新 token 如何都无需重启即可通过校验。

2. 向受保护资源元数据的 `authorization_servers` 列表中添加一个新的 IdP。签发一个由新 IdP 签名的 token，确认校验器接受它。签发一个由未列出 IdP 签名的 token，确认校验器以 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"` 拒绝。

3. 在 `register_client` 中加入限速检查，并在注册器接受请求之前运行。用一个小字典按 IP 索引，为每个源 IP 维护一个令牌桶。

4. 阅读 RFC 7591，找出本课 `/register` 处理器未校验的两个字段，并补上校验。（提示：`software_statement` 与 `redirect_uris` 的 URI scheme。）

5. 添加第二个授权服务器。确认客户端分别存储按签发方分键的注册，并拒绝复用第一个签发方的 token 或 `client_id`。

6. 验证 DoS 修复。向校验器发送一个带有随机 `kid` 的 token，确认 `refresh_jwks` 至多运行一次，且授权服务器的密钥数量没有增长。然后故意把回退改成「轮换并生成」，观察每来一个伪造 token 密钥数量就上升——之后恢复为重新拉取。

7. 用 `native` 和 `web` 两种客户端演练已弃用的 DCR。确认带 HTTP 重定向 URI 的 web 客户端、以及没有精确回环重定向的 native 客户端都会被拒绝。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| ASM | 「OAuth 元数据文档」 | RFC 8414 的 `/.well-known/oauth-authorization-server` JSON |
| CIMD | 「客户端元数据 URL」 | Client ID Metadata Document：用作 `client_id` 的 HTTPS URL；由 AS 拉取 JSON。MCP 2026-07-28 首选注册方式 |
| DCR | 「自助客户端注册」 | RFC 7591 `POST /register`；在当前 MCP 中已弃用，仅为兼容保留 |
| JWKS | 「用于 JWT 校验的公钥」 | JSON Web Key Set，从 `jwks_uri` 获取，按 `kid` 索引 |
| Rotate vs refresh | 「更新密钥」 | *轮换* = AS 生成/退役签名密钥；*刷新* = 资源服务器重新拉取已发布的密钥集。资源服务器只做刷新 |
| Resource indicator | 「受众参数」 | RFC 8707 `resource` 参数，把 token 固定到一个服务器 |
| `aud` claim | 「受众」 | 校验器用来与规范资源 URL 比对的 JWT 声明 |
| Audience replay | 「token 重放」 | 为服务器 A 签发的 token 被出示给服务器 B；由受众校验防御（规范称：访问 token 权限限制） |
| Confused deputy | 「代理 token 滥用」 | 拥有静态客户端 ID 的 MCP 代理在未获得逐客户端同意的情况下转发 token；与受众重放不同 |
| Mix-up attack | 「错误的 token 端点」 | 客户端被引导到攻击者端点兑换诚实 AS 的授权码；由客户端通过 RFC 9207 `iss` 防御 |
| `iss` allow-list | 「受信任的授权服务器」 | 受保护资源元数据 `authorization_servers` 中列出的集合 |
| `resource_metadata` | 「在哪里找到 PRM 文档」 | 401/403 上 `WWW-Authenticate` 参数，给出 RFC 9728 元数据 URL |
| Public client | 「原生或浏览器客户端」 | 没有 `client_secret` 的 OAuth 客户端；由 PKCE 弥补 |
| `WWW-Authenticate` | 「401/403 响应头」 | 携带 `Bearer error=...` 指令，驱动客户端恢复 |

## 延伸阅读

- [MCP 授权规范（2026-07-28）](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization) - 当前的 MCP 授权配置
- [MCP 2026-07-28 变更日志](https://modelcontextprotocol.io/specification/2026-07-28/changelog) - CIMD、签发方校验、DCR 弃用以及按签发方分键的凭据变更
- [OAuth Client ID Metadata Document（draft-ietf-oauth-client-id-metadata-document-00）](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — 发现契约
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR（回退路径）
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — 公共客户端的持有证明
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — 受众固定
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — 资源服务器发现
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) — 用于防御混淆攻击的 `iss` 参数
- [RFC 7662: OAuth 2.0 Token Introspection](https://datatracker.ietf.org/doc/html/rfc7662)
- [RFC 7009: OAuth 2.0 Token Revocation](https://datatracker.ietf.org/doc/html/rfc7009)
