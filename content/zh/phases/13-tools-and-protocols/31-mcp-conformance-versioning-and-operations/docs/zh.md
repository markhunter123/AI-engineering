# MCP 一致性工程：版本、证据与运维

> 一个服务器并不会因为通过某个 SDK 跑通了正常路径就符合规范。一致性存在于线路上、版本边界处、中间设备之间，以及回滚过程中。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 17 (gateways), Phase 13 · 30 (registry admission)
**Time:** ~100 minutes

## 学习目标

- 把 MCP 的规范性规则转化为黄金线路记录与负面线路记录。
- 把严格的 `2026-07-28` 行为与有界的旧版回退彼此分开。
- 区分增量未知字段与非法的未知 `resultType`。
- 把原始 JSON-RPC 证据与 SDK 归一化后的视图进行比较。
- 通过真实的代理边界证明头部与 body 的完整性。
- 用脱敏后的记录、健康检查和回滚证据来把守发布门禁。

## 问题

你的客户端通过 SDK 调用 `tools/list`，拿到了工具列表。集成测试通过。

这个结果留下了许多没有回答的重要问题：

- 请求是否携带了现代版的逐请求协议元数据？
- `MCP-Protocol-Version`、`Mcp-Method` 和 `Mcp-Name` 是否与 JSON-RPC body 一致？
- 响应在线上是否含有合法的 `resultType`，还是由 SDK 合成的？
- 客户端会保留未来新增的增量字段吗？
- 一个被识别的现代错误会不会意外触发旧版握手？
- 代理是否保留了源站状态码和 JSON-RPC 错误？
- 通知序列化器是否发出了被禁止的响应？
- 运维能否在不存储秘密的前提下，证明某个发布为何被提升或回滚？

一致性是一组可观察的不变量。请构建一个测试框架，在生产流量不得不发现这些问题之前就把这些不变量捕获下来。

```figure
mcp-conformance-operations
```

## 从版本纪元开始

MCP `2026-07-28` 使用自包含的逐请求元数据。现代请求携带 `params._meta.io.modelcontextprotocol/protocolVersion` 和 `params._meta.io.modelcontextprotocol/clientCapabilities`。确切的带命名空间键很重要；裸的 `protocolVersion` 或 `clientCapabilities` 别名属于格式错误。当 HTTP 边界上存在镜像路由头时，其值必须与 JSON-RPC body 一致。现代的成功结果携带 `resultType`。

截至 `2025-11-25` 的版本使用更早的初始化纪元。只有在客户端已经选定那个较早纪元之后，缺少 `resultType` 的旧版结果才会被解释为 complete。

不要创建一个同时接受两种形态的宽松校验器。使用两个分支：

| 分支 | 进入证据 | 缺少 `resultType` | 初始化 |
|---|---|---|---|
| 现代 | 成功的 `server/discover`，或被识别的现代响应 | 非法 | 不是默认路径 |
| 旧版 | 配置好的允许列表，加上在一次无结论的现代探测之后得到的合法旧版 `initialize` 结果 | 解释为 complete | 该纪元所要求 |

这种分离可以防止一个格式错误的现代对端因为更宽松的校验而得到「奖励」。

### 严格模式

严格模式要求提供现代行为的证明。一次成功的 `server/discover` 证明了现代分支。一个被识别的现代 JSON-RPC 错误同样能证明它。修正请求，或者就此停止。绝不要因为服务器返回了 `-32020`、`-32021` 或 `-32022` 就降级。

### 回退模式

回退模式执行一次有界的现代探测。超时、空回复、连接关闭或无法识别的响应都只是无结论，并不能证明对端是旧版。只有显式配置或列入兼容性允许列表的端点，才可以接着接受一次有界的旧版探测；客户端只有在校验该探测的 `initialize` 结果和协商出的旧版修订之后，才会选择旧版分支。

回退不是「遇到任何错误就试旧版」。一个被识别的现代错误包含有用的纠正信息。在它之后降级，可能掩盖头部不匹配、能力声明缺失或版本不受支持等问题。

这可以防止攻击者、故障或过滤代理通过丢弃现代响应来强迫降级。要把端点策略、无结论的现代观测、确切的旧版正面证据和选定的纪元一起记录下来。

在每份记录旁记录选定的纪元。缺少这个事实，同一个缺失字段可能在一次测试运行中看起来可接受，在另一次中却是非法的。

## 构建记录语料库

一份记录夹具记录的是跨越边界的内容，而不只是 SDK 调用：

```json
{
  "name": "golden-modern-list",
  "era": "modern",
  "headers": {
    "MCP-Protocol-Version": "2026-07-28",
    "Mcp-Method": "tools/list"
  },
  "request": {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {
      "_meta": {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {}
      }
    }
  },
  "responseStatus": 200,
  "responseBody": {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
      "resultType": "complete",
      "tools": []
    }
  }
}
```

保留两类夹具。

### 黄金记录

黄金记录证明被接受的行为：

- 元数据与头部匹配的现代发现请求或方法请求
- 字段齐全的 complete 结果
- 当方法可以请求更多输入时的 `input_required` 结果
- 只有在对应能力已声明之后才出现的扩展结果
- 不带 `resultType` 的旧版结果，但仅限在已选定的旧版纪元中
- 不产生任何 JSON-RPC 响应的通知处理

黄金记录追求精确，而不是体量大。把易变的 ID 和时间戳做成确定性的，或在比较前将它们归一化。

### 负面记录

负面记录证明拒绝行为：

- 头部与 body 不匹配
- 缺少逐请求能力
- 匹配到的协议版本不受支持
- 缺少现代 `resultType`
- 未知或未声明的 `resultType`
- 响应的 `jsonrpc` 不是 `2.0`，或 ID 在值或 JSON 类型上不一致
- 响应同时含有 `result` 与 `error`，或两者都没有
- 错误缺少整数 `code` 和字符串 `message`
- 已知协议错误被映射到错误的 HTTP 状态码
- 对通知发出了响应
- 格式错误的 JSON-RPC 信封
- 代理将协议错误压平

对每个负面用例，都要断言拒绝边界和稳定的错误码。「调用失败了」太弱。代理生成的 500 和源站的 `-32020` 看起来都是失败，但对运维讲述的却是完全不同的故事。

头部不匹配夹具必须包含服务器真实的 HTTP 400 JSON-RPC 响应，其请求 ID 匹配，错误码为 `-32020`。只要本地校验器观察到 `HeaderMismatch`，就应自动强制执行这一点；不要把响应校验做成可选的夹具开关。一个返回 HTTP 500 且没有 body 的用例，即使本地拒绝码正确，也应判定失败。如果测试框架在自己的请求校验器抛出异常后就停下，那它只测了自己，没有测服务器的线路行为。

官方的 MCP 一致性项目可以作为外部测试套件和带版本的参考。但也要保留你自己的本地记录。它们记录的是你的代理、SDK、认证、扩展和发布路径，这些是通用套件不可能知道的。

## 头部值必须与 RPC body 匹配

在现代 Streamable HTTP 中，中间设备可以利用镜像头来路由或执行策略。JSON-RPC body 仍然是协议的事实来源。不匹配属于完整性故障，而不是「从中挑一个值」的提示。

按以下顺序校验：

1. 解析并校验 JSON-RPC 信封与元数据类型。
2. 比较 `MCP-Protocol-Version` 与 `params._meta.io.modelcontextprotocol/protocolVersion`。
3. 比较 `Mcp-Method` 与 `method`。
4. 当方法带有路由名称时，比较 `Mcp-Name` 与 body 中对应的值。
5. 在确认相等之后，再判断匹配到的版本和能力集合是否受支持。

这个顺序可以把不匹配（`-32020`）与版本不受支持（`-32022`）区分开。它还能阻止网关按头部名称完成授权、而源站却执行了 body 中另一个名称的情况。

HTTP 字段名大小写不敏感，但字段值仍然大小写敏感。查找之前先归一化头部名称，并拒绝相互冲突的重复项。对于不安全、非 ASCII 或首尾带空白的 `Mcp-Name`，要在与 body 比较之前先解码确切的 `=?base64?{Base64EncodedValue}?=` UTF-8 哨兵编码。不完整的哨兵、非法 Base64、非法 UTF-8 或原始的不安全值，一律以 `-32020` 拒绝。即使 body 含有相同的字符，原始的首尾空白也是非法的，因为该值在传输前必须使用哨兵编码。

中间设备可能在请求到达 MCP 服务器之前就拒绝掉格式错误的 HTTP，因此它的失败可能是没有 JSON-RPC 的 HTTP 错误。要记录拒绝来自中间设备还是源站。当源站 MCP 服务器处理的是合法的 JSON-RPC 请求时，应使用协议错误契约。

## 未知字段不等于未知结果

向前兼容需要两条不同的规则。

### 增量未知字段

结果对象和 `_meta` 映射都可能新增字段。校验器应根据自身角色选择保留或忽略增量字段，除非该字段违反了保留契约。示例把完整的原始结果保存在证据中，并在已知结果旁接受 `futureHint`。

如果你是透明代理，保留未知字段通常比剥离它更安全。如果你是应用客户端，忽略它可以是合法的。但你的差异对比测试仍应揭示 SDK 省略了它，使这一行为是有意为之。

### 未知的 `resultType`

`resultType` 是一个判别符。现代核心结果使用 `complete` 或 `input_required`。只有当扩展的能力已被声明时，扩展才能增加其他值。例如，Tasks 扩展可以在协商好的能力上下文中增加 `task`。

未知或未声明的判别符不能被安全地当作 complete。客户端并不知道自己会丢弃什么样的生命周期。拒绝它。

因此，同一份原始响应可能同时包含可接受的未知字段和不可接受的未知结果类型。两种情况都要测试。

判别符只是第一层。之后还要校验方法特定的负载。一个 complete 的 `tools/list` 结果需要一个 `tools` 数组，其描述符具有唯一且非空的名称、有用的描述，以及对象根部的 `inputSchema` 值。`task` 结果只对具备 Tasks 能力且符合资格的 `tools/call` 有效，并要求 `taskId`、已知状态、创建与更新时间戳、`ttlMs`，以及一个合法的可选轮询间隔。一个 complete 的 `completion/complete` 结果要求一个 `completion` 对象，其中至多 100 个字符串值，一个可选的非负整数 `total` 且不小于返回值的数量，以及一个可选的布尔值 `hasMore`。拼写正确的 `resultType` 无法让格式错误的负载变得合规。

## 通知不变量

JSON-RPC 通知没有 `id`。接收方不得发送 JSON-RPC 成功或错误响应。

对于被接受的 HTTP 通知形态，测试框架期望 HTTP `202` 且 body 为空。MCP `2026-07-28` 没有定义任何通过 Streamable HTTP 传输的核心客户端到服务器通知。示例只使用一个带命名空间的课程扩展通知来测试单向序列化器不变量。不要把它当作新的核心方法。

要测试序列化器，而不只是处理器。处理器可能返回 `None`，而中间件却把它包装成一个 JSON 成功对象。请捕获最终的出口字节。

## 加入 SDK 差异对比

SDK 常常把线路对象转换成方便的语言类型。这很有用，但归一化后的对象无法证明实际收到了什么。

对每个高风险夹具，捕获：

1. SDK 解码之前的原始状态码、头部和响应 body。
2. SDK 归一化后的返回值或异常。
3. 针对所选纪元的预期语义投影。
4. 被 SDK 提升、合成、剥离或改变的字段。

示例允许 SDK 在比较应用负载时移除已知的线路簿记字段，如 `resultType`、`_meta`、`ttlMs` 和 `cacheScope`。它会报告 `futureHint` 被丢弃，因为这个未知的语义字段消失了。

不要假设每一处差异都是 SDK 的缺陷。重点在于让转换过程可见。判断你的组件是应用端点（可以忽略增量字段），还是透明中间设备（应当保留它）。

对你交付的每个 SDK 和版本都运行这项差异对比。如果两个 SDK 对同一份记录的归一化方式不同，发布策略应当事先指明哪种行为可接受，而不是事后挑选最方便的输出。

## 采集代理证据

大多数生产环境的 MCP 故障都跨越不止一个进程。记录三个视图：

| 视图 | 最低证据要求 |
|---|---|
| 入口 | 请求头、JSON-RPC body、内容类型、经过认证的路由、接收时间 |
| 源站 | 转发后的头部和 body 摘要、源站状态码、响应头和响应 body |
| 出口 | 客户端可见的状态码、头部、body 和发送时间 |

示例检测两种常见的转换：

- 源站的 HTTP 400 或 404 JSON-RPC 错误变成了通用的代理 500
- 出口的 JSON-RPC body 与源站 body 不一致

为内容类型、`Accept`、压缩、请求级 SSE、缓存头和 trace 关联添加部署特定的断言。在策略允许时，采集 TLS 终结的两侧。绝不要仅仅为了证明路径而记录凭据。

## 在证据离开内存之前脱敏

脱敏是一致性运维的一部分，而不是事后清理工作。要在序列化、哈希、日志、测试产物或故障上传之前完成脱敏。

示例在匹配之前先对键名做大小写折叠并移除分隔符，然后递归替换 `Authorization`、`Cookie`、`Set-Cookie`、`X-Api-Key`、`accessToken`、`clientSecret`、`registrationAccessToken`、`token`、`password`、`secret` 和 `api_key` 等键下的值。归一化与拒绝名单必须使用同一种形态，这样 camelCase、连字符、下划线和点分变体就无法各走一套策略绕过彼此。生产采集器还应加入方法特定的参数策略，因为像 `query` 这样一个看似无害的键，也可能包含个人数据或受监管数据。

对脱敏后的证据包做哈希。只有在某项具体调查需要时，才把原始采集保存在经批准的短期系统中。摘要证明的是哪一份脱敏证据包驱动了决策；它不会泄露被移除的值。

## 把健康检查与回滚纳入门禁

协议一致性是发布的必要条件，但并不充分。一个符合规范的候选版本仍然可能超时、泄漏内存或压垮某个依赖。

在发布之前定义健康窗口：

- 最小样本数
- 最大错误率
- 最大延迟分位数
- 饱和度或资源上限
- 观察时长
- 与已准入基线的比较

同样要在发布之前定义回滚证据：

- 确切的上一版本
- 准入证据摘要
- SHA-256 产物与描述符固定值
- 当前的 Registry 状态
- 当前健康检查结果
- 路由恢复流程
- 由受信任的发布控制器身份对上述确切字段出具的证明（attestation）

要求在提升之前就验证回滚目标并确认其健康，而不是等到候选版本失败之后才做。一次成功发布如果没有可用的恢复路径，就不算生产就绪。

如果候选版本失败，而回滚目标缺少这些证据，就保持流量不动，而不是靠猜。「回滚到原来那个」不是一项运维控制。

不要把就绪性简化为真值检查，例如版本非空、`healthy: "yes"` 或一个任意字符串形式的证据。示例要求的类型必须精确：一个活跃状态、三个 SHA-256 摘要、一个受信任的签名者，以及一个针对完整回滚负载的有效 HMAC-SHA-256 证明。它的确定性演示密钥是不含秘密的夹具。生产环境要在发布边界注入受保护的密钥、KMS 验证结果或公钥证明校验器。

发布门禁还会拒绝空的记录证据、SDK 差异证据或代理证据。每个来源都必须带有合法的证据摘要。一个绿色的健康窗口无法填补一个从未被观察过的边界。

## 构建它

运行这套基于标准库的测试框架：

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations
python3 code/main.py
```

演示会运行恰好十五份黄金与负面记录，包括合法和格式错误的 completion 结果；把原始结果与 SDK 视图比较；检查一个吞掉了源站错误的代理；评估健康状态；认证回滚证据；并选定该目标。

预期的输出形态：

```json
{
  "transcriptsPassed": 15,
  "transcriptsTotal": 15,
  "sdkDroppedFields": ["futureHint"],
  "proxyIssues": [
    "proxy collapsed a protocol error into HTTP 500",
    "proxy changed the origin JSON-RPC body"
  ],
  "releaseAction": "rollback",
  "evidenceDigest": "..."
}
```

按以下顺序阅读 `code/main.py`：

1. `validate_request()` 强制执行纪元特定的请求与头部规则。
2. `validate_result()` 区分缺失的旧版判别符、合法的现代值、扩展和未知值。
3. `select_era()` 实现严格模式与有界回退策略。
4. `run_transcript()` 评估黄金与负面夹具。
5. `compare_sdk_view()` 暴露归一化差异。
6. `inspect_proxy()` 比较入口、源站与出口证据。
7. `redact()` 在证据哈希之前移除明显的秘密。
8. `rollback_evidence_ready()` 校验确切的固定字段与受信任的发布证明。
9. `ReleaseGate.evaluate()` 汇总非空的一致性、SDK、代理、健康和回滚证据。

## 使用它

在四个时点运行这套测试框架：

1. 每次实现变更时，配合进程内测试适配器。
2. 针对真实传输上的已构建客户端与服务器二进制。
3. 在预发布环境中，穿过已部署的代理或网关。
4. 在金丝雀发布期间，结合实时健康与回滚证据。

在各层保持相同且稳定的用例名称。`negative-header-body-mismatch` 在单元、端到端、代理和金丝雀报告中都应指向同一个不变量。证据摘要会因为边界不同而不同；需求本身不应改变。

把夹具 schema 存入版本控制。把脱敏后的运行证据存入发布系统。短期原始采集只存放在纳入事故访问控制的存储中。

## 交互实验

### 实验 A：验证纪元边界

在 `code` 目录下打开 Python：

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations/code
python3 -q
```

运行：

```python
from main import *
validate_result({"tools": []}, "legacy")
validate_result({"tools": []}, "modern")
```

旧版调用会推断出 `complete`。现代调用抛出 `ProtocolViolation`。现在测试回退：

```python
select_era({"kind": "timeout"}, "fallback")
select_era(
    {"kind": "timeout"},
    "fallback",
    legacy_allowed=True,
    legacy_evidence={"kind": "initialize_success", "protocolVersion": LEGACY_VERSION},
)
select_era({"kind": "jsonrpc_error", "code": -32021}, "fallback")
```

第一次超时会安全失败（fail closed），因为沉默不是旧版证据。第二次调用之所以选择旧版，是因为配置允许，并且观察到了一个合法的旧版初始化结果。被识别的缺失能力错误证明了现代分支。

### 实验 B：增量字段与判别符

```python
validate_result({"resultType": "complete", "tools": [], "futureHint": True}, "modern")
validate_result({"resultType": "future_mode", "tools": []}, "modern")
```

第一个结果保留了 `futureHint`。第二个被拒绝，因为生命周期判别符未知。

### 实验 C：检查 SDK 的转换

```python
compare_sdk_view(
    {"resultType": "complete", "tools": [], "futureHint": {"mode": "new"}},
    {"tools": []},
)
```

决定你的组件是可以忽略 `futureHint`，还是必须转发它。把该选择写入发布策略。不要悄悄抹掉这项差异。

### 实验 D：修复代理

修改演示中的交换过程，让出口保留源站的状态码与 body。再次运行 `python3 main.py`。代理问题应当消失，但 SDK 差异仍会阻止提升。然后把 `futureHint` 包含进 SDK 视图，观察当每个证据来源都通过时，动作变为 `promote`。

## 实践练习

向测试框架中加入请求级 SSE 记录。

要求：

- 捕获响应状态码、内容类型、有序的 SSE 事件和流终止。
- 证明每个 JSON-RPC 事件都具有合法的、纪元特定的结果或错误。
- 添加一个负面用例，覆盖会先缓冲整个流再转发的代理。
- 添加一个负面用例，覆盖 SSE 事件中的 JSON-RPC id 与请求不一致的情况。
- 在写入证据之前对事件数据脱敏。
- 在健康窗口中包含流时长、首事件延迟和事件数量。
- 当流失败时，让发布门禁只选择有证据支撑的回滚目标。

成功的标准是：同一个用例既能直接运行，也能穿过代理运行，并且报告能指出究竟是哪个边界改变了行为。

## 交付产物

本课交付 `outputs/skill-mcp-conformance-release-gate.md`。用它把服务器、客户端、网关或 SDK 的变更转化为一份带版本的一致性矩阵和发布决策。该产物要求原始线路证据、负面用例、显式的纪元选择、SDK 差异对比、代理证明、脱敏、健康阈值和回滚证据。

## 验证它

运行演示和确定性测试套件：

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

验证应证明：

- 每一份纳入的黄金与负面记录都达到预期结果
- 现代请求要求确切的带命名空间元数据键
- HTTP 头部名称按大小写不敏感匹配，编码后的 `Mcp-Name` 值被精确解码
- 头部与 body 不匹配时返回现代的不匹配错误码
- 响应的版本、ID、result 与 error 的互斥性、错误形态以及 HTTP 映射都得到校验
- 强制执行方法特定的工具列表、任务和 completion 负载要求
- 每一次观察到的 `HeaderMismatch` 都要求真实的 HTTP 400 JSON-RPC `-32020` 响应
- 原始的 `Mcp-Name` 空白被拒绝，而哨兵编码的精确空白可以往返
- 缺失 `resultType` 只在所选定的旧版纪元中有效
- 增量字段能通过原始校验，而未知结果类型会失败
- 扩展结果类型要求其能力已被声明
- 被识别的现代错误永远不会触发旧版回退
- 通知不产生任何 JSON-RPC 响应
- SDK 移除簿记字段与丢失语义字段被区分开来
- 代理错误压平会被检测到，凭据能在 camelCase 与分隔符变体上递归脱敏
- 提升要求非空的记录、SDK、代理以及健康的运维证据
- 提升与回滚都要求一个经过认证、已固定、活跃且健康的回滚目标

## 生产环境故障模式

| 故障 | 弱测试报告的内容 | 测试框架必须证明的内容 |
|---|---|---|
| SDK 合成了一个缺失的判别符 | 「tools/list 通过了」 | 原始现代结果缺少 `resultType`，因此非法 |
| 客户端在 `-32021` 之后降级 | 「旧版重试成功了」 | 被识别的现代错误禁止回退 |
| 未知结果类型被当作 complete | 「响应已解析」 | 未声明的生命周期判别符被拒绝 |
| 代理授权了工具 A，源站执行了工具 B | 「请求到达了服务器」 | 每一跳上的 `Mcp-Name` 都等于 body 中的路由名称 |
| 测试框架在读取服务器响应之前就抛出异常 | 「头部不匹配测试通过了」 | HTTP 400 与 JSON-RPC `-32020` 响应被捕获并校验 |
| 代理把源站 400 变成通用 500 | 「上游错误」 | 源站与出口的状态码及 JSON-RPC body 都被保留 |
| 通知中间件发出 `{result: null}` | 「处理器返回了 none」 | 最终出口 body 为空，且不存在 JSON-RPC 响应 |
| SDK 剥离了增量字段 | 「类型化对象匹配」 | 原始视图与归一化视图显示出确切被丢弃的字段 |
| 故障产物泄露了 bearer token | 「调试包已上传」 | 脱敏发生在哈希、日志或上传之前 |
| 凭据键名风格绕过了脱敏 | 「拒绝名单包含 api_key」 | camelCase 与分隔符变体共享同一种归一化拒绝名单形态 |
| 金丝雀没有样本但看起来健康 | 「零错误」 | 最小样本数被强制执行 |
| 回滚选择了未知构建 | 「上一部署已恢复」 | 目标版本、准入摘要、固定值、状态和健康均存在 |

## 运营规则

测试你发送的字节、每个中间设备转发的字节、每个 SDK 暴露的语义，以及运维在压力下将要使用的证据。兼容性是一条显式的分支。回滚是一项有证据支撑的发布动作。两者都不应是宽松解析器的意外副作用。

## 延伸阅读

- [MCP 2026-07-28 base protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP version negotiation](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Official MCP conformance project](https://github.com/modelcontextprotocol/conformance)
