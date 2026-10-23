# MCP Registry 供应链：准入、漂移与回滚

> 一条 Registry 记录只告诉你发布者声明了什么。生产环境的准入要证明：你获取了什么、你观察到了什么、你批准了什么，以及你能安全恢复什么。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 17 (gateways and registries), Phase 13 · 18 (production authentication)
**Time:** ~90 minutes

## 学习目标

- 区分 Registry 发布、包来源证明、运行时发现与本地审批。
- 验证 MCP 服务器的命名空间，而不信任其自身记录中的名称。
- 固定不可变的发布物、执行源、来源证明与实时描述符证据。
- 在准入之后检测 Registry 状态变化与运行时漂移。
- 把路由回滚到先前已准入的版本，而不改写历史。
- 维护一份可发现篡改（tamper-evident）的准入账本，能够解释每一项决策。

## 问题

你在 Registry 中找到了 `com.example/inventory`。它的描述看起来没问题。它的包确实存在。服务器也能响应 `server/discover`。

这并不是一个事实，而是一条来自不同权威的事实链：

1. 某个对命名空间通过认证的发布者提交了一条记录。
2. 某个包仓库提供了一个具有特定身份和摘要的产物。
3. 一个正在运行的端点报告了协议版本、能力、工具和诊断用的服务器信息。
4. 你的组织决定，这一确切组合是被允许的。

把这些事实压缩成「它在 Registry 里，所以可信」，会制造一个供应链盲区。一次有效的发布仍然可能被标记为 deprecated。如果你不固定包的摘要，一个包 tag 就可能指向意料之外的产物。服务器可能在评审之后新增一个破坏性工具。回滚也可能悄悄选了一个从未被准入的版本。

解法是一个准入控制器，在每个边界上都留下证据。

## Registry 是索引，不是你的审批系统

官方 MCP Registry 存储服务器元数据。它的 `server.json` 记录声明一个服务器版本，并声明一个或多个包或远程端点。发布规则额外加入命名空间认证、包所有权检查、受限仓库规则，以及一个范围很窄的发布者元数据位置。

这些控制回答的是「发布」问题。你的生产策略仍然要回答「部署」问题：

| 边界 | 问题 | 证据所有者 |
|---|---|---|
| 命名空间 | 发布者是否有权使用这个名字？ | Registry 认证，加上你验证过的命名空间输入 |
| 记录 | 发布者对这个版本声明了什么？ | 不可变的 `server.json` 摘要 |
| 执行源 | 将执行哪个包或远程端点？ | 声明的源字段、已验证的所有权结果、传输方式和可信摘要 |
| 运行时 | 端点现在暴露了什么？ | `server/discover` 与工具描述符 |
| 准入 | 你的策略是否批准了这一确切组合？ | 本地固定记录与账本条目 |
| 运维 | 它现在还安全吗，有什么可以替代它？ | 漂移检查、状态同步、健康检查与回滚路由 |

Registry 的 schema 版本与 MCP 协议版本相互独立。一条记录可能使用已发布的 `2025-12-11` 服务器 schema，而在线服务器支持的是 MCP `2026-07-28`。绝不要由其中一个推断另一个。

```figure
mcp-registry-admission
```

## 一个准入决策中的七项控制

### 1. 命名空间验证

官方 Registry 的名称使用经过认证的命名空间。一个经过验证的域可以映射为反向的域名前缀。例如，控制 `example.com` 可以确立对 `com.example/*` 的使用权。

不要接受字符串前缀检查：

```python
server_name.startswith("com.example")
```

这样也会接受 `com.exampleevil/tool`。请在 `/` 处切分名称，要求 slug 非空，并精确比较命名空间段。更重要的是，把认证结果中经过验证的命名空间传入准入流程。不要从不可信的记录中推导信任。

由 GitHub 支撑的命名空间和由域支撑的命名空间走不同的认证路径。把任一路径归一化为同一个准入输入：精确的、已验证的命名空间字符串。

### 2. 来源证明关联

对于包记录，声明与获取到的产物必须通过显式字段关联起来：

- 包仓库类型
- 包标识符
- 包版本
- 已验证的所有权结果
- 下载产物的摘要

同时要校验声明的包传输方式。只包含远程端点的记录是有效的，不得因为缺少包而被拒绝。对于远程源，要把声明的 URL 和传输类型与独立验证过的端点所有权，以及可信连接或部署证据的摘要关联起来。

本课代码同时支持两种源类型，并把选中的源与 Registry 源、服务器名称、Registry 版本、记录摘要和证据摘要一起做哈希。得到的来源证明摘要是指向完整证据集的一个紧凑指针。它不能替代对证据本身的保留。

绝不要接受仅由你正试图验证的产物所提供的摘要。要在可信的获取边界上计算它，或者从你已验证其校验结果的包服务那里接收它。

### 3. 固定决策，而不只是版本

Registry 版本是唯一的发布标识符。已发布的元数据不可变。记录一旦改变就需要新版本。推荐使用语义化版本，但 Registry 并不强制要求，也不接受版本范围。

这意味着 `^1.4` 不是准入固定项，「latest」也不是。一份有用的固定记录包含：

```json
{
  "server": "com.example/inventory",
  "version": "1.0.0",
  "recordDigest": "...",
  "source": {"kind": "package", "registryType": "pypi"},
  "sourceDigest": "...",
  "toolsetDigest": "...",
  "provenanceDigest": "...",
  "registryStatus": "active"
}
```

固定多个层次，可以让你识别出是哪个边界发生了变化。同一个 Registry 版本下记录摘要改变，属于 Registry 完整性故障。同一个包坐标或远程部署下源摘要改变，属于执行源完整性故障。工具集摘要改变，则是运行时漂移。

### 4. 实时漂移检测

准入应当观察真正会接收流量的那台服务器。调用 `server/discover`，通过可信路径列出或以其他方式获取对外暴露的工具描述符，并验证：

- `2026-07-28` 位于 `supportedVersions` 中
- 所有本地必需的能力都存在
- 每个工具描述符具备必需的身份与 schema 表面
- 后续检查中，归一化后的描述符摘要与已准入的固定记录一致

可选的 `_meta["io.modelcontextprotocol/serverInfo"]` 值属于自报的展示、日志和调试上下文。把它记录为诊断证据，但绝不要用它来确立命名空间、包所有权、端点所有权、准入或任何其他安全决策。`_meta` 之外的直接 `serverInfo` 别名不是契约字段，也不应被提升为诊断证据。

只对顺序没有意义的字段做归一化。示例在哈希之前按稳定的名称对工具列表排序，因此无害的列表顺序变化不会造成漂移。它不会丢弃描述符字段。新增工具、schema 变化、描述变化或新增注解，都会改变固定记录。

示例把格式错误的描述符以及任何描述符摘要变化视为漂移，隔离该固定记录，移除其活跃路由，并把该版本封禁为回滚目标。生产策略可以只允许通过一次新的评审来放行纯文案改动，因为描述会影响模型的工具选择。「表面性」的元数据也能改变智能体的行为。

### 5. Registry 状态是实时状态

Registry API 会在每条服务器记录旁附带一个响应级的 `_meta` 对象。Registry 管理的字段位于 `_meta["io.modelcontextprotocol.registry/official"]` 之下。把响应中的 `_meta` 对象传给准入流程，并读取 `_meta["io.modelcontextprotocol.registry/official"].status`。直接的 `_meta.status` 值不是官方线格式。不要混淆响应元数据与发布记录自身的 `_meta`。状态可以是：

- `active`：默认返回，有资格进入本地准入
- `deprecated`：仍可发现但会带警告，已不再是安全的自动选择
- `deleted`：默认隐藏，但历史记录仍可通过 deleted 或增量视图获取

准入之后要同步状态。如果一个活跃版本变为 deprecated 或 deleted，隔离它的固定记录，并停止把新工作路由给它。保留证据。从默认列表中消失，并不等于允许你擦除审计轨迹。

发布者提供的自定义元数据只能位于发布记录中的 `_meta.io.modelcontextprotocol.registry/publisher-provided` 之下。Registry 管理的响应元数据是另一回事。不要让发布者自行设置官方状态。

### 6. 回滚意味着恢复路由

回滚时不会编辑不可变的发布物。回滚选择的是一个先前已准入、当前仍有资格的固定记录，并改变活跃路由。

一个安全的目标必须：

1. 拥有已完成的准入记录。
2. 在你的策略下仍具有活跃的 Registry 状态。
3. 未被运行时或安全证据隔离。
4. 仍能解析到固定的包与实时描述符集合。
5. 通过当前的健康检查。

示例聚焦于前三个条件。真实的协调器（reconciler）应在激活之前重新获取包，并重新检查在线端点。

### 7. 追加准入账本

准入数据库说明当前什么是活跃的。账本解释为什么。

示例中的每个条目包含序号、时间、事件、服务器、版本、结果、原因、证据、上一条目的哈希以及自身哈希。修改一个较早的结果，会破坏该条目以及之后每一个环节的验证。

这只能做到「可发现篡改」，并非魔法般的「防篡改」。要把周期性的账本头锚定到独立的信任域中，例如签名的发布元数据或一次写入存储。限制谁可以追加。不要让授权 token、包凭据、工具参数和私有端点数据进入证据。

## 构建它

可运行的控制器位于 `code/main.py`。它只使用 Python 标准库。

先运行这个有限演示：

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift
python3 code/main.py
```

演示执行五个操作：

1. 在命名空间、包来源证明、协议、能力和工具均匹配的情况下准入 `1.0.0`。
2. 准入 `1.1.0` 并将其设为活跃。
3. 在运行时观察到一个意料之外的 delete 工具。
4. 观察到 `1.1.0` 的 Registry 状态变为 `deprecated`。
5. 把路由恢复到仍处于已准入状态的 `1.0.0` 固定记录。

预期的输出形态：

```json
{
  "admitted": [true, true],
  "driftAllowed": false,
  "rollbackAllowed": true,
  "activeVersion": "1.0.0",
  "ledgerValid": true
}
```

按以下顺序阅读实现：

1. `namespace_for_domain()` 与 `namespace_matches()` 确立精确的命名权威。
2. `digest()` 与 `normalized_tools()` 产生确定性证据。
3. `RegistryAdmissionController.admit()` 把发布、来源证明、运行时与策略关联起来。
4. `check_live()` 把新的观测结果与固定记录比较。
5. `observe_registry_status()` 隔离 Registry 状态发生变化的版本。
6. `rollback()` 只激活先前已准入且有资格的目标。
7. `AdmissionLedger.verify()` 检测记录历史是否被改动。

## 使用它

把控制器放在发现与路由之间：

```text
Registry sync -> artifact verifier -> live discovery -> admission controller -> route table
                                               |                 |
                                               v                 v
                                          evidence store    admission ledger
```

为这些工作使用彼此分离的身份。Registry 同步 worker 需要读取元数据的权限。产物校验器需要获取包的权限。路由协调器需要激活已批准固定记录的权限。它们谁都不需要拿到全部凭据。

让发布状态显式化。「已批准」意味着证据通过了策略。「活跃」意味着路由当前选中它。「已隔离」意味着它不能接收新工作。「已被取代」意味着另一个已准入版本处于活跃状态。不要把四种含义编码进同一个布尔值。

在通过 `tools/list` 暴露服务器之前先运行准入。否则客户端可能在发布与策略评估之间的空档里发现某个工具。

## 交互实验

你将一次观察一个边界失效。

### 实验 A：命名空间冲突

在代码目录下打开 Python shell：

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift/code
python3 -q
```

然后运行：

```python
from main import namespace_matches
namespace_matches("com.example/inventory", "com.example")
namespace_matches("com.exampleevil/inventory", "com.example")
```

第一个结果是 `True`，第二个是 `False`。在本地把精确比较换成 `startswith`，观察第二个名称为何越过了边界。继续之前请恢复精确比较。

### 实验 B：描述符漂移

```python
from main import *
times = iter(f"2026-08-21T12:00:{n:02d}+00:00" for n in range(10))
c = RegistryAdmissionController(clock=lambda: next(times))
meta = {OFFICIAL_META_KEY: {"status": "active"}}
c.admit(sample_record("1.0.0"), meta, "com.example", evidence_for("1.0.0"), sample_live("1.0.0"))
c.check_live("com.example/inventory", "1.0.0", sample_live("1.0.0", True))
```

检查原因与路由状态。包和 Registry 记录都没有变化，变化的是运行时的工具表面，因此控制器隔离并停用了该固定记录。这就是供应链控制必须在安装之后继续存在的原因。

### 实验 C：状态与回滚

准入 `1.1.0`，把它标记为 deprecated，然后尝试两个回滚目标：

```python
c.admit(sample_record("1.1.0"), meta, "com.example", evidence_for("1.1.0"), sample_live("1.1.0"))
c.observe_registry_status("com.example/inventory", "1.1.0", "deprecated")
c.rollback("com.example/inventory", "1.1.0", "unsafe retry")
c.rollback("com.example/inventory", "1.0.0", "restore known release")
c.ledger.verify()
```

被隔离的目标遭到拒绝。较早的活跃固定记录被接受。账本保持有效。

## 实践练习

为控制器扩展一个双人审批门禁。

要求：

- 把审批存储为已签名的证据引用，而不是固定记录里的可变姓名。
- 当工具集包含带有 `destructiveHint: true` 的工具时，要求两个不同的评审者身份。
- 拒绝重复的评审者身份。
- 审批不完整时，在账本中保留最初的准入尝试。
- 为零个、一个、重复和两个不同审批分别添加测试。
- 不要记录签名、凭据或完整的私有工具参数。

成功的标准是：在两份身份都批准了确切的记录、包和工具集摘要之前，破坏性工具不能变为活跃。

## 交付产物

本课交付 `outputs/skill-mcp-registry-admission.md`。在评审新的 Registry 版本或调查漂移时，可以把它当作一份扁平、可复用的运行手册。它定义了输入、拒绝规则、证据包、状态对账和回滚证明，且不依赖示例中的类名。

## 验证它

运行演示和确定性测试套件：

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

验证应证明：

- 精确的命名空间边界会拒绝形似前缀
- 只有官方命名空间下的 Registry 状态才能让某个版本具备资格
- 未经验证或不匹配的包证据与远程证据会被拒绝
- 发布者元数据无法冒充 Registry 管理的元数据
- 工具顺序被归一化，同时不会掩盖描述符变化
- 格式错误的包结构与工具结构会被安全拒绝
- `serverInfo` 只作为诊断信息，永不提供准入权威
- 描述符漂移会隔离固定记录、停用它，并阻止回滚到它
- 状态变化会隔离活跃的固定记录
- 回滚无法选择被隔离的或未知的版本
- 账本被篡改可以被检测到

## 生产环境故障模式

| 故障 | 发生原因 | 必需响应 |
|---|---|---|
| 名称看起来有效，但命名空间从未经过认证 | 策略信任了记录文本 | 在可信的命名空间校验器给出精确前缀之前一律拒绝 |
| 同一包坐标返回了新的字节 | 上游可变，或分发渠道被攻破 | 停止激活，保留两份摘要，调查获取边界 |
| 未经评审「latest」就发生了变化 | 浮动选择逃出了固定记录 | 只解析确切的已准入版本与摘要 |
| 审批之后出现了新工具 | 运行时漂移，或换成了另一个部署 | 隔离该路由并采集一份新的描述符观测 |
| 已 deprecated 的版本仍然活跃 | 缺少状态同步，或同步延迟 | 按计划并在激活之前对账状态 |
| 已删除的记录从默认同步中消失 | 客户端只请求了活跃记录 | 使用增量式或能感知删除的对账方式，并保留本地历史 |
| 回滚目标从未被准入 | 路由控制与审批状态彼此脱节 | 拒绝回滚，并对目标执行一次新的准入 |
| 攻击者改写了全部条目后，账本在本地仍能通过验证 | 哈希链没有外部锚点 | 把签名的账本头发布到独立的信任域 |
| 证据中包含 bearer token 或工具参数 | 日志复制了完整请求 | 在采集时脱敏，只存储最低限度的证明 |

## 运营规则

发布回答的是「这个身份能否发布这个名字？」准入回答的是「我们是否会执行这一确切的产物，并暴露这一确切的行为？」把这两类决策分开，固定每一次关联，并让回滚依据证据而不是记忆来做选择。

## 延伸阅读

- [Official Registry server.json requirements](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [Official Registry OpenAPI contract](https://registry.modelcontextprotocol.io/openapi.yaml)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
