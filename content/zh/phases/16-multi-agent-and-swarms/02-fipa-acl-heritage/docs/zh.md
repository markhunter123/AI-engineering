# FIPA-ACL 与言语行为的遗产

> 在 MCP 之前，在 A2A 之前，就有了 FIPA-ACL。2000 年，IEEE 智能物理智能体基金会（Foundation for Intelligent Physical Agents）批准了一种智能体通信语言，它有二十个 performative、两种内容语言，以及一组交互协议——合同网（contract net）、subscribe/notify、request-when。它之所以从工业界淡出，是因为本体（ontology）开销对 Web 来说太重；但多智能体系统的 LLM 复兴正在悄悄重新实现同样的想法，却没有那套形式语义：JSON 契约取代了 performative，自然语言取代了本体。本节课认真阅读 FIPA-ACL，让你看清 2026 年的哪些协议决策是重新发明、哪些是真正的新东西，以及当前这波浪潮会在哪里重新发现 2000 年代已经解决的问题。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 minutes

## 问题

2026 年的智能体协议版图很热闹：MCP 管工具，A2A 管智能体，ACP 管企业审计，ANP 管去中心化信任，NLIP 管自然语言内容，再加上 CA-MCP 和几十个研究提案。每个规范都宣称自己是基础性的。

诚实的解读是：其中大多数都在重新发现一棵非常具体的、二十年前的决定树。Austin（1962）和 Searle（1969）的言语行为理论给了我们「话语即行动」。KQML（1993）把它变成了线缆协议。FIPA-ACL（2000 年批准）产出了参考标准：二十个 performative、内容语言 SL0/SL1、合同网与 subscribe-notify 的交互协议。JADE 和 JACK 是 Java 参考平台。这项努力在 2010 年前后淡出，因为本体开销太重，而 Web 正在胜出。

当你审视 MCP 的 `tools/call`、A2A 的任务生命周期，或 CA-MCP 的共享上下文存储时，你看到的是 FIPA 决策更柔和、JSON 原生的翻版。了解这段历史能告诉你两件事：哪些新「创新」其实是重新发明，以及新规范会重新发现哪些旧的失败模式。

## 概念

### 一段话讲清言语行为

Austin 注意到，有些句子不是在描述世界——它们改变世界。「我承诺。」「我请求。」「我宣布。」他把这些称为施事话语（performative utterances）。Searle 将其形式化为五个类别：断言式、指令式、承诺式、表达式和宣告式。KQML（Finin 等，1993）让这一理论对软件智能体可操作：一条消息等于 performative（动作）加 content（动作所涉及的内容）。FIPA-ACL 清理了 KQML 的缺口，并围绕二十个 performative 做了标准化。

### FIPA 的二十个 performative（部分列表）

| Performative | 意图 |
|---|---|
| `inform` | 「我告诉你 P 为真」 |
| `request` | 「我请你做 X」 |
| `query-if` | 「P 是真的吗？」 |
| `query-ref` | 「X 的值是什么？」 |
| `propose` | 「我提议我们做 X」 |
| `accept-proposal` | 「我接受该提议」 |
| `reject-proposal` | 「我拒绝该提议」 |
| `agree` | 「我同意做 X」 |
| `refuse` | 「我拒绝做 X」 |
| `confirm` | 「我确认 P 为真」 |
| `disconfirm` | 「我否认 P」 |
| `not-understood` | 「你的消息无法解析」 |
| `cfp` | 「就 X 征集提案」 |
| `subscribe` | 「当 X 变化时通知我」 |
| `cancel` | 「取消正在进行的 X」 |
| `failure` | 「我尝试了 X 但失败了」 |

完整列表见 `fipa00037.pdf`（FIPA ACL Message Structure）。重点不是背下来——重点是其中每一项都对应一个 LLM 协议最终会重新加入的原语。

### 规范的 FIPA-ACL 消息

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

七个字段承载协议信封；一个字段（`content`）承载载荷。其余字段恰恰就是你每次给 JSON 协议硬加 retries、threading 和 ontology 时重新发明的东西。

### 两个遗留平台

**JADE**（Java Agent DEvelopment framework，1999–2020 年代）是使用最广的 FIPA 兼容运行时。智能体继承一个基类、交换 ACL 消息、在容器内运行，并用「行为（behaviors）」协调。随附的交互协议库包含 contract-net、subscribe-notify、request-when 和 propose-accept。

**JACK**（Agent Oriented Software，商业产品）强调在 FIPA 消息之上的 BDI（Belief-Desire-Intention）推理。更形式化，采用更少。

两者都在 Web 技术栈吞掉多智能体用例后衰落了。MCP 和 A2A 则是 2026 年的运行时「容器」。

### FIPA 为何淡出

- **本体开销。** FIPA 要求共享本体来解析 `content`。就本体达成一致是长达数年的标准流程。而 Web 直接用 HTTP + JSON。
- **没人使用的形式语义。** SL（Semantic Language）给出了严格的真值条件，但大多数生产系统使用自由文本内容，无视形式化。
- **工具锁定。** JADE 只支持 Java；JACK 是商业产品。多语言团队绕开了两者。
- **互联网赢得了技术栈。** REST，然后是 JSON-RPC，然后是 gRPC，取代了 ACL 的传输层。

### LLM 复兴是 FIPA-lite

比较 FIPA 的 `request` 与 MCP 的 `tools/call`：

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

同样的信封，不同的语法。两者都携带：谁、发给谁、意图、载荷、关联 id。彼此之间都不是革命——它们是同一设计上的不同取舍。

Liu 等人 2025 年的综述（「A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP」，arXiv:2505.02279）明确指出了这条谱系：MCP 对应工具使用的言语行为，A2A 对应智能体对等的言语行为，ACP 对应审计追踪的言语行为，ANP 对应去中心化身份的扩展。新规范是 ACL 的后代，只是换成了 JSON 语法、放宽了语义。

### 直白地说清取舍

**FIPA 给了你、而现代规范丢掉的东西：**

- 形式语义——你可以证明 `inform` 蕴含发送者相信该内容。
- 一份权威的 performative 目录——你不必重新争论「我们该不该有 `cancel`？」。
- 数十年的交互协议模式——contract-net、subscribe-notify、propose-accept——且带有已知的正确性性质。

**现代规范给了你、而 FIPA 没有的东西：**

- 与所有现代工具兼容的 JSON 原生载荷。
- LLM 无需手工编码本体即可解释的自然语言内容。
- Web 技术栈传输（HTTP、SSE、WebSocket）。
- 通过实时 MCP `server/discover` 和 A2A Agent Card 进行能力发现。

用更宽松的意图语义换取更简单的实现。这就是确切的取舍。

### 值得移植的交互协议

FIPA 附带约 15 种交互协议。其中三种值得带入 LLM 多智能体系统：

1. **合同网协议（Contract Net Protocol，CNP）。** 管理者发出 `cfp`（征集提案）；投标者用 `propose` 回应；管理者接受或拒绝。这是经典的任务市场模式（第 16 阶段 · 16 谈判）。
2. **Subscribe/Notify。** 订阅者发送 `subscribe`；每当主题变化，发布者发送 `inform`。这就是 2026 年的每一个事件总线。
3. **Request-When。** 「当条件 Y 成立时做 X。」带前置条件的延迟动作。2026 年的对应物是持久化工作流引擎中的延迟任务（第 16 阶段 · 22 生产扩展）。

每一个都能干净地映射到现代消息队列、HTTP + 轮询，或 SSE 流式传输。

### 丢掉本体会坏掉什么

没有共享本体，智能体从自然语言内容中推断含义。2026 年有记录的失败模式是**语义漂移（semantic drift）**：两个智能体把同一个词（`"customer"`）用于微妙不同的概念，接收方的智能体按错误理解行动，没有任何 schema 校验器能抓住它。FIPA 的本体要求本会在解析时拒绝这条消息。

不引入完整本体的缓解措施：

- 对 `content` 使用 JSON Schema——在传输层拒绝结构错误。
- 类型化产物（A2A）——拒绝错误的模态。
- 在信封中显式写出 performative——即使内容是自然语言，也让意图无歧义。

### 2026 年规范映射到言语行为遗产

| 现代规范 | FIPA 对应物 | 保留了什么 | 丢掉了什么 |
|---|---|---|---|
| MCP `tools/call` | `request` | 显式意图、关联 id | 形式语义、本体 |
| MCP `resources/read` | `query-ref` | 显式意图、关联 id | 形式语义 |
| A2A Task lifecycle | contract-net + request-when | 异步生命周期、状态转换 | 形式化完整性保证 |
| A2A streaming events | subscribe/notify | 异步推送 | 类型化谓词订阅 |
| CA-MCP shared context | blackboard（Hayes-Roth 1985） | 多写入者共享内存 | 逻辑一致性模型 |
| NLIP | natural-language content | LLM 原生 | schema |

从上到下读这张表，模式是：保留结构原语，丢掉形式化，让 LLM 去糊弄歧义。

```figure
sw-contract-net
```

## 构建它

`code/main.py` 实现了一个纯 stdlib 的 FIPA-ACL 翻译器。它编解码规范的 ACL 信封，并展示每种 MCP / A2A 消息形状如何归约为同样的七个字段。演示会：

- 将五条 MCP 风格和 A2A 风格的消息编码为 FIPA-ACL。
- 将 FIPA-ACL 解码回现代等价物。
- 使用 `cfp`、`propose`、`accept-proposal`、`reject-proposal`，在一个管理者和三个投标者之间运行一次玩具级合同网协商。

运行：

```
python3 code/main.py
```

输出是并排的跟踪记录，先展示每条现代消息在 2026 JSON 形式和 FIPA-ACL 形式下的样子，然后是一次合同网投标的往返。同样的协议原语在往返中保留下来；只有语法不同。

## 使用它

`outputs/skill-fipa-mapper.md` 是一个技能，它能读取任何智能体协议规范并产出 FIPA-ACL 映射。在采用新协议之前用它来回答：「这真的是新东西，还是换成了 JSON 语法的 `inform`？」

## 交付它

不要把 FIPA-ACL 带回来。把它的检查清单带回来：

- 每条消息的意图原语（performative）是什么？
- 是否有用于请求-响应和取消的关联 id？
- 是否有明确的内容语言（JSON-RPC、纯文本、结构化类型化产物）？
- 交互协议是否是一等公民，还是你在从头重新实现 contract-net？
- 当两个智能体对内容含义产生分歧时会发生什么（语义漂移）？

在把任何新协议交付生产之前，记录这五个问题的答案。

## 练习

1. 运行 `code/main.py`。观察往返编码。指出哪个 FIPA performative 对应 `tools/call`、`resources/read` 和 A2A 任务创建。
2. 用 `cancel` performative 扩展合同网演示，让管理者可以在投标中途撤回任务。`cancel` 解决了什么仅靠重试无法解决的失败情形？
3. 阅读 FIPA ACL Message Structure（http://www.fipa.org/specs/fipa00037/）第 4.1–4.3 节。选一个本节课未覆盖的 performative，描述其现代 JSON-RPC 对应物。
4. 阅读 Liu 等人，arXiv:2505.02279。对 MCP、A2A、ACP、ANP 中的每一个，列出它们保留和丢弃的 FIPA performative 家族。
5. 为你自己系统中 `request` performative 的 `content` 字段设计一个最小 JSON Schema。这个 schema 给了你哪些纯自然语言没有的东西，代价又是什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Speech act | 「做出行动的话语」 | Austin/Searle：话语即行动。ACL 的理论父辈。 |
| FIPA | 「那个旧的 XML 玩意儿」 | IEEE Foundation for Intelligent Physical Agents。2000 年标准化了 ACL。 |
| ACL | 「Agent Communication Language」 | FIPA 的信封格式：performative + content + metadata。 |
| Performative | 「动词」 | 消息的意图类别：`inform`、`request`、`propose`、`cfp` 等。 |
| KQML | 「FIPA 的前身」 | Knowledge Query and Manipulation Language（1993）。更简单、更窄。 |
| Ontology | 「共享词汇表」 | 对内容语言所谈论概念的形式化定义。 |
| SL0 / SL1 | 「FIPA 内容语言」 | Semantic Language 第 0 级和第 1 级——形式化内容语言家族。 |
| Contract Net | 「任务市场」 | 管理者发出 cfp；投标者 propose；管理者接受。经典交互协议。 |
| Interaction protocol | 「消息模式」 | 具有已知正确性的一串 performative 序列：request-when、subscribe-notify 等。 |

## 延伸阅读

- [Liu 等人 — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1)——2025 年把现代规范与 FIPA 遗产联系起来的权威综述
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/)——2000 年批准的信封格式
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/)——完整的 performative 目录
- [MCP specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28)——当前无状态的工具使用版本，等价于 `request`/`query-ref`
- [A2A specification](https://a2a-protocol.org/latest/specification/)——现代智能体对等版本，等价于 contract-net 和 subscribe-notify
