# 共享内存与黑板模式

> 2026 年的多 Agent 系统中并存两种做法：**消息池（message pool）**（每个人都能看到所有人的消息，如 AutoGen GroupChat 或 MetaGPT）和**带订阅的黑板（blackboard with subscription）**（Agent 订阅相关事件，如 Context-Aware MCP 或 Matrix 框架）。两者都是多 Agent 系统中唯一有状态的部分——这也意味着两者都是有趣 bug 的栖身之所。典型的失效模式是**内存投毒（memory poisoning）**：一个 Agent 幻觉出一个「事实」，其他 Agent 将其当作已验证结论，准确率随之逐渐衰减，而这种衰减比直接崩溃难调试得多。本课用标准库构建这两种结构，注入一次投毒攻击，并演示在生产中真正有效的三种缓解手段。

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## 问题

多 Agent 系统需要一个让 Agent 共享事实的地方。最直白的选项是「把所有东西都放在消息里传递」——但这等于用额外的复制重新发明了共享状态。另一个是「给每个人一个全局日志」——但全局日志会无界增长，而且很容易被污染。第三种是「为每个 Agent 投射一个视图」——可扩展，但 schema 负担重。

当某个 Agent 产生幻觉，并把幻觉写入共享状态时，每个读到该状态的下游 Agent 都会把这个幻觉当作事实。等到人类注意到，推理链已经深入五步，而根因是系统写下的第三条消息。调试多 Agent 的准确率衰减，比调试一次崩溃更难。

这就是内存投毒。它是 MAST 分类法（Cemri 等，arXiv:2503.13657）中记录第二多的失效族，而且是结构性的：任何缺少来源追踪和不可写校验器的共享内存设计，最终都会出现这一问题。

## 概念

### 两种主要拓扑

**全量消息池。** 每个 Agent 读取每条消息。AutoGen GroupChat 和 MetaGPT 采用这种设计。简单、透明、可检查，但超过约 10 个 Agent 就无法扩展，因为每个 Agent 的上下文都会被其他 Agent 的工作填满。

```
agent-A ──write──▶ ┌────────────────┐ ◀──read── agent-D
                   │ message pool   │
agent-B ──write──▶ │                │ ◀──read── agent-E
                   │ (global log)   │
agent-C ──write──▶ └────────────────┘ ◀──read── agent-F
```

**带订阅的黑板。** Agent 声明对哪些主题感兴趣；底层只路由相关消息。CA-MCP（arXiv:2601.11595）和 Matrix 去中心化框架（arXiv:2511.21686）采用这种设计。扩展性更好，但需要事先设计 schema，订阅才有意义。

```
                   ┌─ topic: prices ──┐
agent-A ──pub────▶ │                  │ ──▶ agent-D (subscribed)
                   ├─ topic: orders ──┤
agent-B ──pub────▶ │                  │ ──▶ agent-E (subscribed)
                   ├─ topic: alerts ──┤
agent-C ──pub────▶ │                  │ ──▶ agent-F (subscribed)
                   └──────────────────┘
```

### 各自何时胜出

- **全量消息池**在 Agent 数量少（< 10）、异构、对话时程短时胜出。当所有人都能看到一切时，理清谁说了什么就变得很简单。
- **黑板**在 Agent 数量多、角色同构但实例众多（集群）、对话长时间运行时胜出。路由能节省 token 成本并减少上下文污染。

生产系统常常混合使用：顶层用小规模全量池（规划层），下层用黑板（工作层）。

### 一个场景看懂内存投毒

三个 Agent 协作完成一项研究任务。Agent A 负责检索。Agent B 负责摘要。Agent C 负责分析。

1. A 抓取了一个页面，并向共享状态写入：「该研究报告了 42% 的准确率提升。」
2. 抓取的页面实际写的是「4.2% 的提升」。A 幻觉出了一个小数点。
3. B 读取共享状态后写道：「据报告准确率大幅提升 42%（来源：A）。」
4. C 读取共享状态后写道：「建议采用——42% 的提升具有变革性。」
5. 最终报告引用了一个根本不存在的 42%。

没有 Agent 崩溃。没有测试失败。系统「正常工作了」。幻觉通过共享状态，从一个 Agent 的上下文进入了每个下游 Agent 的推理。

### 为什么这是结构性问题

没有共享状态时，Agent A 的幻觉留在 A 的上下文里。下游 Agent 会重新抓取或重新推导，可能发现这个错误。而使用朴素的共享状态，A 的上下文变成了所有人的上下文，幻觉就被洗白成了事实。

问题不在于共享状态本身——而在于**缺少来源追踪、缺少独立校验器**的共享状态。三种缓解手段针对此：

1. **每次写入都记录来源。** 共享状态中的每个条目都记录是谁写的、何时写的、在什么 prompt 下写的，以及（如果适用）Agent 引用了什么来源。下游 Agent 带着对来源的怀疑去阅读。
2. **对写入做版本化；视为只追加。** 更正是一条取代旧条目的新条目，而不是原地修改。审计轨迹得以保留。
3. **至少保留一个无法写入共享状态的 Agent。** 一个只读校验器 Agent 抽样检查条目，重新抓取来源，并标记不一致。由于它无法写入消息池，它就不会被消息池投毒。

### 黑板的历史先例（Hayes-Roth，1985）

黑板模式比 LLM Agent 早出现了四十年。Hayes-Roth（1985，"A Blackboard Architecture for Control"）描述了观察全局黑板的专家知识源（Knowledge Source），它们贡献部分解并触发其他知识源。2026 年的黑板（CA-MCP、Matrix）是同样的模式，只是知识源换成了 LLM Agent，部分解换成了 JSON blob。旧文献中已有针对写竞争、机会式控制和一致性的解决方案，而现代系统正在重新发现它们。

### 投射与全量视图

纯黑板给每个订阅者相同的投射（按主题限定）。更激进的设计是**按 Agent 投射**：每个 Agent 得到为其角色定制的视图。LangGraph 的 state reducer 是 2026 年的典型实现——reducer 函数把全局状态折叠成角色专属的切片。

按 Agent 投射扩展性更好，但需要 schema。没有 schema，你就得在每个 Agent 的 prompt 里重建临时的投射逻辑。

### 写竞争模式

多个 Agent 同时写入是并发问题，而不只是 LLM 问题。三种模式行之有效：

- **顺序写入者（单生产者）。** 所有写入都经由一个协调者 Agent 串行化。简单，但会成为瓶颈。
- **带版本号的乐观并发。** 每个条目都有版本；版本不匹配时写入失败并重试。经典的数据库技术。
- **主题分区。** 不同 Agent 拥有不同主题。没有跨主题竞争。需要设计好分区边界。

多数 2026 年的框架默认采用顺序写入者，因为 LLM 调用足够慢，竞争很少发生，瓶颈也不明显。

### 不可写的校验器

最关键的缓解手段是只读校验器。实现规则：

- 校验器与团队共享状态（读取黑板或消息池）。
- 校验器没有共享状态的写句柄——只能写入单独的校验通道。
- 校验器独立抓取写入中引用的来源，标记分歧。
- 校验器自身的输出路由给人类或独立的决策 Agent，绝不回灌到消息池。

没有这种隔离，校验器的输出就会成为消息池里的新条目，这意味着被投毒的消息池会毒害校验器，进而毒害它的校验结果。

```figure
swarm-blackboard
```

## 构建它

`code/main.py` 用标准库 Python 实现了两种拓扑，外加一个玩具级的投毒攻击和三种缓解手段。

- `MessagePool` —— 线程安全的只追加日志，支持全量读出。
- `Blackboard` —— 按主题键控的发布/订阅，支持按 Agent 订阅。
- `ProvenanceEntry` —— 每次写入都记录（writer、timestamp、prompt_hash、source_uri）。
- `PoisoningScenario` —— 运行一个三 Agent 的研究任务，Agent A 幻觉出一个小数点。打印最终报告。
- `Verifier` —— 一个只读 Agent，重新抓取来源并标记不一致。在有校验器的情况下运行同一场景。

运行：

```
python3 code/main.py
```

预期输出：
- 第 1 次运行（无校验器）：幻觉出的 42% 传播到了最终报告。
- 第 2 次运行（有校验器）：校验器标记出不一致，消息池被标记为 "flagged"，最终报告包含撤回说明。

## 使用它

`outputs/skill-memory-auditor.md` 是一个技能，用于审计任何多 Agent 系统共享内存设计的来源追踪、版本化和校验器隔离。上线前对新多 Agent 架构运行它。

## 交付它

对于任何共享内存设计：

- 每次写入都记录来源：`(writer, timestamp, prompt_hash, tool_calls_cited, source_uri)`。
- 让日志只追加。更正是新条目，并引用被取代的条目。
- 至少部署一个具有独立来源访问能力的只读校验器 Agent。
- 将校验器输出路由到单独的通道，而不是回到共享池。
- 记录写入中属于「取代」的比例——比例上升是幻觉模式的早期证据。

## 练习

1. 运行 `code/main.py`。确认第 1 次运行传播了幻觉，第 2 次运行抓住了它。
2. 增加第二个幻觉：Agent B 编造一个数据集规模。校验器应当能抓住两者，而不需要针对任一个手工调参。
3. 把全量池改成带主题分区（`prices`、`summaries`、`analyses`）的黑板。主题分区让哪些投毒场景更难实施，又对哪些场景没有帮助？
4. 阅读 Hayes-Roth（1985，"A Blackboard Architecture for Control"）。找出论文中本课未讨论、而 2026 年系统会受益的两个控制模式。
5. 阅读 CA-MCP（arXiv:2601.11595）。把它的 Shared Context Store 映射到 `code/main.py` 中的 MessagePool 或 Blackboard 类。CA-MCP 在其上增加了哪些原语？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|----------------|------------------------|
| Message pool | "共享聊天历史" | 每个 Agent 都读取的只追加日志。完全透明，扩展性差。 |
| Blackboard | "共享工作区" | 按主题键控的发布/订阅。Agent 订阅相关主题。扩展得更远。 |
| Provenance | "谁写了什么" | 每次写入的元数据：writer、timestamp、prompt、sources。 |
| Memory poisoning | "幻觉在扩散" | 一个 Agent 的错误进入共享状态，下游 Agent 将其当作事实。 |
| Append-only | "不做原地更新" | 更正是取代旧条目的新条目。保留审计轨迹。 |
| Unwritable verifier | "独立审计员" | 只读 Agent，重新抓取来源并标记不一致。 |
| Projection | "限定范围的视图" | 从全局状态计算出的按 Agent 视图。LangGraph reducer 是典型例子。 |
| Knowledge Source | "专家 Agent" | Hayes-Roth 1985 年对黑板参与者的称呼。 |

## 延伸阅读

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) —— MAST 分类法；内存投毒是协调失效的一个子族
- [CA-MCP — Context-Aware Multi-Server MCP](https://arxiv.org/abs/2601.11595) —— 面向协调式 MCP 服务端的 Shared Context Store
- [Matrix — decentralized multi-agent framework](https://arxiv.org/abs/2511.21686) —— 无中心编排器的、基于消息队列的黑板
- [LangGraph state and reducers](https://docs.langchain.com/oss/python/langgraph/workflows-agents) —— 生产中的按 Agent 投射模式
- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) —— 来自生产部署的来源追踪与校验笔记
