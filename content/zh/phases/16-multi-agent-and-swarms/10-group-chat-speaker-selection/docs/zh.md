# 群聊与发言者选择

> 共享会话式编排把 N 个 Agent 放进同一段对话；由一个选择器函数（LLM、轮询或自定义）决定下一个由谁发言。这是涌现式多 Agent 对话的原型——Agent 并不在静态图中知道自己的角色，它们只是对共享池作出反应。AutoGen GroupChat 与 AG2 GroupChat 是参考实现：AutoGen v0.2 的 GroupChat 语义在 AG2 分叉中得到保留；AutoGen v0.4 则把它重写为事件驱动的 actor 模型。微软在 2026 年 2 月把 AutoGen 置于维护模式，并将其与 Semantic Kernel 合并为 Microsoft Agent Framework（2026 年 2 月 RC）。GroupChat 这一原语在 AG2 与 Microsoft Agent Framework 中都得以延续——学一次，到处可用。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## 问题

当工作流是已知的时候，静态图（LangGraph）很好用。真实的对话并不是静态的：有时程序员问审查者，有时问研究者，有时问写作者。把每一种可能的转交都硬编码，会导致边的组合爆炸。你需要的是*对共享池作出反应的 Agent*，再由某个函数决定下一个谁发言。

AutoGen GroupChat 做的正是这件事。

## 概念

### 结构形态

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

每个 Agent 都能看到每一条消息。每一轮都会调用一次选择器函数，挑出下一个发言者。

### 三种选择器类型

**轮询（Round-robin）。** 固定循环。确定性。随 N 线性扩展，但无视上下文——即使话题是法务审查，程序员也会轮到发言。

**LLM 选择。** 调用 LLM 读取最近的共享池，返回最合适的下一个发言者。能感知上下文，但慢：每一轮都要多一次 LLM 调用。这是 AutoGen 的默认方式。

**自定义。** 用 Python 函数实现任意逻辑。典型做法是 LLM 选择加兜底规则（例如「程序员发言之后总是轮到验证者」）。

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有选择器。当一个 Agent 结束发言，管理者调用选择器，选择器返回下一个 Agent。循环持续到满足终止条件。

### 终止

三种常见模式：

- **最大轮数。** 对总轮次设硬上限。
- **"TERMINATE" token。** Agent 可以发出哨兵消息；一旦出现，管理者就停止。
- **目标达成检查。** 每轮运行一个轻量验证器，完成时停止聊天。

### 谱系：分叉与合并

2025 年初，微软围绕事件驱动的 actor 模型对 AutoGen 进行了大规模重写（v0.4）。社区把 AutoGen v0.2 的 GroupChat 语义分叉为 AG2，保留了早期采用者已经集成的 API。

2026 年 2 月，微软宣布 AutoGen 进入维护模式，事件驱动的 actor 模型并入 **Microsoft Agent Framework**（2026 年 2 月 RC，现与 Semantic Kernel 合并）。GroupChat 概念在两条线中都保留下来，只是实现细节不同。对于兼容 v0.2 的代码，AG2 是首选的上游。

### GroupChat 适用的情况

- **涌现式对话。** 你不想把每一种可能的下一发言者都预先接好。
- **角色混合的任务。** 程序员问研究者，研究者问档案员，档案员再问回程序员。流程不是 DAG。
- **探索式解题。** 想象的是「头脑风暴会议」，而不是「流水线」。

### 失效的情况

- **严格确定性。** LLM 选择器可能不一致。同样的 prompt，不同次运行，下一个发言者可能不同。
- **谄媚级联。** Agent 会听从发言最自信的那个。需要显式地用 prompt 加以反制。
- **上下文膨胀。** 每个 Agent 都要读每一条消息；10 轮之后上下文就非常庞大。用投影（projection，第 15 课）来限定各自可见的范围。
- **热门发言者。** 某个 Agent 因为选择器偏爱它的专长而主导整段对话。把发言均衡作为选择器的一项特性引入。

### 群聊 vs 监督员

同样的原语，不同的默认设置：

- 监督员：一个 Agent 规划，其他人执行。选择器的逻辑是「问规划者该做什么」。
- 群聊：所有 Agent 地位对等；选择器是作用在共享池上的函数。

两者都使用第 04 课中的四个原语。群聊默认采用 LLM 选择的编排方式与全池共享状态。

```figure
swarm-speaker
```

## 构建它

`code/main.py` 用标准库从零实现一个 GroupChat。三个 Agent（程序员、审查者、管理者），包含轮询与 LLM 选择两种变体，并以 `TERMINATE` token 作为终止信号。

演示会打印对话记录，以及两种变体下选择器的决策轨迹。

运行：

```
python3 code/main.py
```

## 使用它

`outputs/skill-groupchat-selector.md` 为给定任务配置 GroupChat 选择器——轮询、LLM 选择还是自定义，以及选择器应使用哪些输入（最近消息、Agent 专长、轮次计数）。

## 交付它

检查清单：

- **最大轮数上限。** 永远要设。一般任务是 10-20 轮。
- **发言均衡指标。** 统计每个 Agent 的轮次；不均衡超过阈值时告警。
- **终止 token。** 用 `TERMINATE`，或设置专门的验证者 Agent。
- **投影或限定范围的记忆。** 大约 10 条消息之后，考虑只给每个 Agent 一个有范围的视图，以防止上下文膨胀。
- **选择器日志。** 对 LLM 选择的变体，同时记录选择器的输入和它的选择。否则调试无从下手。

## 练习

1. 运行 `code/main.py`。比较轮询与 LLM 选择下的对话。两种方式下分别由哪个 Agent 主导？
2. 在选择器中增加「每个 Agent 最大发言次数」规则。它如何影响对话记录？
3. 实现目标达成终止：当审查者返回 "approved" 时停止。它有多大比例会在轮数上限之前触发？
4. 阅读 AutoGen 关于 GroupChat 的稳定版文档（https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html）。找出 `GroupChatManager` 使用的默认选择器。
5. 阅读 AG2 仓库（https://github.com/ag2ai/ag2），对比其 v0.2 GroupChat 与 v0.4 的事件驱动版本。v0.4 增加了哪一项具体性质（吞吐量、容错性、可组合性）？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| GroupChat | 「Agent 在同一个聊天室」 | 共享消息池 + 选择器函数。AutoGen / AG2 的原语。 |
| 发言者选择（Speaker selection） | 「下一个谁说话」 | 挑选下一个 Agent 的函数。轮询、LLM 选择或自定义。 |
| GroupChatManager | 「会议主持人」 | AutoGen 组件，持有选择器并驱动轮次循环。 |
| ConversableAgent | 「基础 Agent」 | AutoGen 基类；能收发消息的 Agent。 |
| 终止 token（Termination token） | 「停止」口令 | 结束聊天的哨兵字符串（通常是 `TERMINATE`）。 |
| 热门发言者（Hot speaker） | 「一个 Agent 独霸」 | 选择器反复挑中同一个 Agent 的失效模式。 |
| 上下文膨胀（Context bloat） | 「共享池无限增长」 | 每个 Agent 读取所有先前消息；上下文随轮次增长。 |
| 投影（Projection） | 「限定范围的视图」 | 按角色划分的共享池视图，用于防止上下文膨胀。 |

## 延伸阅读

- [AutoGen 群聊文档](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) —— 参考实现
- [AG2 仓库](https://github.com/ag2ai/ag2) —— 社区对 AutoGen v0.2 的延续
- [Microsoft Agent Framework 文档](https://learn.microsoft.com/en-us/agent-framework/) —— 合并后的继任者，2026 年 2 月 RC
- [AutoGen v0.4 发布说明](https://microsoft.github.io/autogen/stable/) —— 事件驱动 actor 模型重写的细节
