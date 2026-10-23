# 浏览器智能体与长时程 Web 任务

> ChatGPT agent（2025 年 7 月）把 Operator 和深度研究合并为一个浏览器/终端智能体，并以 68.9% 创下 BrowseComp 的 SOTA。OpenAI 于 2025 年 8 月 31 日关停 Operator——这是产品层的整合。Anthropic 收购 Vercept 后，Claude Sonnet 在 OSWorld 上的成绩从不足 15% 提升到 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修正了原始 WebArena 中 11.3 个百分点的假阴性率，并发布了 258 任务的 Hard 子集。这些数字是真实的，攻击面也是真实的：OpenAI 的 preparedness 负责人公开表示，针对浏览器智能体的间接提示注入「不是一个可以完全修补的 bug」。2025–2026 年已有记录的攻击：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks），以及 Perplexity Comet 中的一键劫持。

**Type:** Learn
**Languages:** Python (stdlib, indirect prompt-injection attack surface model)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~45 minutes

## 问题

浏览器智能体是一种长时程智能体：它读取不可信内容，并采取有后果的动作。智能体访问的每个页面都是用户没有写过的输入。每个页面上的每个表单都是潜在的命令通道。2025–2026 年的攻击语料表明这并非假设：Tainted Memories 让攻击者通过精心构造的页面把恶意指令绑定到智能体的记忆中；HashJack 把命令藏在智能体访问的 URL 片段里；Perplexity Comet 的劫持只需一次点击。

防御图景令人不安。OpenAI 的 preparedness 负责人把大家心照不宣的话说得很直白：间接提示注入「不是一个可以完全修补的 bug」。原因在于攻击存在于智能体的「阅读 vs 行动」边界上，而这个边界在架构上是模糊的——模型读到的每一个 token，原则上都可以被当作指令来读。

本课梳理攻击面、梳理基准版图（BrowseComp、OSWorld、WebArena-Verified），并建模一个最小的间接提示注入场景，以便你在第 14 课和第 18 课中推理真实防御。

## 概念

### 2026 年版图，每个系统一段

**ChatGPT agent（OpenAI）。** 2025 年 7 月推出。统一了 Operator（浏览）和 Deep Research（数小时的研究）。2025 年 8 月 31 日关停独立的 Operator。BrowseComp 达到 68.9% 的 SOTA；在 OSWorld 和 WebArena-Verified 上成绩也很强。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic 收购 Vercept 专注于计算机使用能力。把 Claude Sonnet 在 OSWorld 上从 <15% 提升到 72.5%。Claude Computer Use 以工具 API 的形式提供。

**Gemini 3 Pro 与 Browser Use（DeepMind）。** Browser Use 集成提供了计算机使用控制能力；FSF v3（2026 年 4 月，第 20 课）专门追踪 ML 研发领域的自主性。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 修复了一个有充分记录的问题：原始 WebArena 有约 11.3% 的假阴性率（实际已解决却被判为失败的任务）。Verified 版本用人工筛选的成功标准重新评分，并增加了 258 任务的 Hard 子集（ICLR 2026 论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp、OSWorld 与 WebArena 对比

| 基准 | 衡量内容 | 时程 |
|---|---|---|
| BrowseComp | 在时间压力下于开放 Web 上查找特定事实 | 分钟级 |
| OSWorld | 智能体操作完整桌面（鼠标、键盘、shell） | 数十分钟 |
| WebArena-Verified | 模拟站点中的事务型 Web 任务 | 分钟级 |
| Hard 子集 | 带多页面状态转换的 WebArena-Verified 任务 | 数十分钟 |

它们衡量不同的维度。BrowseComp 高分说明智能体能找到事实，但不能说明它能订机票。OSWorld 更接近「它在我的桌面上能工作吗」。WebArena-Verified 更接近「它能走完一个流程吗」。任何生产决策都需要与任务分布相匹配的基准。

### 攻击面逐项列举

1. **间接提示注入。** 不可信的页面内容中包含指令。智能体读到它们，并执行它们。公开案例：2024 年 Kai Greshake 等、2025 年 Tainted Memories 论文、2026 年 HashJack（Cato Networks）。
2. **URL 片段 / 查询串注入。** 抓取的 URL 中的 `#fragment` 或查询串包含命令。它从不被可见地渲染，但仍处于智能体的上下文中。
3. **记忆绑定攻击。** 页面指示智能体写入一条持久记忆（第 12 课讲解持久状态）。下一次会话，这条记忆在没有任何可见触发的情况下放出载荷。
4. **针对已认证会话的 CSRF 式攻击。** Tainted Memories 这一类：智能体在某个站点已登录；攻击者的页面发出改变状态的请求，智能体带着用户的 cookie 执行它们。
5. **一键劫持。** 一个视觉上无害的按钮搭载了智能体会跟随的载荷。Comet 这一类。
6. **智能体宿主面上的 CSP 漏洞。** 渲染层和工具层本身也可能是攻击向量；浏览器智能体内部的浏览器栈非常宽。

### 为什么「无法完全修补」

这种攻击与智能体的能力同构。智能体必须阅读不可信内容才能完成工作。它读到的任何内容都可能包含指令。它遵循的任何指令都可能与用户的实际请求不一致。防御措施（信任边界、分类器、工具允许列表、对高后果动作的 HITL）会提高攻击成本、缩小影响半径，但无法封闭这一类攻击。

这与勒布定理（第 8 课）的推理模式相同：智能体无法证明下一个 token 是安全的；它只能搭建一个让不安全 token 更容易被检测的系统。

### 真正能落地的防御姿态

- **读 / 写边界。** 读取永远不产生后果。写入（提交表单、发布内容、调用有副作用的工具）在发起内容来自信任边界之外时，需要新的人工批准。
- **按任务配置工具允许列表。** 智能体可以浏览；除非该工具为此任务显式启用，否则它不能发起电汇。第 13 课讲解预算。
- **会话隔离。** 浏览器智能体会话只使用范围受限的凭据。不使用生产认证，不使用个人邮箱。每个 HTTP 请求的日志都保留以供审计。
- **内容清洗器。** 抓取的 HTML 在拼接到模型上下文之前，会剔除已知的恶意模式。（减少容易的攻击，但无法阻止复杂的载荷。）
- **对高后果动作启用 HITL。** 先提议后提交模式（第 15 课）。
- **在记忆上放置 canary token。** 如果某条记忆条目被触发，用户会看到（第 14 课）。

```figure
injection-boundary
```

## 使用它

`code/main.py` 建模一次针对三个合成页面的微型浏览器智能体运行。一个页面是良性的，一个在可见文本中带有直接的提示注入内容，一个带有 URL 片段注入（不可见，但位于智能体上下文中）。脚本展示 (a) 朴素智能体会怎么做，(b) 读 / 写边界能抓住什么，(c) 清洗器能抓住什么，(d) 两者都抓不住什么。

## 交付它

`outputs/skill-browser-agent-trust-boundary.md` 界定一个候选浏览器智能体部署的范围：它触达哪些信任区域、被授权写入什么，以及首次运行前哪些防御必须就位。

## 练习

1. 运行 `code/main.py`。指出哪种攻击是清洗器能抓住而读 / 写边界抓不住的，哪种攻击只有读 / 写边界能抓住。

2. 扩展清洗器以检测一类 HashJack 式 URL 片段注入。在带有合法片段的良性 URL 上测量误报率。

3. 选一个你熟悉的真实浏览器智能体工作流（例如「订机票」）。列出每一次读取和每一次写入。标出哪些写入需要 HITL，并说明原因。

4. 阅读 WebArena-Verified 的 ICLR 2026 论文。找出一个原始 WebArena 评分不可靠的任务类别，并解释 Verified 子集如何解决它。

5. 为浏览器智能体场景设计一个记忆 canary。你会存什么、存在哪里、什么会触发警报？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| 间接提示注入 | 「页面文本有问题」 | 智能体读取的页面中，不可信内容包含智能体会执行的指令 |
| Tainted Memories | 「记忆攻击」 | 智能体把攻击者提供的指令写入持久记忆；下次会话时触发 |
| HashJack | 「URL 片段攻击」 | 藏在 URL 片段 / 查询串中的载荷处于智能体上下文中，但不会被可见地渲染 |
| 一键劫持 | 「恶意按钮」 | 可见的交互元素搭载智能体会执行的后续载荷 |
| BrowseComp | 「网络搜索基准」 | 在开放 Web 上查找特定事实；分钟级时程 |
| OSWorld | 「桌面基准」 | 完整操作系统控制；多步 GUI 任务 |
| WebArena-Verified | 「修复后的 Web 任务基准」 | ServiceNow 重新评分并附带 Hard 子集的 WebArena |
| 读 / 写边界 | 「副作用门控」 | 读取永远无后果；内容来自信任边界之外时，写入需要新的批准 |

## 延伸阅读

- [OpenAI —— Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) —— Operator 与深度研究的合并；BrowseComp SOTA。
- [OpenAI —— Computer-Using Agent](https://openai.com/index/computer-using-agent/) —— Operator 的谱系以及后来成为 ChatGPT agent 的架构。
- [Zhou 等 —— WebArena](https://webarena.dev/) —— 原始基准。
- [WebArena-Verified（OpenReview）](https://openreview.net/forum?id=94tlGxmqkN) —— ICLR 2026 修复子集论文。
- [Anthropic —— 在实践中测量智能体自主性](https://www.anthropic.com/research/measuring-agent-autonomy) —— 包含针对计算机使用智能体的攻击面讨论。
