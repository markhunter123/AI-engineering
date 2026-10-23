# 人在回路：先提议后提交

> 2026 年关于 HITL 的共识是具体的。它不是「Agent 询问，用户点击 Approve」。它是先提议后提交：提议的操作以幂等键持久化到持久化存储；连同意图、数据血缘、触及的权限、爆炸半径和回滚计划一起呈现给审查者；只有在明确确认后才提交；执行后进行验证，确认副作用确实发生。LangGraph 的 `interrupt()` 加 PostgreSQL checkpointing、Microsoft Agent Framework 的 `RequestInfoEvent`、Cloudflare 的 `waitForApproval()` 都实现同一形态。典型的失效模式是橡皮图章式批准：「Approve?」未经审查就被点击。有文档记录的缓解措施是带明确检查清单的挑战-应答。

**Type:** Learn
**Languages:** Python (stdlib, propose-then-commit state machine with idempotency)
**Prerequisites:** Phase 15 · 12 (Durable execution), Phase 15 · 14 (Tripwires)
**Time:** ~60 minutes

## 问题

一个 Agent 执行一个操作。用户必须决定：批准还是不批准。如果决策是瞬间的，它很可能不是审查。如果决策是结构化的，它很慢但可信。工程问题是：如何让结构化审查成为阻力最小的路径。

2023 年时代的 HITL 模式是同步提示：「Agent 想向 X 发送正文为 Y 的邮件——批准？」用户点击 Approve。所有人都觉得系统是安全的。实践中这个界面被大量橡皮图章化：用户批准得很快，批准几乎预测不了什么，而当 Agent 出错时，审计轨迹显示一长串用户无法回忆的批准记录。

2026 年的模式——先提议后提交——把 HITL 移到持久化基座上，附加结构化元数据，并要求明确提交。每个托管 Agent SDK 都提供一个版本：LangGraph `interrupt()`、Microsoft Agent Framework `RequestInfoEvent`、Cloudflare `waitForApproval()`。API 名称不同；形态相同。

## 概念

### 先提议后提交状态机

1. **提议。** Agent 产生一个提议的操作。持久化到持久化存储（PostgreSQL、Redis、Durable Object）。包括：
   - 意图（Agent 为什么这样做）
   - 数据血缘（是什么来源导致了这个提议）
   - 触及的权限（哪些 scope / 文件 / 端点）
   - 爆炸半径（最坏情况是什么）
   - 回滚计划（如果提交了，我们如何撤销）
   - 幂等键（每个提议唯一；重新提交返回同一条记录）
2. **呈现。** 审查者看到带有全部元数据的提议。审查者是人（不是 Agent 自我审查）。
3. **提交。** 明确确认。操作执行。
4. **验证。** 执行后，读回副作用并确认。如果验证步骤失败，系统处于已知的坏状态，告警启动。

### 幂等键

没有幂等键，瞬时故障后的重试可能重复执行一个已批准的操作。具体例子：用户批准「从 A 向 B 转账 100 美元」。网络抖动。工作流重试。用户只批准了一次，但转账执行了两次。幂等键把批准绑定到单一、唯一的副作用上；第二次执行是空操作。

这与 Stripe 和 AWS API 使用的幂等模式相同。把它复用到 Agent 批准上，在 Microsoft Agent Framework 文档中有明确说明。

### 持久性：为什么批准能比进程活得久

批准等待室是 Agent 不拥有的一块状态。工作流被暂停（第 12 课）。当批准到达时，工作流从正是那个点恢复。这就是为什么 LangGraph 把 `interrupt()` 与 PostgreSQL checkpointing 配对，而不只是内存状态——两天后到达的批准仍然能找到完好的工作流。

### 橡皮图章式批准与挑战-应答缓解措施

HITL 的默认 UI（「Approve」/「Reject」按钮）产生快速、没有真正审查的批准。有文档记录的缓解措施：挑战-应答检查清单，要求对特定问题给出肯定回答后才启用 Approve 按钮。具体形态：

- 「你理解这触及的是哪个资源吗？[ ]」
- 「你验证过爆炸半径可以接受吗？[ ]」
- 「如果失败，你有回滚计划吗？[ ]」

不是为官僚而官僚——而是强制函数。无法勾选的审查者要么要求澄清（升级），要么拒绝（安全默认）。Anthropic 的 Agent 安全研究明确把检查清单驱动的 HITL 列为橡皮图章批准模式的缓解措施。

### 什么算有后果

不是每个操作都需要先提议后提交。2026 年的指导：

- **有后果的操作**（始终 HITL）：不可逆写入、金融交易、对外通信、生产数据库变更、破坏性文件系统操作。
- **可逆操作**（有时 HITL）：本地文件编辑、staging 环境变更、有清晰回滚的可逆写入。
- **读取与检查**（从不 HITL）：读文件、列出资源、调用只读 API。

### 行动后验证

「提交运行了」不等于「副作用发生了」。网络分区和竞态条件可能产生一个自认为成功、而后端并未持久化的工作流。验证步骤在提交后重新读取目标资源以确认。这与带 `RETURNING` 子句的数据库事务或 `PutObject` 之后的 AWS `GetObject` 是同一模式。

### EU AI Act 第 14 条

第 14 条要求对欧盟的高风险 AI 系统进行有效的人类监督。「有效」不是装饰性的。监管措辞明确排除橡皮图章模式。在 Microsoft Agent Governance Toolkit 的合规文档中，带挑战-应答的先提议后提交是能通过第 14 条审查的形态。

```figure
mx-propose-then-commit
```

## 使用它

`code/main.py` 用 stdlib Python 实现一个先提议后提交状态机。持久化存储是一个 JSON 文件。幂等键是 (thread_id, action_signature) 的哈希。驱动程序模拟三种情况：干净的批准流程、瞬时故障后的重试（不得重复执行）、以及橡皮图章默认流程与挑战-应答流程的对比。

## 交付它

`outputs/skill-hitl-design.md` 审查拟议的 HITL 工作流是否符合先提议后提交形态，并标出缺失的元数据、幂等、验证或挑战-应答层。

## 练习

1. 运行 `code/main.py`。确认已批准提议的重试使用持久化记录，不会重新执行。现在把幂等键改成包含时间戳，展示重试会重复执行。

2. 扩展提议记录，加入 `rollback` 字段。模拟一个验证步骤失败的执行。展示回滚自动触发。

3. 阅读 Microsoft Agent Framework 的 `RequestInfoEvent` 文档。找出该 API 包含而玩具引擎缺少的一个元数据字段。添加它，并解释它防范什么。

4. 为某个具体操作（例如「发布到公开的 Twitter 账号」）设计一个挑战-应答检查清单。审查者必须回答哪三个问题？为什么是这三个？

5. 选一个同步「Approve?」提示就足够（不需要持久化存储）的案例。解释为什么，并说出你接受的风险类别。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 先提议后提交 | 「两阶段批准」 | 持久化提议 + 明确提交 + 验证 |
| 幂等键 | 「重试安全 token」 | 每个提议唯一；第二次执行空操作 |
| 数据血缘 | 「它从哪里来」 | 导致该提议的具体源内容 |
| 爆炸半径 | 「最坏情况」 | 操作出错时的影响范围 |
| 橡皮图章 | 「快速批准」 | 未经真正审查就点击「Approve」 |
| 挑战-应答 | 「强制检查清单」 | 审查者必须明确确认特定问题 |
| RequestInfoEvent | 「MS Agent Framework 原语」 | 带结构化元数据的持久 HITL 请求 |
| `interrupt()` / `waitForApproval()` | 「框架原语」 | LangGraph / Cloudflare 的同一形态等价物 |

## 延伸阅读

- [Microsoft Agent Framework —— 人在回路](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) —— `RequestInfoEvent`、持久化批准。
- [Cloudflare Agents —— 人在回路](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) —— `waitForApproval()` 与 Durable Objects。
- [Anthropic —— 在实践中衡量 Agent 自主性](https://www.anthropic.com/research/measuring-agent-autonomy) —— 作为长视野风险缓解措施的 HITL。
- [EU AI Act —— 第 14 条：人类监督](https://artificialintelligenceact.eu/article/14/) —— 高风险系统的监管基线。
- [Anthropic —— Claude 的宪法（2026 年 1 月）](https://www.anthropic.com/news/claudes-constitution) —— 围绕监督的宪法框架。
