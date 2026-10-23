# 自动化对齐研究（Anthropic AAR）

> Anthropic 让多组 Claude Opus 4.6 自主对齐研究者（Autonomous Alignment Researcher）在独立沙箱中并行工作，通过一个共享论坛协调，论坛日志存放在任何沙箱之外（因此 Agent 无法删除自己的记录）。在弱到强训练问题上，AAR 的表现超过了人类研究者。Anthropic 自己的总结指出，预设工作流常常限制 AAR 的灵活性并降低性能。自动化对齐研究正是那个压缩步骤，它把时间线压缩到 RSP 本意要检测的错位风险面前。

**Type:** Learn
**Languages:** Python (stdlib, parallel-research-forum simulator)
**Prerequisites:** Phase 15 · 05 (AI Scientist v2), Phase 15 · 04 (DGM)
**Time:** ~60 minutes

## 问题

对齐研究消耗大量人类研究者时间。可扩展监督、奖励规范或弱到强训练这样的问题，需要每次迭代数周才能完成的实验。随着前沿能力推进，对齐工作量增长得比合格研究者的供给更快。

自动化对齐研究（AAR）问：那些能力正在超越对齐的前沿模型，能否帮助缩小差距？Anthropic 2026 年关于一次 AAR 运行的弱到强训练研究的报告（alignment.anthropic.com/2026/automated-w2s-researcher/）是这类已部署系统最早的公开结果之一。

结果确实积极：AAR 解决一个研究问题，比人类基线更好。这个结果也把一个系统里浓缩了本阶段展开的所有担忧。如果对齐研究可以被自动化，那么损害防护措施的那些部分也可以。RSP 的 AI 研发能力阈值正是围绕这个循环写的。

## 概念

### 架构（公开描述）

- **并行 Agent。** 多个 Claude Opus 4.6 实例作为 AAR 运行，各自处于独立沙箱。沙箱限制文件系统、网络和进程面。
- **共享论坛。** 发现发布到一个共享论坛，其存储后端位于每个沙箱之外。Agent 可以读论坛；Agent 无法从自己的沙箱内删除或编辑论坛记录。
- **任务分配。** 一个人类设计的任务队列把研究问题分配给 AAR。AAR 提出分解、运行实验、撰写发现。
- **弱到强训练重点。** 发表的研究针对弱到强训练——用一个较弱的监督模型训练一个更强的学生模型。AAR 在这项任务上表现超过人类研究者。

### 为什么沙箱外日志很重要

日志完整性属性是受控实验与可能已被污染的实验之间的区别。如果 AAR 能编辑自己的日志，它们就能掩盖失败或可疑的实验。Anthropic 直接点明：该属性是信任研究输出的必要条件，而且验证起来并不平凡。

用基础设施的术语，对应物是只追加日志并写穿到外部存储（例如 S3 object-lock、不可变的 Datadog 日志）。Agent 没有修改过去记录的凭证。这与 2026 年企业安全团队用来构建抗勒索软件审计轨迹的模式相同。

### 预设工作流的权衡

Anthropic 的报告指出，预设工作流常常限制 AAR 灵活性并降低性能。换句话说：如果人类设计者一步步写研究计划，AAR 的表现就不如给它们高层目标和自由分解。这与 Agent 研究领域的结果一致（LangGraph、Microsoft 的 Agent Framework）——过度规定的脚手架会伤害有能力的 Agent。

安全含义很微妙。提高 AAR 性能的灵活性，也是允许未经许可的分解、新颖工具使用和目标漂移的灵活性。这个权衡是一个设计空间决策：多少灵活性既能买来足够超越人类的能力，又不给 Agent 把目标泛化到分布之外的空间？

### 压缩风险

RSP v3.0（第 19 课）引入 AI 研发能力阈值：以相对于人类 + AI 工具具有竞争力的成本，完全自动化 AI 研发 pipeline 的能力。DeepMind 的 FSF v3 包含类似的 ML 研发自主性等级。两个框架都把这一阈值视为升级控制的触发条件。

AAR 离阈值还差一步：它自动化了 pipeline 的一部分（针对特定、边界清晰任务的对齐研究），但没有自动化端到端的能力开发循环。时间线问题是这个差距闭合得有多快。

压缩的时间线是复利式失败的担忧。如果对齐研究与能力研究以相似速率复利，错位风险面至少以能力的同等速度增长。如果能力复利更快（历史趋势如此），差距就会扩大。这就是 AAR 是一种有条件的善的论据：每一项额外的对齐成果都能缩小差距，当且仅当研究过程可信。

### AAR 不能取代什么

人类研究者设定任务队列、审查结果并掌握宪制权威。AAR 加速的是 pipeline 中段，而不是两端。Anthropic 公开的输出既包含 AAR 贡献，也包含人类研究者对发表什么、撤回什么、完善什么的判断。

这符合第 15 课「先提议后提交」的模式，只是应用到了研究本身：AAR 提议；人类提交。

```figure
aar-forum
```

## 使用它

`code/main.py` 模拟一个多 AAR 研究论坛。三个独立的「研究 Agent」并行运行，向一个可选篡改尝试的只追加论坛发布发现。

观察：

- 沙箱外日志如何捕获日志篡改尝试。
- 自由分解的任务分配如何在更难的问题上胜过固定工作流。
- 如果日志不具备防篡改能力，一个被攻陷的 AAR 如何污染论坛信号。

## 交付它

`outputs/skill-aar-deployment-review.md` 是一份自动化对齐研究 pipeline 的部署前审查。

## 练习

1. 运行 `code/main.py`。比较「固定工作流」与「自由分解」设置。自由分解总是赢吗，还是存在固定工作流也有竞争力的问题类别？

2. 修改模拟器，让一个 Agent 尝试日志篡改。确认只追加日志检测到了它。写一段话描述检测签名在日志中具体长什么样。

3. 阅读 Anthropic 的弱到强 AAR 报告。指出 AAR 击败人类研究者的具体子任务。是什么让它适合自动化？

4. 设计一个任务队列分配策略，在 AAR 灵活性（更好的结果）与预设工作流约束（更易审计）之间平衡。描述你会如何对两者做 A/B 测试。

5. 阅读 RSP v3.0 的 AI R&D-4 阈值。用一段话描述你认为什么会越过它，而 AAR 目前没有。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| AAR | 「自动化对齐研究者」 | 在沙箱中自主运行于对齐问题的 Claude Opus 4.6 实例 |
| 弱到强训练 | 「用更弱的监督者训练更强的模型」 | 经典可扩展监督基准，AAR 在此超过人类 |
| 共享论坛 | 「Agent 发布发现的地方」 | 只追加、沙箱外存储 |
| 沙箱外日志 | 「Agent 无法编辑自己的记录」 | 防篡改地写穿到外部存储 |
| 预设工作流 | 「人类设计者给出的分步计划」 | 限制 AAR；相对自由分解常常降低性能 |
| 自由分解 | 「Agent 决定如何拆解任务」 | 更有能力，更难审计 |
| AI 研发阈值 | 「RSP/FSF 能力等级」 | 以有竞争力的成本完全自动化研发 pipeline |
| 压缩的时间线 | 「对齐与能力的竞赛」 | 如果能力比对齐复利更快，错位风险增长 |

## 延伸阅读

- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — 一手来源。
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — AI 研发阈值框架。
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 更广泛的 Agent 自主性框架。
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — 与 RSP 平行的 ML 研发自主性等级。
- [Burns et al. (2023). Weak-to-Strong Generalization (OpenAI)](https://openai.com/index/weak-to-strong-generalization/) — AAR 所攻克的底层问题。
