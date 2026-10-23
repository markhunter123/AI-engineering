# 将 Agent 指令写成可执行约束

> 写成散文的指令只是愿望，写成约束的指令才是测试。工作台把每条规则变成 Agent 运行时可以检查、评审者事后可以验证的东西。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 32 (Minimal Workbench)
**Time:** ~50 minutes

## 学习目标

- 把路由说明与操作规则分开。
- 将启动规则、禁止操作、完成定义、不确定处理和审批边界表达为机器可检查的约束。
- 实现一个规则检查器，按规则集给一次运行打分。
- 让规则集对 diff 友好，使评审能看清改了什么。

## 问题

典型的 `AGENTS.md` 读起来像入职文档。它告诉 Agent“要小心”“充分测试”“不确定就问”。三天后，Agent 交付了一个没有测试的变更，写入了禁止目录，而且从不发问，因为它从来不知道边界在哪里。

指令在可操作时很有力，在空泛时很无力。修复方式是写出工作台可以解释、评审者可以打分的规则。

## 概念

规则应当放在 `docs/agent-rules.md`，远离简短的根路由器。每条规则有一个名称、一个类别和一个检查。

```mermaid
flowchart LR
  Router[AGENTS.md] --> Rules[docs/agent-rules.md]
  Rules --> Checker[rule_checker.py]
  Checker --> Report[rule_report.json]
  Report --> Reviewer[Reviewer]
```

### 覆盖大多数规则的五类

| 类别 | 规则回答的问题 | 示例 |
|----------|---------------------------|---------|
| 启动 | 工作开始前必须满足什么？ | “状态文件存在且是最新的” |
| 禁止 | 什么绝不能发生？ | “不要编辑 `scripts/release.sh`” |
| 完成定义 | 什么能证明任务完成？ | “pytest 退出码为 0，且验收行通过” |
| 不确定 | 不确定时 Agent 该怎么做？ | “提交一条问题记录，而不是靠猜” |
| 审批 | 什么需要人类批准？ | “任何新依赖、任何生产写入” |

一条不属于这五类的规则，通常应该拆成两条规则。强制拆开。

### 规则是机器可读的

每条规则有一个 slug、一个类别、一行描述，以及一个 `check` 字段，指向 `rule_checker.py` 中的函数。新增规则意味着新增检查；检查器随工作台一起成长。

### 规则对 diff 友好

规则位于单个 markdown 文件中，每条规则一个标题。重命名在 diff 中可见。新规则放在所属类别的顶部。过期规则直接删除，而不是注释掉，因为工作台才是事实源，不是团队上个季度感受的聊天记录。

### 规则与框架防护栏

框架防护栏（OpenAI Agents SDK guardrails、LangGraph interrupts）在运行时层面执行规则。本课的规则集是那些防护栏所实现的、人类可读且可评审的契约。两者都需要：运行时在一轮执行中捕获违规，规则集证明运行时确实在做正确的事。

### 渐进披露：给地图，不给百科全书

`AGENTS.md` 不断变长的原因，是每次事故都新增一条规则，却没有一次事故删除规则。一年之后，文件有两千行，Agent 只读第一屏，注意力预算耗尽，只按被告知内容的一小部分行动。巨型指令文件失败的原因，和四十页入职文档失败的原因一样：读者只浏览一次，再也不回到真正重要的部分。

解决办法不是更短的文件，而是分层的文件。根路由器保持足够短，每次会话都能读完，只放指针。深度内容放在主题文件里，只有当任务涉及该主题时 Agent 才加载。给 Agent 一张地图，而不是整套百科全书，让它自己走到需要的页面。

```
AGENTS.md                  # router, < 50 lines: what this repo is, where to look, the 5 hard rules
docs/
  agent-rules.md           # the full rule set (this lesson)
  architecture.md          # loaded when the task touches module boundaries
  testing.md               # loaded when the task writes or runs tests
  deploy.md                # loaded only for release work, gated behind an approval rule
feature_list.json          # the backlog (Phase 14 · 36)
```

| 层级 | 所在位置 | 何时读取 | 大小预算 |
|------|----------|-----------|-------------|
| 路由器 | `AGENTS.md` | 每次会话，始终读取 | 约 50 行以内 |
| 规则 | `docs/agent-rules.md` | 每次会话，启动时读取 | 每个类别一屏 |
| 主题文档 | `docs/<topic>.md` | 仅当任务涉及该主题时 | 按需深入 |

有两项测试保证分层不流于形式。可达性测试：Agent 从路由器出发，最多跳两次就应到达任何规则，因此路由器必须按路径链接每个主题文档，而不是用散文描述它。新鲜度测试：路由器要短到评审者每次 PR 都会重读，这是唯一能阻止它悄悄长回它所取代的百科全书的东西。一个失效的指针比缺失的规则更糟，因此路由器中的坏链本身就是启动检查违规。

```figure
wb-rule-checkoff
```

## 构建它

`code/main.py` 提供：

- `agent-rules.md` 解析器，把规则加载进 dataclass。
- `rule_checker.py` 风格的检查函数，每个 `check` 引用对应一个。
- 一次演示运行，违反两条规则；一次检查通过，抓住它们。

运行方式：

```
python3 code/main.py
```

输出：解析后的规则集、运行轨迹、每条规则的通过/失败结果，以及保存在脚本旁边的 `rule_report.json`。

## 生产环境中的模式

以下三种模式，决定规则集是能撑过一个季度，还是在一周内腐化。

**写入时打严重级别标签。** 每条规则携带 `severity`：`block`、`warn` 或 `info`。检查器报告三者；运行时只拒绝 `block`。多数团队早期夸大严重度，然后在交付压力下悄悄削弱；在写入时打标签会迫使团队提前校准。配合验证门（第 14 阶段 · 38）：任何对 `block` 规则的覆盖都会签名写入 `overrides.jsonl` 审计日志。

**用规则过期作为强制函数。** 每条规则携带 `expires_at` 日期（默认自编写起 90 天）。当一条尚未过期的规则连续 60 天零违规时，检查器发出警告；下一次季度评审要么论证保留，要么降为 `info`，要么删除。Cloudflare 的生产 AI Code Review 数据（2026 年 4 月，30 天内 5,169 个仓库的 131,246 次评审运行）显示，显式设置过期的规则集在每个仓库保持在 30 条以下；没有过期机制的规则集增长到 80 条以上，其中大多数从未触发。

**Markdown 作源，JSON 作缓存。** `agent-rules.md` 是人工编写的文件；`agent-rules.lock.json` 是检查器在热路径中读取的缓存。锁文件由 pre-commit 钩子重新生成。Markdown 的 diff 可评审；JSON 解析不必进入每一轮。形态与 `package.json` / `package-lock.json`、`Cargo.toml` / `Cargo.lock` 相同。

## 使用它

生产环境：

- Claude Code、Codex、Cursor 在会话开始时读取规则，并在拒绝操作时引用它们。检查器在 CI 中重新运行规则，以捕获悄悄发生的偏移。
- OpenAI Agents SDK guardrails 把同样的检查注册为输入和输出防护栏。Markdown 是文档工作面；SDK 是运行时工作面。
- LangGraph interrupts 在运行中的节点违反规则时触发。中断处理器读取规则，询问人类，然后恢复。

规则集可跨这三者移植，因为它只是 markdown 加函数名。

## 交付它

`outputs/skill-rule-set-builder.md` 会访谈项目负责人，把现有散文式指令归入五个类别，并生成带版本的 `agent-rules.md` 加一个检查器 stub。

## 练习

1. 如果你的产品确实需要，增加第六个类别。论证为什么它不能归并进现有五类之一。
2. 扩展检查器，让规则可以携带严重级别（`block`、`warn`、`info`），报告按级别汇总。
3. 把检查器接入 CI：如果 block 级规则在最近一次 Agent 运行中失败，就让构建失败。
4. 为每条规则增加“过期”字段。连续 90 天没有检查失败后，规则进入评审。
5. 找一份真实的 `AGENTS.md`，改写成五类规则。其中多少行是可操作的？多少行是空泛的？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 可操作规则（Operational rule） | “真正的指令” | 工作台能在运行时检查的规则 |
| 空泛规则（Aspirational rule） | “要小心” | 没有检查的规则；要么删除，要么升级 |
| 完成定义（Definition of done） | “验收” | 任务完成所需的客观、文件为凭的证明 |
| block 严重级别（Block severity） | “硬规则” | 违反即中止运行；没有操作员不能静默忽略 |
| 规则过期（Rule expiry） | “过期规则清理” | 连续 N 天没有失败记录的规则进入退役流程 |

## 延伸阅读

- [OpenAI Agents SDK guardrails](https://openai.github.io/openai-agents-python/guardrails/)
- [LangGraph interrupts](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [Rick Hightower, Agent RuleZ: A Deterministic Policy Engine](https://medium.com/@richardhightower/agent-rulez-a-deterministic-policy-engine-for-ai-coding-agents-9489e0561edf)——生产环境中的 block/warn/info 严重级别
- [Cloudflare, Orchestrating AI Code Review at Scale](https://blog.cloudflare.com/ai-code-review/)——131k 次评审运行，规则组合的经验
- [microservices.io, GenAI development platform — part 1: guardrails](https://microservices.io/post/architecture/2026/03/09/genai-development-platform-part-1-development-guardrails.html)——规则与 CI 之间的纵深防御
- [Type-Checked Compliance: Deterministic Guardrails (arXiv 2604.01483)](https://arxiv.org/pdf/2604.01483)——以 Lean 4 作为“规则即检查”的上界
- [logi-cmd/agent-guardrails](https://github.com/logi-cmd/agent-guardrails)——合并门禁实现：范围、变异测试、违规预算
- 第 14 阶段 · 32——这套规则所接入的最小工作台
- 第 14 阶段 · 38——消费规则报告的验证门
- 第 14 阶段 · 39——给规则合规性打分的评审 Agent
