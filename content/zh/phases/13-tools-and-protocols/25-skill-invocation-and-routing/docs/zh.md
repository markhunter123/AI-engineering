# 技能调用与路由

> 调用先是一个权限决策，然后才是相关性决策。好的描述帮助模型做出选择；好的策略决定这个选择是否被允许。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 24 (Skill Discovery and Progressive Disclosure)
**Time:** ~105 minutes

## 学习目标

- 区分显式用户调用、隐式模型调用、应用调用和技能间调用。
- 将人类可见性与模型资格建模为彼此独立的策略维度。
- 编写带有正触发条件和近似反例（near-miss）边界的路由描述。
- 在 trace 和测试中分开资格、选择、激活、参数绑定和执行。
- 适配特定运行时的调用字段，而不把它们呈现为可移植 frontmatter。

## 问题

你安装了一个 `database-migration` 技能。用户可以按名称运行它，但模型也能看到它的描述，并在有人提出一般数据库问题时选择它。于是技能为一个只需要解释的任务提出了 schema 变更。

你添加 `user-invocable: false`，期望阻止人们手动运行它。在另一个运行时中，该字段被忽略。你添加 `disable-model-invocation: true`，期望技能完全消失。在理解该字段的运行时中，用户仍然可以显式调用它。

字段名本身没有问题。错的是心智模型。「用户可以看到它」「模型可以选择它」「应用可以预加载它」「其中的工具可以执行」是彼此独立的事实。一个名为 `invocable` 的布尔值无法表达这些。

路由还有第二种失败模式。如果描述含糊，就会有好几个技能看起来都合理。如果描述塞满关键词，无关任务也会触发它们。目录是一个概率性接口：足够紧凑以放得下，足够具体以完成路由。

## 概念

### 五个通道可以启动生命周期

| 角色 | 调用形态 | 典型用途 | 主要风险 |
|---|---|---|---|
| 人类用户 | 在 UI 或 prompt 中指名技能 | 有意的流程选择 | 用户期望宿主并不授予的可用性或权限 |
| 模型或自主智能体 | 根据任务上下文选择目录条目 | 自动的专家流程 | 误报路由 |
| 应用 | 通过运行时代码激活或预加载技能 | 固定的产品工作流 | 与单一宿主的隐藏耦合 |
| 另一个技能或子智能体 | 作为工作流依赖请求某个确切技能 | 组合 | 循环、依赖缺失或上下文串扰 |
| 评估框架 | 在固定场景下激活某个确切技能 | 可重复测量 | 测试技能时意外绕过正在研究的生产策略 |

可移植的 Agent Skills 规范定义的是包。它并不标准化通用的斜杠命令 UI、隐式路由标志、应用 API 或子智能体生命周期。

### 五个调用阶段

```figure
skill-invocation-stages
```

精确使用这些词：

- **合格（Eligible）** 表示策略允许该角色请求该技能。
- **已选择（Selected）** 表示用户指名了它，或路由器判定它相关。
- **已激活（Activated）** 表示它的指令进入了工作上下文。
- **执行中（Executing）** 表示智能体已在这些指令下开始模型或工具工作。
- **已完成（Completed）** 表示输出通过了一项独立的成功检查。

只记录 `skill_used=true` 的 trace 会隐藏失败发生的边界。

### 人类与模型调用构成 2x2 矩阵

| 人类可调用 | 模型可调用 | 模式 | 适用示例 |
|:---:|:---:|---|---|
| 是 | 是 | 共享 | 代码讲解、测试规划、文档审查 |
| 是 | 否 | 仅人类 | 发布准备、账单导出、破坏性清理计划 |
| 否 | 是 | 仅模型 | 内部风格指南、领域参考、自动支持流程 |
| 否 | 否 | 禁用或仅应用 | 分阶段上线、已弃用包、程序化预加载 |

这个矩阵是策略模型，不是标准 YAML。

当前有一个宿主用 `disable-model-invocation: true` 表示仅人类行，用 `user-invocable: false` 表示仅模型行。默认两者皆可。另一个宿主使用带 `allow_implicit_invocation: false` 的 `agents/openai.yaml`，在保留显式调用的同时禁用隐式选择。这些都是运行时适配器。未知宿主可能忽略它们。

令人困惑的细节很重要：`user-invocable: false` 并不意味着「模型不能用这个技能」。它在定义该字段的宿主中移除用户的直接调用。`disable-model-invocation: true` 并不意味着「技能被禁用」。它移除模型发起的选择，同时保留用户的显式访问。

### 显式调用以身份为先

显式调用直接提供身份：

```text
/release-readiness v2.4.0
```

或者：

```text
release-readiness check v2.4.0 without publishing
```

当前 Codex 接口文档说明了用 `/skills` 进行选择，以及在请求中使用纯技能名进行显式调用。Claude Code 文档说明了 `/skill-name` 和宿主特定的参数展开。确切的语法、菜单可见性、引用规则和变量展开属于宿主。

显式请求仍然要经过策略。指名技能不应绕过缺失的权限、工作区约束、审批门禁或运行时隔离。

### 隐式调用以描述为先

对隐式路由而言，模型最初看到的是目录元数据，而不是完整正文。因此描述就是技能的路由接口。

弱描述：

```yaml
description: Helps with releases.
```

过宽：

```yaml
description: Use for release, version, package, build, deploy, publish, tag, changelog, GitHub, CI, or software tasks.
```

有边界：

```yaml
description: Inspect an already prepared release candidate and produce a readiness report. Use when the user asks whether a version, tag, package, or image is ready to publish; do not use for ordinary build failures or feature development.
```

有边界的版本包含：

1. **能力：** 检查已准备好的候选。
2. **输出：** 就绪报告。
3. **正边界：** 询问发布产物是否已就绪。
4. **负边界：** 普通构建和功能开发不在范围内。

当两个相近技能共享词汇时，负边界很有用。它们不能替代近似反例评估。

### 路由是带弃权选项的分类

对于技能 `s` 和请求 `x`，设想一个路由器评分：

```text
score(s, x) = capability_match + trigger_match + context_match - exclusion_match - ambiguity_penalty
```

实际评分可能由 LLM 决定，而不是算术计算。工程原则仍然成立：选择必须超过阈值，并胜过竞争技能。证据不足时，弃权。

```figure
skill-routing-abstention
```

对于高影响技能，即使描述很强，隐式路由也可能不合适。当误报的代价超过自动选择带来的便利时，使用仅人类策略。

### 资格过滤必须先于排序

不要给每个已发现技能打分，选出最强匹配，然后再检查那一个技能的策略。被阻止的最高匹配会错误地阻止较合格的低分候选被考虑。

对隐式路由使用这个顺序：

1. 按请求角色和当前宿主适配器过滤已发现的技能。
2. 只对合格候选打分。
3. 如果最强合格匹配超过阈值和歧义规则，就选择它。
4. 当没有候选合格，或没有合格分数足够强时，弃权。

假设 `incident-triage` 得分为 `0.80`，但其宿主扩展禁用了模型调用。`incident-review` 得分为 `0.55`，并允许模型调用。路由器应把 `incident-review` 评估为最佳合格候选。它不应选择 `incident-triage`，被拒绝后就停止。

这个顺序还能防止策略变化改变相关性评分的含义。资格定义选择集。相关性对该集合排序。

### 路由评估需要近似反例

正例证明召回率：

```json
{"prompt":"Is version 2.4.0 ready to publish?","expected":"release-readiness"}
```

明确的负例证明基本精确率：

```json
{"prompt":"Explain rotary position embeddings.","expected":null}
```

近似反例暴露边界质量：

```json
{"prompt":"Why did today's package build fail?","expected":"build-diagnostics"}
```

这个近似反例与发布技能共享 `package` 和 `build` 词汇，但属于另一个流程。只由明显正例和无关负例组成的路由集，会高估质量。

### 参数有三种表示形式

调用参数会跨越多个边界：

```figure
skill-argument-boundaries
```

在每个边界上，保留意图，但不要把文本当作代码。

- 宿主解析器决定命令语法和引用方式。
- 技能根据宿主规则接收绑定后的文本或变量。
- 指令校验必填值和默认值。
- 工具调用把值转换为带类型的 schema，并再次校验。

不要把原始参数插值进 shell 命令。优先使用以参数向量调用的脚本，或带类型的 MCP 工具。

### 应用调用是显式编排

产品可以因为其工作流已知任务类型而激活技能。例如，拉取请求审查服务可以在用户按下 Review 后预加载 `pull-request-risk-review`。

这消除了路由不确定性，但引入了对运行时 API 的依赖。把该适配器放在可移植正文之外：

```figure
skill-host-adapter
```

当被另一个合规客户端打开时，技能应仍然可理解。

### 技能间调用是类工具的边

假设 `release-readiness` 在依赖文件变化时请求 `security-change-review`。

调用方应提供：

- 目标技能身份；
- 有界的任务和产物路径；
- 预期的响应契约；
- 调用原因；
- 不可用时的回退；
- 最大深度或循环规则。

```json
{
  "target_skill": "security-change-review",
  "task": "Review dependency changes in the candidate diff",
  "inputs": ["artifacts/release.diff"],
  "expected": "risk-report.json",
  "max_depth": 2
}
```

第二个技能不会被盲目粘贴进第一个技能。宿主决定如何激活它，以及它是共享上下文、在 fork 中运行，还是通过工具结果返回。

### 上下文生命周期因宿主而异

激活之后，技能正文可能留在对话中、在压缩时被摘要，或运行在委派上下文中。工具许可可能只持续一个回合，而指令持续更久。子智能体可能收到技能，但没有父级的完整历史。

不要编写依赖不可见生命周期假设的技能。把持久输出放入文件或带类型状态，使重新进入安全，并说明中断后必须重新加载什么。

```markdown
On resume, read `artifacts/release-readiness.json` if it exists.
Revalidate the candidate commit before continuing.
Do not repeat an external write whose idempotency key is already recorded.
```

## 构建它

`code/main.py` 将策略和路由实现为独立的适配器。

模型包括：

- `Actor`，表示人类、模型、自主智能体、应用、技能和评估框架调用方；
- `SkillMetadata`，表示路由身份；
- `InvocationPolicy`，表示人类/模型矩阵；
- `InvocationRequest` 和 `InvocationDecision`，表示可追踪的输入与结果；
- `CorePolicyAdapter`，表示不带宿主扩展的可移植行为；
- `ExtensionPolicyAdapter`，表示已识别的运行时字段；
- `build_invocation_matrix(policy)`，用于生成 2x2 视图；
- `route_request(skills, request, adapter)`，在相关性排序、选择和拒绝之前进行资格过滤。

运行它：

```bash
cd phases/13-tools-and-protocols/25-skill-invocation-and-routing
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

演示会打印一个矩阵，以及针对显式人类、隐式模型、自主智能体、应用、技能组合和评估框架通道的决策。其扩展适配器结果展示了被阻止的最高词法匹配如何在合格替代项参与排序之前被移除。它还包含精确名称允许列表。不需要模型 API。确定性路由器存在的目的是让策略边界可检查，而不是声称词法匹配能复现生产环境的模型路由。

### 为什么核心适配器与扩展适配器分开

如果某个解析器为观察到的每个 frontmatter 字段赋予含义，它就会悄悄把运行时约定提升为伪标准。分离的适配器迫使调用方指明当前生效的是哪种宿主语义。

`CorePolicyAdapter` 只使用应用提供的策略。`ExtensionPolicyAdapter` 识别一组显式的宿主字段，并记录是哪个字段改变了决策。

## 使用它

在发布技能之前编写调用契约：

```yaml
actors:
  human: allow
  model: deny
  application: allow
  skill: deny
explicit_name: release-readiness
arguments:
  candidate: required
  publish: fixed_false
ambiguity: ask_user
missing_dependency: stop
context:
  durable_state: artifacts/release-readiness.json
  max_composition_depth: 2
```

这个契约是供适配器和测试使用的设计文档。除非某个标准明确采纳，否则它不是可移植的 `SKILL.md` frontmatter。

## 交付它

本课产出 `skill-invocation-router` 包。它包含调用模型参考、示例宿主策略，以及一个非执行式 CLI：评估一次人类、模型、自主智能体、应用、技能组合或评估框架请求，并返回包含通道、适配器、分数和原因的 JSON 决策。

单请求 CLI 是策略探针，而不是完整的触发器评估。使用第 27 课中带标签的正例与近似反例设计，计算混淆计数、精确率、召回率和重复运行稳定性。

## 练习

1. 创建人类/模型矩阵的全部四行，并为每一行写出一个合理用例。
2. 为 `CorePolicyAdapter` 添加仅应用激活。证明人类和模型调用方仍然被拒绝。
3. 为一个部署技能编写十个近似反例。每个 prompt 必须与技能共享词汇，却属于不同的工作流。
4. 在排名前两位的路由分数之间添加歧义余量。当余量太小时返回 `ask`。
5. 为技能间请求添加最大组合深度，并检测两个技能之间的循环。
6. 用同一组带标签样本分别通过核心适配器和扩展适配器运行。解释每一个发生变化的决策。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 显式调用 | 「斜杠命令」 | 角色直接提供技能身份，并受策略约束 |
| 隐式调用 | 「模型自己选」 | 路由器根据任务上下文从合格目录元数据中选择 |
| 用户可调用 | 「人类可以用它」 | 宿主特定的菜单或直接调用属性，不是核心字段 |
| 模型可调用 | 「智能体可以用它」 | 在宿主策略下具备隐式模型选择的资格 |
| 调用适配器 | 「frontmatter 解析器」 | 把宿主字段和 API 映射到声明式策略模型的代码 |
| 近似反例 | 「困难负例」 | 与技能预期输入相似但不应触发技能的请求 |
| 弃权 | 「没有选择技能」 | 证据缺失或存在歧义时有意给出的路由结果 |

## 延伸阅读

- [优化技能描述](https://agentskills.io/skill-creation/optimizing-descriptions)，了解正触发条件、具体性和评估。
- [评估技能](https://agentskills.io/skill-creation/evaluating-skills)，了解触发器和输出评估设计。
- [OpenAI：构建技能](https://learn.chatgpt.com/docs/build-skills)，了解当前 Codex 的显式与隐式调用控制。
- [Claude Code 技能](https://code.claude.com/docs/en/skills)，了解某个宿主的 `user-invocable`、`disable-model-invocation`、参数和委派上下文。
