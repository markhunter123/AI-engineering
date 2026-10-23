# Agent Skills：可移植契约与运行时边界

> 技能不是改了个好文件名的长 prompt。它是一个可发现的包，包含指令、资源和可执行辅助脚本，通过运行时契约进入 Agent 的上下文。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 01 (The Tool Interface), Phase 13 · 05 (Tool Schema Design)
**Time:** ~90 minutes

## 学习目标

- 定义 Agent Skill，并能把它与 prompt、仓库指令、工具、钩子、子 Agent、插件区分开。
- 阅读可移植的 `SKILL.md` 契约，并将它与特定运行时的扩展区分开。
- 将发现、选择、激活、资源加载、工具使用和验证解释为彼此独立的生命周期阶段。
- 在运行时把技能放入 Agent 目录之前，先校验技能包。
- 针对具体任务，在技能、MCP 工具、钩子、子 Agent 或普通代码之间做出选择。

## 十分钟初次成功

在做长篇解释之前先完成这一步。你会创建一个小技能，将完整的审查器包安装到真实的 Agent 宿主中，调用它，验证结果，再把它移除。这用一个可观察的结果验证整个生命周期。

### 真实宿主实验的预检

真实宿主 checkpoint 需要 Node.js、`npx`、Python 3、一个选定的支持技能的宿主，以及在安装器中选择的项目或用户作用域的写权限。先验证本地命令：

```bash
node --version
npx --version
python3 --version
```

在安装之前先确定你要使用的宿主和作用域。如果任一要求无法满足，可以在网站上阅读本课，或继续完成下面的手动包练习。那条回退路径能教授契约，但无法证明宿主发现、调用、打包脚本执行或卸载行为。请把这些观察项标记为待定。

### 1. 从空工作目录开始

在你保存学习工作的任意父目录中运行这些命令：

```bash
mkdir -p agent-skills-first-run
cd agent-skills-first-run
TARGET_ROOT="$(pwd -P)"
printf 'TARGET_ROOT=%s\n' "$TARGET_ROOT"
ls -A
```

最后一条命令应不输出任何内容。如果它打印出文件，请换一个空目录，以便审查有清晰的边界。

为你的第一个技能创建目录：

```bash
mkdir -p my-first-skill
```

创建 `my-first-skill/SKILL.md`，内容如下：

```markdown
---
name: my-first-skill
description: Turn rough meeting notes into a compact decision record when the user asks to capture a technical decision.
---

# Decision record

Extract the decision, context, alternatives, owner, and next review date.
If the notes do not contain a decision, ask one clarifying question instead
of inventing one.
```

确认你在预期的目录中创建了该文件：

```bash
test -f my-first-skill/SKILL.md
```

没有输出且退出码为 0，表示文件存在。

### 2. 安装完整的审查器包

留在 `agent-skills-first-run` 中并运行：

```bash
npx skills add rohitg00/ai-engineering-from-scratch --skill skill-contract-reviewer --full-depth
```

选择你正在使用的 Agent 宿主和作用域。安装器应列出 `skill-contract-reviewer` 以及它写入的目标位置。必须使用 `--full-depth`，因为本课的技能是一个嵌套包，包含引用、脚本和资源。

把 `SKILL_ROOT` 设置为安装器报告的绝对目录。它必须是包含已安装 `SKILL.md` 的目录，而不是课程源目录，也不是当前工作区：

```bash
# Replace the placeholder with the destination printed by the installer.
SKILL_ROOT="$(cd "/absolute/path/to/skill-contract-reviewer" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\n' "$SKILL_ROOT"
```

如果 Agent 会话此前已经打开，请启动新会话，或使用该宿主提供的技能重新扫描命令。不要假设每个宿主都会热重载其目录。

### 3. 显式调用它

在已安装的 Agent 中，以 `agent-skills-first-run` 为工作目录，使用该宿主支持的语法：

| 宿主 | 显式调用 |
|---|---|
| Codex | `skill-contract-reviewer`，或从 `/skills` 中选择它，然后提供审查请求 |
| Claude Code | `/skill-contract-reviewer`，随后跟上审查请求 |
| 可移植回退方式 | `Use skill-contract-reviewer to review the target package.` |

在请求中使用打印出的 `SKILL_ROOT` 和 `TARGET_ROOT` 绝对路径值。要求宿主在执行前展开这些值，并显示解析后的确切命令，而不是依赖进程工作目录的命令：

```text
Use skill-contract-reviewer to review <TARGET_ROOT>/my-first-skill. The installed bundle root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/check_skill.py <TARGET_ROOT>/my-first-skill. Before running it, show the fully resolved argv. Return the validation report, selected primitives, and one sentence for each selection. Include the resolved script path, resolved target path, cwd, argv, and exit code as execution evidence.
```

解析后的命令应具有这种形态，且不残留任何占位符：

```bash
python3 "/absolute/install/path/skill-contract-reviewer/scripts/check_skill.py" \
  "/absolute/workspace/path/agent-skills-first-run/my-first-skill"
```

成功的结果同时具备全部三个属性：

1. 宿主按名称找到 `skill-contract-reviewer`。
2. 审查器读取包契约并运行它自带的校验器。
3. 响应包含针对示例的、没有结构错误的校验报告，以及有理由的原语选择。

执行证据还必须指明脚本路径、目标路径、cwd、确切的参数向量和退出码。一份没有这些字段的流畅报告，并不能证明已安装的配套脚本确实运行过。

如果宿主报告技能不可用，请核实安装目标位置，重新扫描或重启一次，然后重试显式请求。不要为了掩盖安装失败而改写技能描述。

### 4. 探测隐式选择

开始一个新的 Agent 回合，在不点名技能的情况下输入同一任务：

```text
Review <TARGET_ROOT>/my-first-skill as a reusable agent package and tell me whether its package contract is valid.
```

如果宿主会暴露已选择的技能，请记录它是否选择了 `skill-contract-reviewer`。如果宿主不暴露路由信息，请将隐式选择标记为未验证。显式调用是可移植的回退方式。

### 5. 清理

只移除已安装的审查器包：

```bash
npx skills remove skill-contract-reviewer
```

选择与安装时相同的宿主和作用域。在重新扫描或开启新会话之后，显式请求 `skill-contract-reviewer` 应报告它不可用。保留 `my-first-skill` 供后续课程使用，或者在完成本路线后删除实验目录。

## 问题

假设你的团队有一套可靠的发布流程。它会查找已合并的变更、检查迁移说明、更新 changelog、运行打包命令，并产出一份审查清单。

把这条工作流塞进一个 prompt 里，粘贴起来很容易，但难以操作。这个 prompt 没有稳定身份、没有发现规则、没有资源边界、没有可测试的包形态，也无法回答一些基本问题：谁可以调用它？模型何时应选择它？它可以运行哪些脚本？哪些文件是可信的？上下文被压缩后，什么还能保留？

相反的错误是把每一条可复用指令都当成技能。仓库约定、确定性自动化、外部工具、事件钩子和委派 Agent 解决的是不同问题。把它们全部打包进 `SKILL.md`，会得到一个看起来可移植、却依赖某个宿主未记录行为的目录。

第一项工程任务是分类。先确定产物是什么，再决定如何打包。

## 概念

### 技能编码的是程序性知识

Agent Skill 是一个目录，入口文件是 `SKILL.md`。入口文件包含 YAML frontmatter，后面是 Markdown 指令。该目录还可以包含引用、脚本和资源。

```figure
skill-package-anatomy
```

可部署单元是整个目录，而不只是 Markdown 文件。复制了一份 `SKILL.md` 却缺少引用文件，即使 frontmatter 能解析，也是一个损坏的包。

### 相邻的抽象

| 产物 | 主要职责 | 加载或运行时机 | 不应冒充什么 |
|---|---|---|---|
| prompt | 塑造一次模型交互 | 由应用或用户引入 | 带资源的版本化包 |
| 仓库指令 | 说明某个代码库的常驻规则 | 编码运行时进入该作用域时 | 可复用的任务工作流 |
| Agent Skill | 提供可复用的程序性知识 | 显式或隐式激活时 | 硬性授权边界 |
| MCP 工具 | 暴露带类型的远程能力 | 模型或应用调用它时 | 详细的操作流程 |
| 钩子（hook） | 在事件发生时运行确定性逻辑 | 声明的事件发生时 | 概率性的模型路由 |
| 子 Agent | 以独立的上下文和状态委派工作 | 编排器创建或调用它时 | 静态指令包 |
| 插件 | 分发更大范围的运行时扩展 | 宿主安装或启用它时 | 可移植技能契约本身 |
| 习得技能库 | 存储通过经验发现的行为 | 策略检索先前的程序或轨迹时 | 基于标准的 `SKILL.md` 包 |

一个发布技能可以告诉 Agent 如何检查一次发布。一个 MCP server 可以暴露发布注册表。一个钩子可以禁止直接推送。一个子 Agent 可以独立审计候选版本。这些部件能够组合，是因为各自职责不同。

### 「skill」一词指代两种不同概念

研究系统有时会把学习到的程序、成功轨迹或特定环境的策略片段称为 skill。Agent 可以在探索过程中创建这些产物，按任务相似度检索它们，执行它们，并根据反馈修订整个库。Phase 14 · 10 构建了这种终身学习库。

本小专题中的 Agent Skill 则不同。它是一个编写出来的包，具有声明的文件系统契约、目录元数据、渐进式披露、由运行时中介的调用，以及宿主控制的工具。它可以由 Agent 生成或改进，但这种格式并不要求学习过程。

| 维度 | Agent Skill 包 | 习得技能库 |
|---|---|---|
| 主要单元 | `SKILL.md` 目录 | 程序、策略、轨迹或记忆记录 |
| 创建方式 | 编写、生成或精选 | 通常从环境经验中发现 |
| 选择方式 | 目录描述加运行时策略 | 基于任务状态的检索或策略 |
| 执行方式 | 模型遵循指令并调用宿主工具 | 环境运行存储的行为或代码产物 |
| 可移植性 | 包契约可跨越兼容宿主 | 通常绑定单一环境和动作空间 |
| 评估方式 | 路由、产物、安全性和宿主兼容性 | 奖励、成功率、迁移和库增长 |

两种概念都在打包可复用的能力。它们不应仅因同名就共享实现层面的主张。

### 可移植核心

Agent Skills 规范要求两个 frontmatter 字段：

```yaml
---
name: release-readiness
description: Inspect a release candidate when the user asks whether a version is ready to publish.
---
```

`name` 是稳定标识符。它必须满足规范的命名规则，并与父目录一致。`description` 既是文档，也是路由元数据。它应说明技能做什么以及何时适用。

可移植的可选字段有：

| 字段 | 用途 | 可移植性说明 |
|---|---|---|
| `license` | 声明包的条款 | 核心规范 |
| `compatibility` | 声明环境要求 | 核心规范 |
| `metadata` | 承载字符串值的扩展数据 | 核心规范 |
| `allowed-tools` | 建议预先批准的工具 | 实验性；宿主支持程度不一 |

Markdown 正文承载操作指令。它应定义工作流、决策点、失败行为，以及指向支撑资源的直接路径。

```markdown
# Release readiness

Use this workflow for a release candidate, not for ordinary development builds.

1. Read `references/release-policy.md`.
2. Run `python3 scripts/inspect_release.py --format json`.
3. Stop if the report contains a blocking failure.
4. Produce the checklist from `assets/release-checklist.md`.
5. Ask for approval before any publish or tag action.
```

### 运行时扩展是第二层

有些宿主接受额外的 frontmatter 或配套配置。这些字段可能有用，但不会自动具备可移植性。

| 行为 | 宿主扩展示例 | 可移植核心？ |
|---|---|---|
| 在保留用户直接调用的同时，对模型路由隐藏技能 | `disable-model-invocation` | 否 |
| 在允许模型路由的同时，对用户命令菜单隐藏技能 | `user-invocable` | 否 |
| 在命令菜单中显示参数帮助 | `argument-hint` | 否 |
| 在委派上下文中运行技能 | `context`、`agent` | 否 |
| 固定模型或推理设置 | `model`、`effort` | 否 |
| 注册生命周期自动化 | `hooks` | 否 |
| 在 Codex 中禁用隐式调用 | `agents/openai.yaml` 策略 | 否 |

把每个扩展都当作适配器。在没有它时保持核心工作流仍然有效，记录回退方式，并对消费它的宿主进行测试。运行时可能忽略未知字段、拒绝它，或保留它但不实现相应行为。

### Frontmatter 是可执行元数据

元数据会在技能正文被读取之前改变系统行为。

- 格式错误的 `name` 会导致发现失败。
- 含糊的 `description` 会把错误的请求路由过来。
- 一个仅限人类的标志可能把技能从模型目录中移除。
- 工具许可会影响宿主是否请求权限。
- 上下文设置可能把执行移入独立的 Agent 会话。

像审查配置代码一样审查 frontmatter。校验它、为它做版本管理，并把它的行为纳入评估。

### 技能生命周期

```figure
skill-runtime-lifecycle
```

每个箭头都是一个边界，各自带有失败模式。

1. **发现（Discovery）** 在配置的位置查找可能的包。
2. **校验（Validation）** 在发布到目录之前拒绝格式错误或不安全的包。
3. **编目（Cataloging）** 只暴露紧凑的 `name` 和 `description`，而不是整个包。
4. **选择（Selection）** 判断该技能是否相关。
5. **激活（Activation）** 把正文加载进模型可见的上下文。
6. **披露（Disclosure）** 只在某个分支需要时才读取引用或资源。
7. **执行（Execution）** 在宿主的权限与隔离规则下使用宿主工具。
8. **验证（Verification）** 独立于模型的自述，检查产出的产物。

把这些阶段混为一谈会形成错误的心智模型。被发现不等于已激活。已激活的技能也无权执行它描述的所有事情。一次被允许的 tool call 并不能证明结果正确。

### 技能与工具相互正交

MCP 回答的是：「这个应用可以调用哪些能力，它们的 schema 是什么？」技能回答的是：「Agent 应如何着手这一类任务？」

```figure
skill-tool-orthogonality
```

技能可以点名某个工具，但真正的能力注册表归宿主所有。如果工具不存在，技能应说明回退方式或明确失败。它绝不应暗示点出一个能力就等于创建了该能力。

### 技能与仓库指令属于不同作用域

仓库指令描述你当前所处的环境：命令、约定、生成文件以及边界。技能则为可能跨多个仓库发生的任务提供可复用的流程。

当两者同时适用时，当前用户请求和仓库规则会约束技能。一个通用的重构技能不得覆盖「禁止编辑生成文件」的仓库规则。

### 技能之间不互相导入

一个技能可以指示 Agent 调用另一个技能，但这不是语言层面的 import。第二个技能仍然要经过运行时发现、资格判断、激活、权限和上下文处理。

把跨技能依赖写成可观察的工作流边：

```markdown
After producing the candidate changelog, invoke the `release-risk-review` skill.
Pass the candidate path and require a blocking or non-blocking verdict.
If that skill is unavailable, stop and report the missing dependency.
```

这让依赖变得可测试，也给宿主一个执行策略的机会。

## 构建它

`code/main.py` 实现了一个偏向标准的小型校验器和产物选择器。它仅使用标准库，以便每条规则都清晰可见。

校验器暴露：

- `parse_frontmatter(text)`，把元数据与正文分开。
- `validate_skill_text(text, directory_name, allowed_runtime_extensions=())`，检查必填字段、命名、未知扩展、正文存在性和可移植限制。
- `ValidationIssue` 和 `SkillReport`，返回结构化证据，而不是一个不透明的布尔值。
- `FrontmatterSyntaxError`，用于无法安全解释的输入。

选择器暴露 `TaskShape` 和 `select_primitives(task)`。它把任务需求映射到普通代码、仓库指令、技能、钩子、子 Agent 或 MCP 工具。

运行实验：

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/22-skills-and-agent-sdks
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

这个命令块需要本地克隆，并且必须从该克隆内的任意位置启动，以便 `git rev-parse --show-toplevel` 能解析仓库根目录。

演示会为以下对象打印 JSON：一个有效的可移植技能、一个带宿主扩展的技能、一个无效包，以及若干任务形态决策。检查问题代码。包校验器应解释如何修复产物，而不是替作者猜测。

### 校验顺序很重要

先校验成本低的结构性事实，再处理更深层的内容规则：

```figure
skill-validation-order
```

这个顺序可以防止次要错误掩盖第一个被破坏的不变量。

## 使用它

在编写技能之前，先填写这张决策卡：

| 问题 | 如果是 | 可能的原语 |
|---|---|---|
| 这是否需要跨多个步骤复用模型判断？ | 流程稳定但决策多变 | 技能 |
| 是否必须在每次事件触发时发生？ | 漏掉一次执行不可接受 | 钩子或应用代码 |
| 模型是否需要带类型输入的外部能力？ | 操作发生在模型上下文之外 | 工具或 MCP server |
| 工作是否需要隔离的上下文、状态或归属？ | 独立工作者返回有界结果 | 子 Agent |
| 这条指导是否只针对某一个仓库？ | 它描述本地命令和约束 | 仓库指令 |
| 一次交互就够了吗？ | 不需要包生命周期 | prompt |

许多生产工作流会用到不止一行。这张卡防止某一种产物假装提供所有属性。

## 交付它

本课在 `outputs/` 下产出 `skill-contract-reviewer` 包。它包含：

- 一个可移植的 `SKILL.md`，用于审查拟议的技能包；
- 针对可移植契约和原语选择的引用清单；
- 一个确定性校验脚本；
- 覆盖 prompt、技能、工具、钩子、普通代码和子 Agent 的任务形态夹具。

安装完整的包，而不只是它的入口文件：

```bash
cd "$(git rev-parse --show-toplevel)"
python3 scripts/install_skills.py /tmp/aiefs-skills --phase 13 --type skill
```

课程安装器会报告每个复制过去的 Phase 13 技能，并写入 `/tmp/aiefs-skills/manifest.json`。这个干净的目标目录用于检查包形态；上面的初次成功流程则在真实宿主中检查发现与调用。

后续课程会深入每个生命周期阶段。第 24 课构建发现与渐进式披露。第 25 课构建调用策略与路由。第 26 课把权限与沙箱分离。第 27 课把整个包变为经过评估的发布产物。

## 练习

1. 使用 `TaskShape` 对你团队中的五个工作流进行分类。为你选择多个原语的每个案例给出理由。
2. 添加边界测试，证明 500 个字符的 `compatibility` 值通过，而 501 个字符的值作为规范错误失败。
3. 向允许列表添加一个运行时扩展。编写测试，证明同一文件仍然能与仅可移植的技能区分开。
4. 把一个 400 行的 prompt 拆成 `SKILL.md`、一个引用、一个脚本契约和一个输出模板。让每个文件只负责一类信息。
5. 为一个引用了不可用 MCP 工具的技能设计失败响应。不要静默替换为权限更宽的工具。
6. 审查一个现有技能，把每句话标注为路由、流程、策略、引用指针或输出契约。把不属于其中的内容移走。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| Agent Skill | 「保存下来的 prompt」 | 一个可发现的目录，包含程序性指令和可选资源 |
| 可移植核心 | 「每个运行时共享的字段」 | 由 Agent Skills 规范定义的契约 |
| 运行时扩展 | 「额外的 frontmatter」 | 宿主特定配置，其行为需要兼容的适配器 |
| 激活 | 「技能运行了」 | 技能正文进入模型可见上下文；执行可能稍后发生 |
| 技能依赖 | 「导入另一个技能」 | 一条由运行时中介的调用边，带有可用性和策略检查 |
| 工具契约 | 「一个函数 schema」 | 某项能力的输入、输出、权限、副作用、错误和证据 |

## 延伸阅读

- [Agent Skills 规范](https://agentskills.io/specification)，了解可移植目录和 frontmatter 契约。
- [Agent Skills 最佳实践](https://agentskills.io/skill-creation/best-practices)，了解范围、指令和资源组织。
- [OpenAI：构建技能](https://learn.chatgpt.com/docs/build-skills)，了解当前 Codex 的发现与调用行为。
- [Claude Code 技能](https://code.claude.com/docs/en/skills)，了解某个运行时的调用、参数、工具和委派上下文扩展。
