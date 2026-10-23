# 技能发现与渐进式披露

> 技能在正文加载之前就已经有用。它的名称和描述在目录中赢得一席之地；更深层的文件只有在任务触达时才被允许进入上下文。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 22 (Agent Skills: Portable Contract and Runtime Boundary)
**Time:** ~105 minutes

## 学习目标

- 构建一条文件系统发现 pipeline，将作用域、校验、冲突策略和目录发布分开。
- 解释三个披露层级：目录元数据、激活指令和任务特定资源。
- 设计引用，使 Agent 能直接抵达所需细节，而无需加载整个包。
- 将目录空间预算与激活技能上下文预算分开考虑。
- 在技能读取自身资源时，拒绝路径遍历和符号链接逃逸。

## 问题

你的 Agent 安装了 200 个技能。在会话开始时加载每个 `SKILL.md`、引用文件、脚本和模板，会让当前任务淹没在无关流程中。什么都不加载则会迫使用户记住确切的文件系统路径。

常见的折中方案是目录：向模型展示每个合格技能的紧凑身份和路由描述，只在选择之后才加载完整正文。这会带来两个新的工程问题。

第一，发现不只是递归文件搜索。技能可以存在于项目、用户、管理员、插件或内置作用域。两个包可以同名。一个符号链接可以指向可信根之外。一个格式错误的包可能占用目录空间，或者变得无法调用。

第二，渐进式披露可能变成渐进式困惑。如果 `SKILL.md` 写着「阅读相关指南」，而包里包含十二份指南，模型只能靠猜。如果每份指南又指向另外三个文件，加载就变成无界的图遍历。

好的运行时让发现具有确定性，让披露具有意图。

## 概念

### 发现是一条编译器 pipeline

把文件系统当作源输入。不要把原始路径直接发布给模型。

```figure
skill-discovery-pipeline
```

每个阶段都应产出结构化数据和结构化失败。发现日志应回答：

- 搜索了哪些根目录？
- 找到了哪些候选？
- 哪些候选被拒绝，为什么？
- 哪个包在冲突中胜出？
- 哪些目录条目因预算而被缩短或省略？

没有这些证据，「模型没有使用我的技能」几乎无法诊断。

### 作用域是运行时策略

可移植规范定义的是技能包，而不是一种通用的安装路径或优先级顺序。宿主决定到哪里搜索。

一个通用运行时可能使用这些作用域：

| 作用域 | 示例根目录 | 预期归属 |
|---|---|---|
| 工作区 | `<repo>/.agents/skills/` | 项目维护者 |
| 用户 | `<user-data>/skills/` | 单个开发者 |
| 管理员 | `<system>/skills/` | 机器或组织策略 |
| 插件 | 签名的插件包 | 插件发布者与安装器 |
| 内置 | 运行时包 | 运行时供应商 |

截至 2026 年 8 月，Codex 文档说明项目发现范围从 `$CWD/.agents/skills` 沿祖先目录一直到仓库根目录，外加用户、管理员和内置位置。它支持符号链接的技能目录。重名可能同时出现，而不是被合并。这些是 Codex 的行为，不是 `SKILL.md` 的要求；编写适配器时请核对当前的 [Codex 技能文档](https://learn.chatgpt.com/docs/build-skills)。

绝不要从目录名称臆造优先级。把它声明为策略并加以测试。本课实验为每个 `Scope` 使用显式整数等级，因此同一候选集总能以相同方式解析。

### 冲突需要 `name` 之外的标识

两个都叫 `release-readiness` 的包可以是合法的。一个可能是工作区覆盖，另一个是用户默认值。因此目录条目至少需要：

```json
{
  "name": "release-readiness",
  "description": "Inspect a release candidate for this repository.",
  "scope": "workspace",
  "source": "/repo/.agents/skills/release-readiness",
  "selected": true
}
```

常见的冲突策略包括：

| 策略 | 好处 | 风险 |
|---|---|---|
| 保留每个候选 | 不隐藏任何内容 | 模型看到有歧义的名称 |
| 高优先级作用域胜出 | 调用简单 | 本地包可能遮蔽可信包 |
| 拒绝重复项 | 不会静默遮蔽 | 合法的覆盖无法工作 |
| 按来源限定名称 | 身份明确 | 面向用户的名称变长 |

为宿主选择一种策略。即使被拒绝或被遮蔽的候选不出现在模型目录中，也要在诊断信息里保留它们。

### 三个披露层级

Agent Skills 规范描述了分阶段加载。关键在于每一层有不同用途。

```figure
skill-disclosure-levels
```

#### 第 1 层：目录元数据

模型需要足够的信息来区分该技能与邻近技能。规范估计每个目录条目约 100 token，但实际序列化和 tokenization 由宿主负责。

有用的描述包含两个分句：

```yaml
description: Validate a release candidate and produce a readiness report. Use when the user asks whether a version, tag, or package is ready to publish.
```

第一个分句说明能力。第二个分句说明触发边界。第 25 课用正例和近似反例来评估这一边界。

#### 第 2 层：激活的指令

激活之后，正文应同时充当地图和流程。规范建议将 `SKILL.md` 保持在 500 行以内。这是一个设计信号，不是要填满的目标。

正文应包含：

- 任务边界；
- 默认工作流；
- 分支条件；
- 指向更深层文件的直接引用；
- 工具和脚本契约；
- 失败与停止行为；
- 预期输出及其验证方式。

不要仅仅为了让入口文件变短，就把中心工作流搬进引用文件。激活必须给模型足够的上下文来正确开始。

#### 第 3 层：支撑资源

引用提供文本或数据。脚本提供确定性计算。资源会被复制、填充或转换为交付物，而不是被当作指令。

| 目录 | 模型会读取吗？ | 模型会执行吗？ | 典型内容 |
|---|:---:|:---:|---|
| `references/` | 是，需要时 | 否 | schema、策略、领域指南 |
| `scripts/` | 可能会检查 | 通过获准的工具 | 校验器、转换器、收集器 |
| `assets/` | 仅在有用时 | 否 | 模板、夹具、图像、起始文件 |

这些名称是约定，不是魔法能力。宿主仍然需要文件访问权和执行工具。

### 按分支组织的引用优于主题大杂烩

把入口文件写成决策地图：

```markdown
## Choose the path

- For a Python package, read `references/python-release.md`.
- For a container image, read `references/container-release.md`.
- For a documentation-only release, read `references/docs-release.md`.
- If the release combines artifact types, read only the guides for those artifacts.
```

这给每个引用一个可观察的加载条件。「阅读 `references/` 了解更多」则没有。

保持引用图浅层。官方指南建议从 `SKILL.md` 直接链接，避免过深的链条。一跳让可达性可测试，并降低所需约束从未进入上下文的概率。

```figure
skill-reference-map
```

### 目录预算与激活上下文是两种预算

令 `c_i` 为技能 `i` 的序列化目录成本，`B_c` 为目录预算，`b_j` 为激活正文成本，`r_k` 为实际加载的资源。

```text
catalog_cost = sum(c_i for every published skill)
active_cost = sum(b_j for every activated skill) + sum(r_k for every disclosed resource)
```

减少一种预算不会自动减少另一种。简短描述可以节省目录空间，但一个已激活的 900 行正文仍会压垮任务。只有运行时和指令真正避免加载无关分支时，把正文拆分到引用中才能降低激活成本。

当上下文窗口大小已知时，Codex 目前把初始技能列表预算设为上下文窗口的 2%。8,000 字符值仅在窗口大小未知时作为回退；它不是与 2% 规则叠加的第二个上限。当目录超出适用预算时，描述可能被缩短或省略。请把这些数字视为 Codex 当前策略，而不是 Agent Skills 标准的属性。

### 资源路径是信任边界

技能应只读取其包内的文件。字面字符串前缀检查还不够：

```text
references/../../../../.ssh/config
references/external-link -> /private/company-secrets
```

用文件系统语义解析包根和候选路径，拒绝绝对路径输入，并验证解析后的候选仍位于解析后的根目录之下。在发现之前决定是否允许符号链接。如果允许，每次都要检查解析后的目标。

```figure
skill-resource-containment
```

路径包含并不能建立内容信任。一个位于包内的合法引用仍可能包含恶意指令。第 26 课处理这种威胁。

### 加载过程必须可观测

记录披露事件，但不要记录密钥：

```json
{
  "event": "skill.resource.loaded",
  "skill": "release-readiness",
  "resource": "references/python-release.md",
  "reason": "candidate contains pyproject.toml",
  "bytes": 2840
}
```

原因字段把一次上下文选择变成可审查的证据。它还有助于找出那些导致 Agent「以防万一」加载每个文件的指令。

## 构建它

`code/main.py` 构建了一个确定性的发现与披露引擎。

发现接口包括：

- `Scope`，用于来源与优先级元数据；
- `SkillCandidate`，表示尚未校验的文件系统候选；
- `discover_scope(scope)`，枚举直接子技能目录；
- `resolve_collisions(candidates, precedence)`，应用一种声明的策略；
- `CatalogEntry` 和 `build_catalog(...)`，发布有界元数据；
- `CatalogBudget`，统计序列化条目，而不假装字符就是通用 token。

披露接口包括：

- `load_skill_body(entry, ...)`，用于第 2 层激活；
- `validate_reference(skill_dir, reference)`，用于路径包含检查；
- `load_reference(...)`，用于有界的第 3 层读取。

运行实验：

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/24-skill-discovery-and-progressive-disclosure
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

这个代码块需要本地克隆，并可从该克隆内的任意工作目录解析仓库根目录。

演示会创建临时的项目作用域和用户作用域，插入一个冲突，在刻意设置的小预算下构建目录，激活一个技能，并分别尝试一次合法引用读取和一次路径遍历逃逸。不会安装任何永久文件。

### 为什么发现是浅层的

`discover_scope` 检查直接子目录中的 `SKILL.md`。它不会递归地把每个嵌套的 `SKILL.md` 都当作独立包。这保持了包边界，并避免意外发布已安装技能内部的示例或夹具。

### 为什么实验不解析任意 YAML

实验只支持其目录所需的标量 frontmatter。生产运行时应使用安全的 YAML 解析器，具备显式 schema、大小限制，并禁用自定义对象构造。「仅用标准库」是教学约束，不是悄悄发明一种不完整 YAML 方言的许可。

## 使用它

把这份清单应用于任何发现适配器：

1. 列出每个已配置的根目录以及谁可以写入。
2. 说明是否允许符号链接包。
3. 校验包名、目录名、必填元数据和入口正文大小。
4. 在内部身份中保留来源和作用域。
5. 声明并测试重名行为。
6. 测量发送给模型的确切序列化目录。
7. 记录正文或资源被加载的原因。
8. 让资源读取始终位于解析后的包根之内。
9. 当被引用的文件缺失时明确失败。
10. 当安装或策略变化时重建目录。

## 交付它

本课产出 `skill-catalog-builder` 包。它扫描显式排序的根目录，拒绝符号链接的入口文件和名称与目录不匹配的情况，解析跨作用域冲突，拒绝同优先级重复项，并让选中的元数据符合声明的条目数、描述长度和序列化字符预算。

它的 JSON 报告包含选中条目、被遮蔽候选、省略条目、校验错误、优先级和预算使用情况。正文与引用加载仍是独立的运行时操作，因此目录构建器不会执行脚本，也不会把整个包放入上下文。

## 练习

1. 添加一个插件作用域，并将其置于用户和内置优先级之间。用测试证明冲突结果。
2. 将冲突策略从高优先级胜出改为按名称限定。在目录中保留两个条目。
3. 为 `load_reference` 添加字节大小限制。测试一个恰好达到限制的文件和一个超出 1 字节的文件。
4. 创建两个听起来几乎相同的描述。改写它们，使触发边界不再重叠。
5. 添加一个包含每个引用和脚本哈希的清单。在加载资源之前检测它是否被修改。
6. 对演示进行埋点，分别报告第 1 层、第 2 层和第 3 层的字节数。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 技能发现 | 「找到每个 SKILL.md」 | 搜索配置的作用域、校验包、附加来源信息并应用策略 |
| 技能目录 | 「已安装技能的列表」 | 面向模型、面向合格包的紧凑路由元数据 |
| 冲突策略 | 「哪个重复项胜出」 | 针对来自不同来源的同名候选的声明规则 |
| 渐进式披露 | 「懒加载」 | 从目录到正文再到分支特定资源的分阶段上下文准入 |
| 引用图 | 「技能链接的文件」 | 可达的资源结构及其加载条件 |
| 路径包含 | 「留在文件夹里」 | 验证解析后的资源目标仍位于解析后的包根之内 |

## 延伸阅读

- [Agent Skills 规范](https://agentskills.io/specification)，了解包形态和渐进式披露层级。
- [优化技能描述](https://agentskills.io/skill-creation/optimizing-descriptions)，了解目录路由元数据。
- [Agent Skills 最佳实践](https://agentskills.io/skill-creation/best-practices)，了解直接引用和入口文件大小。
- [OpenAI：构建技能](https://learn.chatgpt.com/docs/build-skills)，了解当前 Codex 的发现作用域和目录限制。
