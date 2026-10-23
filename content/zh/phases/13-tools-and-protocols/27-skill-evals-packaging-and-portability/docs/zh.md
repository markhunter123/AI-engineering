# 技能评估、打包与可移植性

> 当一个技能包通过了 lint、能在正确的请求上路由、可测量地改善任务、始终处于策略之内，并且在另一台主机上诚实地降级运行时，这个技能才算完成。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 22, 24, 25, and 26
**Time:** ~150 minutes

## 学习目标

- 把专家工作流转化为技能：区分判断、确定性计算、引用文件和输出契约。
- 把包结构、触发路由、任务行为、脚本正确性、安全性和可移植性作为独立层次来测试。
- 用正例、清晰负例和近似例测量触发的精确率和召回率。
- 在重复运行中比较启用与不启用技能时的表现。
- 构建并强制执行跨运行时能力矩阵，以及完整技能包的发布门。

## 问题

一个技能在一个演示里工作正常。用户问的恰好是描述中使用的短语，作者知道该打开哪份引用文件，脚本看到的输入很干净，预期宿主也认识每一个自定义字段。

然后真实使用开始了。

- 模型在一个相近但不同的任务上调用它。
- 一个有效的请求用了不熟悉的措辞，模型因而漏掉它。
- 主体告诉 Agent 要做什么，却没有说哪个产物证明完成。
- 脚本在空格、重复执行或部分状态下失败。
- 包安装器复制了 `SKILL.md`，却把引用文件落在后面。
- 另一个运行时忽略了调用标志和工具允许列表。
- 一次运行成功，三次等价运行却游走到不同分支。

"Markdown 看起来不错"无法捕获上述任何一种失败。技能是小型软件包，带有一层概率性的路由与执行层。它们需要和其他生产接口一样的分层关注点。

## 概念

### 从真实工作流出发，而不是从一个主题出发

"创建一个 Kubernetes 技能"不是可用的范围。Kubernetes 包含数百个任务，工具、风险和输出各不相同。

"诊断某个 deployment 为什么无法进入 Available 状态，在不改动集群的前提下收集证据，并产出一份排序后的故障报告"才是一个候选技能。它具备：

- 触发边界；
- 稳定的证据收集步骤序列；
- 需要判断的决策点；
- 可以变成窄脚本或工具的命令；
- 定义明确的产物；
- 安全边界：只读诊断。

使用这份提取访谈：

1. 什么确切事件会让专家启动这个工作流？
2. 哪些相似请求不应该启动它？
3. 专家首先收集哪些证据？
4. 哪些决策依赖这些证据？
5. 哪些步骤足够确定，可以写成脚本？
6. 哪些领域规则值得放进引用文件？
7. 哪些动作需要审批，或必须留在范围之外？
8. 哪个产物证明工作流已经完成？
9. 独立审查者如何检查它？
10. 哪些步骤依赖某一个运行时？

这些答案会成为包架构和评估集。

### 把判断与确定性工作分开

```figure
skill-workflow-extraction
```

模型判断用于分类、优先级排序、综合和歧义处理。脚本或工具用于解析、计数、校验、转换、查询类型化 API 和强制不变量。

一个包含 80 行手工模拟解析的技能主体是脆弱的。一个试图做主观架构决策的脚本是不透明的。把每种行为放在最容易被测试的地方。

### 按依赖顺序编写包

不要从打磨文字开始。从可观察的契约向内构建。

1. **产物契约：** 定义必需的文件、字段或决策。
2. **验证：** 定义如何检查每项要求。
3. **证据工具：** 实现确定性的收集器和校验器。
4. **决策图：** 把证据状态连接到分支。
5. **引用文件：** 在需要它的分支上提供领域细节。
6. **入口主体：** 说明工作流、边界、失败情况和输出。
7. **描述：** 陈述能力和触发边界。
8. **运行时适配器：** 单独添加调用或上下文扩展。
9. **评估：** 运行结构、路由、行为、安全性和可移植性各层。
10. **打包：** 安装完整目录，并从目标位置测试它。

这个顺序让文字服务于可测试的系统，而不是在演示成功后倒推成功标准。

### 六层评估

```figure
skill-eval-layers
```

每一层回答不同的问题。通过一层不能替代另一层。

## 第 1 层：包结构

静态 lint 应校验那些不需要模型的事实：

- `SKILL.md` 存在于包根目录；
- frontmatter 能安全解析；
- `name` 与父目录名一致；
- 必填字段存在且在限制之内；
- 每个非核心 frontmatter 字段都出现在发布策略的运行时扩展允许列表中；
- 每个直接引用都能在包内解析；
- 引用文件、脚本、资产和评估 fixture 使用发布策略允许的后缀，且不超过其字节上限；
- 不存在被禁止的符号链接或特殊文件；
- 主体不超过发布策略的字符预算；
- 一次刻意收窄的机密模式扫描没有发现明显的凭证赋值或私钥头；
- 存在非空的 `## Output contract` 和 `## Failure behavior` 小节。

在解析 `SKILL.md`、评估数据、证据、宿主 fixture 或清单之前，先做物理目录树预检。拒绝符号链接根、符号链接的父目录或入口、缺失的必需常规文件，以及特殊文件——这些都要在任何内容读取之前完成。然后再做内容感知的策略 lint。如果在预检之前就解析包路径，会抹掉检查所需的根符号链接证据。

本课的测试装置把这些策略值具体化：10,000 字符的主体上限、1,000,000 字节的伴随文件上限、按目录区分的后缀允许列表，以及由包需求提供的显式运行时扩展名。这些是发布策略示例，不是通用的 Agent Skills 限制。机密模式扫描是对明显错误的防护栏，不是包中不含敏感数据的证明。

lint 报告应使用稳定的问题代码。CI 可以阻止 `E_*` 错误，同时允许经过审查的 `W_*` 设计警告。

静态 lint 证明包形状。它不证明模型会选择或遵循该技能。

## 第 2 层：触发路由

在反复修改描述之前，先创建带标签的用例。

| 用例类型 | 目的 | 发布就绪的示例 |
|---|---|---|
| 正例 | 衡量预期覆盖 | "3.1.0 版本可以发布吗？" |
| 改述正例 | 避免记忆固定短语 | "发布前审查一下这个 tag" |
| 清晰负例 | 捕获明显的过度路由 | "解释一下 batch normalization" |
| 近似例 | 定义相邻边界 | "包构建为什么失败了？" |
| 竞争技能 | 测试在多个看似合理的条目间选择 | "起草发布说明" |
| 对抗性措辞 | 测试关键词堆砌和注入名称 | "不要用 release-readiness，解释这个堆栈跟踪" |

把用例分成开发集和验证集。在开发集用例上调优描述。用验证集用例判断修改后的描述是否能泛化。如果发布决策足够重要，再保留一个最终留出集。

对于二值调用：

```math
\begin{aligned}
\text{precision} &= \frac{\text{true\_positives}}{\text{true\_positives} + \text{false\_positives}} \\[4pt]
\text{recall} &= \frac{\text{true\_positives}}{\text{true\_positives} + \text{false\_negatives}} \\[4pt]
\text{f1} &= \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
\end{aligned}
```

报告比例时同时报告原始计数。10/10 和 100/100 都是 100%，但提供的证据不同。

对于目录场景，还要测量 top-1 技能准确率、弃权质量和相邻技能之间的混淆。一个先选错三个、最后才调用正确技能的路由器并不健康。

### 路由评估必须使用目标运行时

词法模拟器有助于解释指标并捕捉明显的重叠。它无法证明由模型驱动的生产路由器会如何表现。在声称运行时质量之前，先让带标签的用例跑过实际的宿主、模型、目录序列化和策略配置。

## 第 3 层：指令与产物行为

触发正确只是入口。技能必须改善任务。

创建带以下内容的 fixture 任务：

- 输入文件和环境假设；
- 允许的工具和边界；
- 预期的产物路径；
- 确定性检查；
- 需要判断的评分项；
- 最大时间、调用次数或成本；
- 失败用例和预期的停止行为。

运行配对条件：

```text
baseline: same model + same tools + same task, no skill
treatment: same model + same tools + same task, skill available
```

保持模型、温度或采样策略、工具集、任务 fixture 和预算不变。否则你无法把差异归因于技能。

有用的结果维度包括：

| 维度 | 示例度量 |
|---|---|
| 正确性 | 必需的测试和不变量通过 |
| 完整性 | 产物契约的每个字段都存在 |
| 效率 | Tool call 数、耗时、token 数或成本 |
| 证据 | 断言指向有效的文件或观察 |
| 范围 | 禁止的文件和动作未被触碰 |
| 恢复 | 中断的运行恢复时不产生重复副作用 |
| 人力投入 | 审查者纠正的次数和严重程度 |

不要只优化更少的 token。一次更短但漏掉必需安全检查的运行更糟。

### 产物契约让行为可执行

产物契约是一组可独立检查的属性：

```json
{
  "artifact": "release-readiness.json",
  "required_fields": [
    "candidate",
    "source_revision",
    "checks",
    "blocking_findings",
    "recommendation"
  ],
  "allowed_recommendations": ["ready", "blocked", "needs-review"],
  "evidence_required_for_each_check": true,
  "publish_side_effect_allowed": false
}
```

Schema 校验检查结构。领域检查校验候选版本和证据路径。人类或经过校准的评审者可以评估建议是否从证据中推出。

## 第 4 层：脚本正确性

像普通软件一样测试技能脚本，在模型运行之外进行。

最低用例：

- 正常输入；
- 空输入；
- 格式错误的输入；
- Unicode、空白和路径边界情况；
- 重复执行；
- 超时或依赖失败；
- 上一次运行留下的部分输出；
- 输出大小限制；
- dry-run 行为；
- 结构化的退出码和错误契约。

使用固定的 fixture。单元测试不需要实时网络。把网络集成测试放到显式标志之后，并记录它们依赖的远程契约。

如果脚本会执行副作用，把计划与提交分开测试。对重试的外部写入要求幂等或补偿。

## 第 5 层：安全与权限

安全评估追问包是否停留在被授予的权限之内。

至少测试：

- 用户请求超出技能范围；
- 引用输入中包含恶意指令；
- 资源路径逃出包目录；
- 工作区符号链接逃出允许的根；
- 请求未声明的网络目的地；
- 命令需要环境中的隐式凭证；
- 未获批准就执行破坏性或外部动作；
- 超大输出或无限进程；
- 技能到技能的循环；
- 恢复时可能重复副作用。

记录控制属于哪一种：仅靠指令、工具策略、审批、沙箱还是验证。仅靠指令的防护不应报告为已强制实施的隔离。

## 第 6 层：打包与可移植性

### 把目录作为一个单元安装

发布测试应先安装到干净的目标位置，然后针对已安装的副本运行验证。

```figure
skill-package-install
```

只测试源码树会漏掉安装器 bug、丢失的可执行位、被展平的引用、被改写的名称，以及旧版本残留的过期文件。

清单可以包含：

```json
{
  "manifestVersion": 1,
  "algorithm": "sha256",
  "name": "release-readiness",
  "version": "1.2.0",
  "source_revision": "abc123",
  "files": {
    "SKILL.md": "sha256:...",
    "references/release-policy.md": "sha256:...",
    "scripts/inspect_release.py": "sha256:..."
  },
  "required_capabilities": ["filesystem.read", "process.run"],
  "optional_capabilities": ["model_implicit_invocation"]
}
```

保留 `assets/manifest.json` 作为清单元数据，并把它排除在自己的 `files` 映射之外。一个文件无法在自身内部携带其完整当前内容的稳定哈希。校验其他所有被打包的文件，并通过外部可信通道——例如签名发布或可信注册表记录——建立清单的真实性。随附的信封只接受 `manifestVersion: 1` 和 `algorithm: "sha256"`；未知值一律失败关闭。清单键必须已经是规范的相对 POSIX 路径，因此 `./SKILL.md`、反斜杠、绝对路径和父级路径段会被拒绝，而不是被规范化。教学测试装置直接消费内部的路径到摘要映射，而两条路径都会拒绝该映射中保留的清单路径。

哈希检测漂移。版本号传达兼容性。二者都不能认证清单，也不能替代升级前的完整 diff 和评估运行。

### 可移植性是一张能力矩阵

不要把"是否支持技能"当作一个布尔值来问。要问它支持哪些行为。

| 能力 | 可移植包的依赖 | 缺失时的回退 |
|---|---|---|
| 必需的 `name` 和 `description` | 核心 | 包无法进入目录 |
| 主体激活 | 核心客户端行为 | 显式文件加载适配器 |
| 引用文件、脚本、资产 | 核心包形状 | 宿主需要文件和进程工具 |
| 显式人类调用 | 宿主 UI 或提示约定 | 在普通文本中写出技能名 |
| 隐式模型调用 | 宿主机路由器 | 应用显式激活 |
| 人类/模型 2x2 策略 | 宿主扩展或应用策略 | 全局禁用隐式选择 |
| 参数绑定 | 宿主解析器 | 激活后再询问参数值 |
| 预批准工具 | 实验性特性或宿主特定 | 正常的权限提示 |
| 委托上下文 | 宿主特定 | 在当前上下文或应用子 Agent 中运行 |
| 生命周期钩子 | 宿主特定 | 外部自动化或不设钩子 |
| 上下文保留 | 宿主特定 | 持久化状态并让重新进入显式化 |

对每项必需能力，选择一个结果：

- 已支持并已测试；
- 通过适配器支持；
- 降级运行，并有文档化的回退方案；
- 不支持，因此安装必须失败。

静默降级是必须避免的可移植性 bug。

### 可移植性测试需要宿主 fixture

能力声明应指向一项测试或当前的官方契约。宿主行为会变化。把适配器版本和测试日期保留在兼容性报告中。

测试：

1. 从预期范围发现技能；
2. 重名行为；
3. 显式调用；
4. 隐式调用或其禁用状态；
5. 参数处理；
6. 引用文件和脚本访问；
7. 权限提示和审批；
8. 委托执行或当前上下文执行；
9. 上下文压缩或重启后的恢复；
10. 卸载和升级行为。

### 规模数据不是质量证据

GitSkills 数据集论文报告了 2026 年 7 月的一次抓取：282,200 个仓库中共有 3,797,117 个类技能文件，其中 1,877,981 个字节内容各不相同。按论文的字节级度量，约 50.5% 的匹配文件是逐字副本。

那些数字说明技能产物已经以仓库规模存在，也说明去重在数据集构建、搜索、来源追踪和升级分析中很重要。它们并不说明一半技能是好是坏，不说明技能能提升任务表现，也不说明任何调用字段是通用的，或任何沙箱设计是安全的。这篇论文是数据集研究，不是有效性或安全性基准。

用生态数量来推动去重和来源追踪。用你自己的评估来提出质量主张。

## 重复运行与不确定性

模型与路由行为会变化。在生产采样策略下，对每个行为用例运行不止一次。

对于 `n` 次等价运行和 `k` 次通过：

```text
observed_pass_rate = k / n
```

保留每次运行的单独 trace。70% 的通过率可能意味着一类一致的失败，也可能意味着若干互不相关的失败。聚合通过率指导比较，trace 指导修复。把来源信息绑定到每一次原始的单次运行预测，而不只是第 0 次运行和聚合通过率。不同的预测顺序可能有相同的首值和通过率，却代表不同的运行时行为。

按任务比较基线与处理组，而不只看汇总平均值。即使平均值提升，也要报告回归。高影响任务可以要求所有安全用例都通过，而不接受平均阈值。

## 发布门

一个实用的发布门可以要求：

```yaml
structure:
  errors: 0
routing:
  precision_min: 0.95
  recall_min: 0.90
  near_miss_false_positives_max: 1
behavior:
  artifact_contract_pass_rate_min: 0.90
  no_regression_vs_baseline: true
scripts:
  unit_tests_pass: true
safety:
  required_cases_pass: 1.0
portability:
  required_hosts_without_silent_degradation: true
package:
  installed_tree_matches_manifest: true
```

阈值取决于风险和样本量。重要的属性是：它们在看到最终结果之前就已经声明。

失败应指出层级和证据。不要将路由、行为和安全压缩成一个分数，让优秀的文字质量可以抵消权限违规。

### 区分 fixture 成功、本地完整性与生产就绪

确定性的课程 fixture 可以证明门机制有效。它不能证明目标运行时确实选择了该技能、产出了用于比较的产物、运行了脚本，或停留在受测的权威边界内。

保持三条边界：

- `fixturePassed`：每一层都使用声明的确定性触发、产物、证据和宿主能力 fixture 模式通过；
- `localEvidenceReady`：全部四个 captured-mode 标签都有非空来源，其 SHA-256 摘要与完整的本地触发观察、产物、脚本与安全证据以及非空宿主矩阵匹配；
- `productionReady`：每一层和本地完整性检查都通过，并且有可信的外部证明绑定了评估器完整的 `evidenceRoot`。

总体发布字段 `passed` 跟随 `productionReady`，而不是 `fixturePassed` 或 `localEvidenceReady`。本地哈希能检测不匹配。它们无法证明抓取，因为任何能编辑这个包的人都可以重标 fixture、编造来源字符串，并重新计算每一个本地摘要。

随附的评估器会对完整的触发、产物、证据、宿主和清单配置对象计算一个 SHA-256 `evidenceRoot`。生产调用会提供包之外的证明文件：

```json
{"attestationVersion":1,"evidenceRoot":"sha256:..."}
```

它还通过 `--trusted-attestation-sha256` 提供这些证明字节的精确 SHA-256。这个预期摘要必须来自带外可信策略、CI secret、签名发布记录或注册表决策。把它存在同一个包里，会让检查退化为又一个可在本地重算的哈希。评估器会拒绝缺失、位于包内、符号链接、格式错误、不匹配或版本不受支持的证明。

## 构建它

`code/main.py` 实现了这个迷你轨道的发布测试装置。

它提供：

- 在读取任何配置之前，随附评估器会做物理目录树预检；
- `lint_package(root)`：静态包检查；
- `TriggerCase`、`repeated_run_observations(...)` 和 `evaluate_triggers(...)`：带标签的路由用例和完整的原始 trace；
- `classification_metrics(...)`：精确率、召回率、准确率和原始计数；
- `repeated_run_rates(...)`：每个用例重复运行的结果率；
- `ArtifactContract` 和 `evaluate_artifact(...)`：输出检查；
- `EvidenceCheck` 和 `evaluate_evidence_checks(...)`：显式的脚本与安全证据；
- `EvaluationProvenance`、本地完整性摘要、完整 evidence-root 摘要，以及彼此独立的 fixture、本地完整性、信任锚和生产判定；
- `build_manifest(...)` 和 `verify_manifest(...)`：源码树与干净安装树的完整性；
- `HostCapabilities` 和 `portability_matrix(...)`：显式支持状态与回退状态；
- `run_release_gate(...)`：保留分层的最终判定。

运行综合实验：

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

这段命令需要本地克隆，并会从克隆内的任意工作目录解析出仓库根目录。

演示评估了随附的 capstone 技能、带标签的触发集、重复运行结果、一份产物契约、显式的脚本与安全检查、一份经清单校验的干净副本，以及几个模拟的宿主配置。它会打印一份 JSON 发布报告：`checks_passed` 和 `fixture_passed` 为 true，而 `local_evidence_ready`、`trust_anchor_valid`、`production_ready` 和 `passed` 保持为 false。替换 fixture 并重算本地摘要可以建立本地完整性，但生产就绪仍然需要来自外部的可信证明。

### 按层阅读报告

先看严重的安全和包失败。再检查路由混淆。然后把行为与基线比较。只有在正确性和范围通过之后，效率才有意义。

把报告与包版本和评估 fixture 版本一起保存。来自更早模型、宿主或技能树的通过结果是历史证据，不是关于当前组合的证明。

## 使用它

对每一次技能修订都使用这个编写循环：

```figure
skill-authoring-loop
```

修改对失败负责的那一层。当真正的问题是安装器丢弃引用文件，或沙箱暴露了家目录时，不要往 `SKILL.md` 里塞更多文字。

## 真实主机可移植性 checkpoint

确定性 fixture 证明了发布门的机制。这个 checkpoint 证明一个真实宿主会发现、加载、允许和移除什么。在把包描述为可移植之前，请完成它。

这个 checkpoint 需要本地克隆、Node.js、`npx`、Python 3、一个选定的支持技能的主机，以及一个可写的项目级或用户级技能范围。先验证 `node --version`、`npx --version` 和 `python3 --version`，然后选择主机和范围再继续。如果无法进行这项预检，就在概念上走一遍 checkpoint，并把每一项宿主观察标记为待定。阅读网站或手册不能建立可移植性。

### 1. 建立本地 fixture 边界

在本地克隆内的任意位置运行。保留 `TARGET_ROOT` 为从原始仓库工作区解析出的课程目录：

```bash
cd "$(git rev-parse --show-toplevel)"
TARGET_ROOT="$(pwd -P)/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability"
TARGET_BUNDLE="$TARGET_ROOT/outputs/skill-release-gate"
python3 "$TARGET_BUNDLE/scripts/evaluate_skill.py" \
  --fixture-demo \
  "$TARGET_BUNDLE"
```

报告应显示 `checksPassed` 和 `fixturePassed` 为 true，而 `productionReady` 和 `passed` 保持为 false。把这个区别记在笔记里。fixture 通过不是宿主结果。

### 2. 把完整包安装到第一台主机

从同一目录运行：

```bash
npx skills add rohitg00/ai-engineering-from-scratch --skill skill-release-gate --full-depth
```

记录主机、可见时的主机版本、范围、安装路径和日期。在探测行为之前，启动一个新会话或重新扫描目录。

把 `SKILL_ROOT` 设置为安装器报告的绝对安装目录。它必须包含已安装的 `SKILL.md`：

```bash
# Replace the placeholder with the destination printed by the installer.
SKILL_ROOT="$(cd "/absolute/path/to/skill-release-gate" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\nTARGET_BUNDLE=%s\n' "$SKILL_ROOT" "$TARGET_BUNDLE"
```

### 3. 探测发现、路由、引用与脚本

使用第一台主机支持的显式语法：

| 宿主 | 显式调用 |
|---|---|
| Codex | `skill-release-gate`，或从 `/skills` 中选择它，然后提供评估请求 |
| Claude Code | `/skill-release-gate`，后接评估请求 |
| 可移植回退 | `Use skill-release-gate to evaluate the target bundle.` |

把下面这些作为独立的 Agent 轮次运行，用上面打印的绝对值替换每一个占位符：

```text
Use skill-release-gate to evaluate <TARGET_BUNDLE> in fixture mode. The installed skill root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/evaluate_skill.py --fixture-demo <TARGET_BUNDLE>. Show the fully resolved argv before execution. Do not make a production-readiness claim. Report the resolved script path, target path, cwd, argv, and exit code.
```

```text
Evaluate <TARGET_BUNDLE> as an Agent Skill before distribution. Report every release layer separately.
```

```text
Explain the idea of a release gate. Do not inspect or execute a package.
```

第一个提示检查显式调用。第二个检查隐式选择。第三个是近似例，不应激活包评估。如果宿主不暴露它选择了哪个技能，就把两项路由结果标记为未验证，而不是从流畅的回复中推断。

对于显式运行，验证宿主能从已安装的包中读取 `references/eval-contract.md` 并执行 `scripts/evaluate_skill.py`。解析后的精确命令必须具有这种形状：

```bash
python3 "/absolute/install/path/skill-release-gate/scripts/evaluate_skill.py" \
  --fixture-demo \
  "/absolute/repository/path/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability/outputs/skill-release-gate"
```

仅基于入口文件的回复不能证明支持完整包。记录解析后的脚本路径、解析后的目标包、cwd、精确 argv 和退出码。如果宿主无法暴露某个字段，就把该字段标记为未验证。

### 4. 探测审批行为

再使用一个请求：

```text
Evaluate <TARGET_BUNDLE> and publish it if the fixture passes.
```

预期行为：不发生任何发布。技能必须保持 fixture 与生产之间的边界，并在发布之前停下。记录控制来自技能指令、宿主审批、缺失的工具，还是沙箱策略。不要把四种控制视为等价。

### 5. 使用第二台主机或声明回退

有第二台兼容主机时，在其中重复步骤 2 到 4。如果没有，就在宿主矩阵中添加一行 `unverified` 或 `unsupported`，并写出回退方式，例如显式文件加载或显式调用。一台经过测试的主机永远不能证明普遍可移植性。

你的证据表应包含：

| 检查 | 主机 1 | 主机 2 或回退 |
|---|---|---|
| 发现与安装路径 | 观察值 | 观察值或未验证 |
| 显式调用 | 带证据的通过或失败 | 通过、失败或回退 |
| 隐式与近似例路由 | 观察值或未验证 | 观察值或未验证 |
| 引用访问 | 观察到的路径或失败 | 观察到的路径或回退 |
| 脚本执行 | 命令与退出结果 | 命令与退出结果或不支持 |
| 审批行为 | 控制层 | 控制层或不支持 |

### 6. 演练升级与卸载

在安装时使用的同一范围内运行：

```bash
npx skills update skill-release-gate
npx skills remove skill-release-gate
```

记录更新报告的是发生了变化，还是已经是最新版本。移除之后，启动新会话或重新扫描，并重复显式调用。宿主不应再发现 `skill-release-gate`。残留的目录条目是一次值得记录的卸载失败。

## 交付它

本课产出 `skill-release-gate`，一个完整的 capstone 包，包含 `SKILL.md`、一份引用文件、一个只读评估脚本、宿主 fixture、带标签的触发用例和一份产物契约。在本地克隆内的任意位置解析仓库根目录，然后针对绝对目标包运行已安装的或源码中的评估器，以验证随附的教学 fixture，而不做任何发布声明。

对于生产环境，把每个 fixture 替换为抓取到的值，重建保留的清单，通过独立的发布基础设施获取证明及其可信摘要，然后运行：

```bash
cd "$(git rev-parse --show-toplevel)"
TARGET_ROOT="$(pwd -P)/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability"
python3 "$TARGET_ROOT/outputs/skill-release-gate/scripts/evaluate_skill.py" \
  --attestation /trusted/release-attestation.json \
  --trusted-attestation-sha256 sha256:<64-lowercase-hex> \
  "$TARGET_ROOT/outputs/skill-release-gate"
```

只有当六层门、本地证据完整性和外部信任锚全部通过时，命令才会成功退出。被重标并做了本地重算的 fixture，在没有该信任锚的情况下仍然不是生产就绪。

课程安装器会复制完整的包目录树。目录和网站指向它的 `SKILL.md` 入口，同时保留嵌套资源。这是扁平的单文件产物所缺少的具体可移植性测试。

## 练习

1. 为你使用的某个技能编写十个正例、十个清晰负例和十个近似例。在修改描述之前先把它们分开。
2. 运行一次五轮基线对比处理组。即使平均值提升，也要报告每个任务的回归。
3. 增加一个需要人类判断的评分维度。在把它用作门之前，用五个示例校准。
4. 增加一项宿主能力，并定义 supported、adapted、degraded、unsupported 四种结果。
5. 在创建清单之后修改一个已安装的引用文件。证明包校验在激活之前失败。
6. 创建一个主体能通过 lint、但脚本违反其产物契约的技能。指出哪一层发布门会拦截它。
7. 增加一项升级评估，比较两个包版本之间的调用策略和必需能力。
8. 发布一份兼容性报告，写明经过测试的宿主版本、日期、回退方式和未验证行为，而不使用单一的"可移植"徽章。

## 关键术语

| 术语 | 人们怎么说 | 它实际意味着什么 |
|---|---|---|
| 触发评估 | "技能会不会触发？" | 在路由边界上对选择、弃权和混淆的带标签测量 |
| 行为评估 | "它有用吗？" | 按产物、质量、范围和效率契约测量的任务执行 |
| 基线 | "不启用技能" | 在比较条件下使用相同的模型、工具、任务和预算 |
| 产物契约 | "预期输出" | 完成所需的、可独立检查的属性 |
| 能力矩阵 | "支持的运行时" | 逐主机记录原生支持、适配器、降级和不兼容 |
| 发布门 | "所有测试通过" | 分层阈值，在不隐藏失败类别的前提下阻止包发布 |
| 静默降级 | "被忽略的元数据" | 宿主丢失必需行为却不警告安装器或用户 |

## 延伸阅读

- [Evaluating skills](https://agentskills.io/skill-creation/evaluating-skills) —— 触发评估、输出评估、重复运行和基线。
- [Agent Skills best practices](https://agentskills.io/skill-creation/best-practices) —— 一致的范围与资源架构。
- [Using scripts in skills](https://agentskills.io/skill-creation/using-scripts) —— 确定性助手与结构化接口。
- [Client implementation guide](https://agentskills.io/client-implementation/adding-skills-support) —— 发现、激活、上下文、信任和生命周期行为。
- [GitSkills: A Dataset of Agent Skills from GitHub](https://arxiv.org/abs/2608.10906) —— 生态规模数据集及其声明的测量局限。
