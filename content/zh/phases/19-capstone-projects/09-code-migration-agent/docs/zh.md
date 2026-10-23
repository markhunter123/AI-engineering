# 项目 09 —— 代码迁移智能体（仓库级语言 / 运行时升级）

> 亚马逊的 MigrationBench（Java 8 到 17）和 Google 的 App Engine Py2-to-Py3 迁移器设定了 2026 年的标杆。Moderne 的 OpenRewrite 能在规模上做确定性的 AST 重写。Grit 用 codemod 风格的 DSL 瞄准同一个问题。生产模式把两者结合起来：用确定性底座做安全重写，用智能体层处理含糊的情况，用沙箱做逐分支构建，再用一套在 PR 开启前就变绿的测试框架。项目要迁移 50 个真实仓库，并公布通过率和失败分类法。

**Type:** Capstone
**Languages:** Python (agent), Java / Python (targets), TypeScript (dashboard)
**Prerequisites:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure)
**Phases exercised:** P5 · P7 · P11 · P13 · P14 · P15 · P17
**Time:** 30 hours

## 问题

大规模代码迁移是 2026 年编码智能体最干净的生产应用之一。判据显而易见（迁移后测试套件会通过吗？），回报是真实的（一次 Java 8 机群迁移是人力规模的项目），基准是公开的（MigrationBench 的 50 仓库子集）。Moderne 的 OpenRewrite 负责确定性那一侧。智能体层负责 OpenRewrite 配方处理不了的一切：含糊的重写、构建系统漂移、长尾语法、传递依赖断裂。

你会构建一个智能体：输入一个 Java 8 仓库（或 Python 2 仓库），输出一个 CI 全绿的迁移分支。你会测量通过率、测试覆盖率保持情况、每仓库成本，并建立失败分类法。与纯确定性基线的并排对比，会告诉你智能体的价值到底在哪里。

## 概念

流水线分两层。**确定性底座**（Java 用 OpenRewrite，Python 用 libcst）安全地完成大部分机械式重写：import、方法签名、空安全改写、try-with-resources、废弃 API 替换。它速度快，产生的 diff 可审计。**智能体层**（基于 Claude Opus 4.7 和 GPT-5.4-Codex 的 OpenAI Agents SDK 或 LangGraph）处理配方无能为力的情况：构建文件升级（Maven/Gradle/pyproject）、传递依赖冲突、测试抖动、自定义注解。

每个仓库分到一个预装目标运行时的 Daytona 沙箱。智能体反复迭代：跑构建、分类失败、施加修复、重跑。硬性上限：每仓库 30 分钟、8 美元、20 个智能体轮次。如果所有测试通过且覆盖率变化不为负，分支就开 PR。否则仓库连同证据归入某个失败类别。

失败分类法才是交付物。50 个仓库里，坏在哪里？传递依赖？自定义注解？构建工具版本？与迁移无关的测试抖动？每个类别给出计数和一个范例 diff。后续写配方的人可以瞄准前三类。

## 架构

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## 技术栈

- 确定性底座：OpenRewrite（Java）或 libcst（Python）
- 智能体：基于 Claude Opus 4.7 + GPT-5.4-Codex 的 OpenAI Agents SDK 或 LangGraph
- 沙箱：逐分支的 Daytona devcontainer，预装目标运行时（Java 17 / Python 3.12）
- 构建系统：Maven、Gradle、uv（Python）
- 基准：Amazon MigrationBench 的 50 仓库子集（Java 8 到 17）、Google App Engine 的 Py2 到 Py3 仓库
- 测试框架：并行运行器，Java 用 Jacoco、Python 用 coverage.py 统计覆盖率
- 可观测性：Langfuse + 每仓库一个 trace 包，包含每个 diff 片段
- 仪表盘：失败分类仪表盘，带逐类计数和范例 diff

```figure
ce-migration-funnel
```

## 构建它

1. **配方阶段。** 先跑 OpenRewrite（Java）或 libcst（Python）配方。抓住那 70-80% 机械式的迁移。作为「recipe」提交。

2. **构建试跑。** Daytona 沙箱：安装目标运行时，跑构建。如果绿了，直接跳到测试。如果红了，交给智能体。

3. **智能体循环。** 用带工具的 LangGraph：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。智能体对失败分类（依赖、语法、测试、构建工具），并施加针对性修复。然后重跑。

4. **预算上限。** 每仓库 30 分钟墙钟时间、8 美元成本、20 个智能体轮次。任何一项超限都立即停止，并把当前 diff 归入「budget_exhausted」。

5. **测试 + 覆盖率门禁。** 构建转绿后，跑测试套件。与基准仓库比较覆盖率。如果覆盖率下降超过 2%，归入「coverage_regression」。

6. **开 PR。** 成功时推送分支，开 PR，附上 diff，以及哪些配方被应用、哪些提交由智能体编写。

7. **失败分类法。** 对每个失败的仓库打上类别标签：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。构建仪表盘。

8. **50 仓库跑批。** 在 MigrationBench 子集上执行。报告逐类通过率、每仓库成本、覆盖率保持情况，以及与纯确定性基线的对比。

## 使用它

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## 交付它

`outputs/skill-migration-agent.md` 是交付物。给定一个仓库，它先执行确定性配方，再跑智能体循环，产出一个可合并的绿色迁移分支；做不到时，就把仓库归入某个分类类别。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | MigrationBench 通过率 | 50 仓库子集的 pass@1 |
| 20 | 测试覆盖率保持 | 相对 base 的平均覆盖率变化 |
| 20 | 每个迁移仓库的成本 | 通过运行上的 $/repo |
| 20 | 智能体 / 确定性工具集成 | OpenRewrite 处理的修复与智能体编写的修复比例 |
| 15 | 失败分析报告 | 分类法完整度与范例 |
| **100** | | |

## 练习

1. 只跑 OpenRewrite（不用智能体）执行迁移流水线。对比通过率与完整流水线。找出只有智能体能补齐差异的那些案例。

2. 实现「lint 干净」检查：迁移后跑风格 linter（Java 用 spotless，Python 用 ruff）。如果出现新的 lint 错误就让 PR 失败。测量「覆盖率保持但风格回退」的比例。

3. 加一个「最小 diff」优化器：智能体的分支通过测试后，用第二遍修剪不必要的改动。报告 diff 体积缩减。

4. 扩展到第三种迁移：Node 18 到 Node 22。复用沙箱封装；把配方层换成自定义 codemod。

5. 把首次构建转绿时间（TTFGB）当作 UX 指标测量。目标：p50 在 10 分钟以内。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Deterministic substrate | 「配方引擎」 | OpenRewrite / libcst：带安全保证的声明式 AST 重写 |
| Codemod | 「代码修改程序」 | 机械地改写源码的重写规则 |
| Build drift | 「工具版本偏移」 | Maven / Gradle / uv 在大版本之间的细微行为差异 |
| Failure class | 「分类桶」 | 仓库未能迁移的标注原因：依赖、语法、测试、构建工具、预算 |
| Coverage delta | 「覆盖率保持」 | 从基准分支到迁移分支的测试覆盖率百分比变化 |
| Agent turn | 「工具调用轮」 | 智能体循环中的一次 计划 -> 行动 -> 观察 周期 |
| Budget exhaustion | 「撞到上限」 | 仓库耗尽了 30 分钟 / 8 美元 / 20 轮的限制仍未通过 |

## 延伸阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) —— 2026 年的标准基准
- [Moderne.io OpenRewrite 平台](https://www.moderne.io) —— 确定性底座参考
- [OpenRewrite 文档](https://docs.openrewrite.org) —— 配方编写
- [Grit.io](https://www.grit.io) —— 另一种 codemod DSL
- [OpenAI 沙箱化迁移 cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) —— Agents SDK 参考
- [Google App Engine Py2 到 Py3 迁移器](https://cloud.google.com/appengine) —— 另一种迁移基准
- [libcst](https://github.com/Instagram/LibCST) —— Python 确定性底座
- [Daytona 沙箱](https://daytona.io) —— 逐分支沙箱参考
