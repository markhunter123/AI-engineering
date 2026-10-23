# 项目 16 — GitHub Issue 到 PR 的自主 Agent

> 给 issue 打上标签，就得到一个 PR——这是 2026 年自主编码 Agent 的产品形态：在云沙箱中运行 Agent，验证测试通过，并提交一个附有变更理由、可进入评审的 PR。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud 与 Google Jules 都已交付这一形态。难点在于自动复现仓库的构建环境、防止凭据泄露、按仓库执行预算，以及确保 Agent 无法 force-push。本项目构建自托管版本，并在成本与通过率上与托管方案对比。

**Type:** Capstone
**Languages:** Python (agent), TypeScript (GitHub App), YAML (Actions)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure)
**Phases exercised:** P11 · P13 · P14 · P15 · P17
**Time:** 30 hours

## 问题

异步云端编码 Agent 是与交互式编码 Agent（项目 01）不同的产品品类。它的交互界面就是一个 GitHub 标签。你给 issue 打上 `@agent fix this` 标签，一个 worker 会在云沙箱中启动，克隆仓库、运行测试、编辑文件、验证，然后开一个 PR，PR 正文中包含 Agent 的变更理由。没有交互循环，也没有终端。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 与 Factory Droids 都汇聚到这一形态。

工程挑战很具体：环境复现（Agent 必须在没有缓存开发镜像的情况下从零构建仓库）、不稳定测试（必须重跑或隔离）、凭据作用域（使用最小细粒度权限的 GitHub App）、按仓库按天执行预算，以及禁止 force-push 的策略。本项目衡量通过率、成本与安全性，并与托管方案对比。

## 概念

触发器是 GitHub webhook（issue 标签或 PR 评论）。分发器把工作入队到 ECS Fargate 或 Lambda。worker 把仓库拉取到 Daytona 或 E2B 沙箱中，并依据仓库推断出一个通用 Dockerfile（语言、框架）。Agent 针对 Claude Opus 4.7 或 GPT-5.4-Codex 运行 mini-swe-agent 或 SWE-agent v2 循环。它反复迭代：读代码、提出修复、应用补丁、运行测试。

验证是门禁步骤。PR 创建之前，完整 CI 必须在沙箱中通过。系统会计算覆盖率变化；如果下降超过阈值，PR 仍然会创建，但会被打上 `needs-review` 标签。Agent 把变更理由作为 PR 描述发布，并开一个 `@agent` 线程，评审者可以在此 @ 它追问。

安全性通过 GitHub 的两个不同界面来限定：App 提供一个短期 installation token，具有 `workflows: read` 以及较窄的仓库内容/PR 权限；分支保护（而非 App 权限）强制执行「不得直接写入 `main`」与「禁止 force-push」——App 永远不加入绕过名单。对 `.github/workflows` 做路径级只读访问并非 GitHub App 的真实原语，因此 Agent 对文件编辑的允许列表必须在 worker 侧执行。按仓库按天的预算上限在分发器处执行（例如每个仓库每天最多 5 个 PR，每个 PR 20 美元）。

## 架构

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## 技术栈

- 触发：带细粒度 token 的 GitHub App；通过 Lambda 或 Fly.io 接收 webhook
- Worker：ECS Fargate 任务（或 GitHub Actions 自托管 runner）
- 沙箱：每个任务一个 Daytona devcontainer 或 E2B 沙箱
- Agent 循环：mini-swe-agent 基线，或基于 Claude Opus 4.7 / GPT-5.4-Codex 的 SWE-agent v2
- 检索：tree-sitter 仓库地图 + ripgrep
- 验证：沙箱内完整 CI + 覆盖率变化门禁
- 可观测性：Langfuse，按 PR 归档追踪，并从 PR 正文链接
- 预算：按仓库的每日美元上限；每个仓库每天最大 PR 数

```figure
cf-issue-to-pr
```

## 构建它

1. **GitHub App。** 使用细粒度 installation token：issues 读写、pull_requests 写、contents 读写、workflows 读。分支保护（唯一能做到这一点的界面）强制执行「不得直接推送到 `main`」与「禁止 force-push」；App 不在绕过名单中。worker 在拟议的 diff 上以允许列表检查的方式执行「不得在 `.github/workflows` 下写入」，因为 GitHub App 的权限并不按路径限定。

2. **Webhook 接收器。** Lambda 函数接收 issue 标签 / PR 评论 webhook。按 `@agent fix this` 标签过滤。入队到 SQS。

3. **分发器。** 从 SQS 取出任务。执行按仓库按天的预算。启动一个 ECS Fargate 任务，带上仓库 URL、issue 正文和一个全新的 Daytona 沙箱。

4. **环境推断。** 检测语言（Python、Node、Go、Rust）与包管理器（uv、pnpm、go mod、cargo）。如果不存在 Dockerfile，则即时生成一个。

5. **Agent 循环。** mini-swe-agent 或 SWE-agent v2，使用 Claude Opus 4.7。工具：ripgrep、tree-sitter 仓库地图、read_file、edit_file、run_tests、git。硬性限制：20 美元成本、30 分钟墙钟时间、30 个 Agent 轮次。

6. **验证。** 循环结束后，在沙箱内运行完整测试套件。通过 jacoco / coverage.py 计算覆盖率变化。若 CI 失败：停止，不创建 PR。若覆盖率下降超过 2%：创建 PR 并打上 `needs-review` 标签。

7. **PR 发布。** 推送 Agent 分支。通过 GitHub API 创建 PR，包含：标题、变更理由、diff 摘要、追踪 URL、成本、轮次。

8. **凭据卫生。** worker 使用短期的 GitHub App installation token 运行。日志在归档前清除密钥。

9. **评测。** 30 个难度各异的预置内部 issue。测量通过率、PR 质量（diff 大小、风格、覆盖率）、成本、延迟。在同一批 issue 上与 Cursor Background Agents 和 AWS Remote SWE Agents 对比。

## 使用它

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## 交付它

`outputs/skill-issue-to-pr.md` 是交付物：一个 GitHub App + 异步云 worker，把打过标签的 issue 变成可进入评审的 PR，并控制成本上限与凭据作用域。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 30 个 issue 上的通过率 | 端到端成功（CI 通过 + 覆盖率合格） |
| 20 | PR 质量 | diff 大小、覆盖率变化、风格符合度 |
| 20 | 每个已解决 issue 的成本与延迟 | 每个 PR 的美元成本与墙钟时间 |
| 20 | 安全性 | 限定作用域的 token、按仓库预算、禁止 force-push、凭据卫生 |
| 15 | 运维体验 | 变更理由评论、重试入口、@ 提及追问 |
| **100** | | |

## 练习

1. 增加「修复不稳定测试」模式：标签 `@agent stabilize-flake TestX` 会在沙箱内运行该测试 50 次，并提出一个使其稳定的最小改动。

2. 在三个共同 issue 上比较成本与 Cursor Background Agents。报告各工具在哪些方面占优。

3. 实现预算看板：按仓库按天成本、按用户成本。出现异常时告警。

4. 构建「演练」模式：不运行 CI 就创建草稿 PR，让评审者以低成本查看计划。

5. 增加保留策略：超过 7 天未合并的 PR 分支自动删除。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|------------------------|
| GitHub App | 「限定作用域的机器人身份」 | 具有细粒度权限 + 短期 installation token 的 App |
| 异步云 Agent | 「后台 Agent」 | 在云沙箱而非终端中运行的非交互式 worker |
| 环境推断 | 「Dockerfile 合成」 | 检测语言 + 包管理器，不存在 Dockerfile 时生成一个 |
| 验证 | 「沙箱内 CI」 | 创建 PR 之前在 worker 内运行完整测试套件 |
| 覆盖率变化 | 「覆盖率保持」 | 从基线到 Agent 分支的测试覆盖率百分比变化 |
| 按仓库预算 | 「每日上限」 | 在分发器处执行的美元与 PR 数量上限 |
| 变更理由 | 「PR 正文说明」 | Agent 对改动内容与原因的总结；PR 正文中必需 |

## 延伸阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 典型的异步云 Agent 参考
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI 参考
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — 商业替代方案
- [OpenAI Codex（云端）](https://openai.com/codex) — 托管竞品
- [Google Jules](https://jules.google) — Google 的托管版本
- [Factory Droids](https://www.factory.ai) — 另一商业参考
- [GitHub App 文档](https://docs.github.com/en/apps) — 限定作用域的机器人身份
- [Daytona 云沙箱](https://daytona.io) — 参考沙箱
