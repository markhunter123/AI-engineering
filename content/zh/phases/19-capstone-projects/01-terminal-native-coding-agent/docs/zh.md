# 项目 01 — 终端原生编码 Agent

> 到 2026 年，编码 Agent 的形态已经定型。一个 TUI harness、一份有状态的计划、一套沙箱化的工具面，以及一个计划、行动、观察、恢复的循环。Claude Code、Cursor 3 和 OpenCode 远看如出一辙。这个项目要求你端到端地构建一个 —— 从 CLI 输入到 pull request 输出 —— 并在 SWE-bench Pro 上与 mini-swe-agent 和 Live-SWE-agent 对比评测。你会体会到，难点不在模型调用，而在工具循环、沙箱，以及 50 轮运行的成本上限。

**Type:** Capstone
**Languages:** TypeScript / Bun (harness), Python (eval scripts)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools and protocols), Phase 14 (agents), Phase 15 (autonomous systems), Phase 17 (infrastructure)
**Phases exercised:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**Time:** 35 hours

## 问题

2026 年，编码 Agent 已成为主导性的 AI 应用类别。Claude Code（Anthropic）、带 Composer 2 和 Agent Tabs 的 Cursor 3（Cursor）、Amp（Sourcegraph）、OpenCode（112k stars）、Factory Droids 和 Google Jules，发布的都是同一架构的变体：终端 harness、带权限控制的工具面、沙箱，以及围绕前沿模型构建的计划—行动—观察循环。前沿很窄 —— Live-SWE-agent 借助 Opus 4.5 在 SWE-bench Verified 上达到 79.2% —— 但工程技艺很宽。大多数失败模式并不是模型的错误，而是工具循环不稳定、上下文污染、失控的 token 成本和破坏性的文件系统操作。

你无法从外部理解这些 Agent。你必须亲手构建一个，看着循环在第 47 轮因为 ripgrep 返回 8MB 匹配结果而崩溃，然后重建截断层。这正是这个项目的意义所在。

## 概念

harness 有四个面。**Plan** 维护一个 TodoWrite 风格的状态对象，由模型每轮重写。**Act** 派发 tool call（读取、编辑、运行、搜索、git）。**Observe** 捕获 stdout / stderr / 退出码，截断后把摘要回传。**Recover** 处理工具错误，既不撑爆上下文窗口，也不无限循环。2026 年的形态还多了一样东西：**hooks**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact` —— 可配置的扩展点，运维者在这里注入策略、遥测和防护栏。

沙箱用 E2B 或 Daytona。每个任务都在全新的 devcontainer 中运行，并以读写方式挂载一个 git worktree。harness 从不触碰宿主机文件系统。无论成功还是失败，worktree 都会被销毁。成本控制在三个层面执行：每轮 token 上限、每会话美元预算，以及硬性轮次上限（通常为 50）。可观测性层是遵循 GenAI 语义约定的 OpenTelemetry span，发送到自托管的 Langfuse。

## 架构

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## 技术栈

- Harness 运行时：Bun 1.2 + Ink 5（React-in-terminal）
- 模型访问：OpenRouter 统一 API，接入 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（用于最难的任务）
- 工具传输：Model Context Protocol StreamableHTTP（MCP 2026 修订版）
- 沙箱：E2B 沙箱（JS SDK）或 Daytona devcontainer
- 代码搜索：ripgrep 子进程，覆盖 17 种语言的 tree-sitter 解析器（预编译）
- 隔离：每个任务执行 `git worktree add`，成功 / 失败后清理
- 评测 harness：SWE-bench Pro（验证子集）+ Terminal-Bench 2.0 + 自建的 30 任务 holdout
- 可观测性：带 `gen_ai.*` semconv 的 OpenTelemetry SDK → 自托管 Langfuse
- PR 提交：使用细粒度 token 的 GitHub App，作用域限定为目标仓库

```figure
ce-agent-loop
```

## 构建它

1. **TUI 与命令循环。** 用 Ink 搭建一个 Bun 项目。接受 `agent run <repo> "<task>"`。输出分栏视图：计划面板（上）、tool call 流（中）、token 预算（下）。加入 Ctrl-C 取消功能，在退出前触发 `SessionEnd` hook。

2. **计划状态。** 定义带类型的 TodoWrite schema（pending / in_progress / done 条目，各带备注）。模型每轮以 tool call 的形式重写完整状态 —— 不要让它增量修改。把计划持久化到 `.agent/state.json`，以便崩溃后恢复。

3. **工具面。** 定义六个工具：`read_file`、`edit_file`（带 diff 预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带超时）、`git`（status / diff / commit / push）。通过 MCP StreamableHTTP 暴露，使 harness 与传输方式无关。每个工具都返回截断后的输出（每次调用上限 4k token）。

4. **沙箱封装。** 每个任务启动一个 E2B 沙箱。用 `git worktree add -b agent/$TASK_ID` 创建全新分支。所有 tool call 都在沙箱内执行。宿主机文件系统不可达。

5. **Hooks。** 实现 2026 年的全部八种 hook 类型。至少接入四个用户自编 hook：(a) `PreToolUse` 破坏性命令防护，阻止 worktree 之外的 `rm -rf`；(b) `PostToolUse` token 计账；(c) `SessionStart` 预算初始化；(d) `Stop` 写出最终的 trace 包。

6. **评测循环。** 克隆 SWE-bench Pro Python 的 30 个 issue 子集。用你的 harness 逐个运行。在 pass@1、每任务轮数和每任务美元成本上，与 mini-swe-agent（最小基线）对比。把结果写入 `eval/results.jsonl`。

7. **成本控制。** 硬性上限：50 轮、200k 上下文、每任务 5 美元。`PreCompact` hook 在 150k 处把较早的轮次总结成一个 prior-state 块，为新的观察腾出空间，同时不丢失计划。

8. **PR 提交。** 成功时，最后一步是 `git push`，再调用 GitHub API 创建 PR，正文中附上计划和 diff 摘要。

## 使用它

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## 交付它

交付物 skill 位于 `outputs/skill-terminal-coding-agent.md`。给定仓库路径和任务描述，它会在沙箱中运行完整的计划—行动—观察循环，返回 PR URL 和 trace 包。这个项目的评分标准：

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 对比基线 | 在 30 个对齐的 Python 任务上，你的 harness 对比 mini-swe-agent |
| 20 | 架构清晰度 | 计划/行动/观察的分离、hook 面、工具 schema —— 对照 Live-SWE-agent 的布局评审 |
| 20 | 安全性 | 沙箱逃逸测试、权限提示、破坏性命令防护通过红队测试 |
| 20 | 可观测性 | trace 完整性（100% 的 tool call 有 span）、每轮 token 计账 |
| 15 | 开发者体验 | 冷启动 < 2s、崩溃恢复能续上计划、Ctrl-C 能在工具执行中途干净取消 |
| **100** | | |

## 练习

1. 把后端模型从 Claude Sonnet 4.7 换成 vLLM 上服务的 Qwen3-Coder-30B。对比 pass@1 和每任务美元成本。报告开源模型在哪些地方表现较差。

2. 增加一个 `reviewer` 子 Agent，在提交 PR 前阅读 diff，并可要求进入修订循环。衡量误报的评审是否会让 SWE-bench 通过率降到单 Agent 基线以下（提示：通常是）。

3. 对沙箱做压力测试：写一个尝试 `curl` 外部 URL 的任务，再写一个尝试在 worktree 之外写入的任务。确认两者都被 PreToolUse hook 拦截。记录这些尝试。

4. 用更小的模型（Haiku 4.5）实现 `PreCompact` 摘要。衡量在 3 倍压缩下计划保真度损失多少。

5. 把 MCP StreamableHTTP 传输换成 stdio。基准测试冷启动和每次调用的延迟。为纯本地使用选出更优方案。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Harness | 「Agent 循环」 | 围绕模型、负责派发工具、维护计划状态并执行预算的代码 |
| Hook | 「Agent 事件监听器」 | 由 harness 在八种生命周期事件之一触发运行的用户自编脚本 |
| Worktree | 「Git 沙箱」 | 挂在不同路径上的关联 git checkout；用完即弃，不影响主克隆 |
| TodoWrite | 「计划状态」 | 模型每轮重写的待处理/进行中/已完成条目的带类型列表 |
| StreamableHTTP | 「MCP 传输」 | 2026 版 MCP 修订：支持双向流式传输的长连接 HTTP；取代 SSE |
| Token 上限 | 「上下文预算」 | 每轮或每会话对输入+输出 token 的上限；触发压缩或终止 |
| pass@1 | 「单次尝试通过率」 | 首次运行即解决、不重试也不偷看测试集的 SWE-bench 任务比例 |

## 延伸阅读

- [Claude Code 文档](https://docs.anthropic.com/en/docs/claude-code) —— Anthropic 的参考 harness
- [Cursor 3 更新日志](https://cursor.com/changelog) —— Agent Tabs 和 Composer 2 的产品说明
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) —— 用于 SWE-bench harness 对比的最小基线
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) —— 借助 Opus 4.5 在 SWE-bench Verified 上达到 79.2%
- [OpenCode](https://opencode.ai) —— 开源 harness，112k stars
- [SWE-bench Pro 排行榜](https://www.swebench.com) —— 本项目针对的评测
- [Model Context Protocol 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) —— StreamableHTTP、能力元数据
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— tool call 和 token 用量的 span schema
