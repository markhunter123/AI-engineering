# 项目 10 —— 多 Agent 软件工程团队

> 2026 年多 Agent 工程团队的形态已经收敛：架构师做规划，N 个编码 Agent 在并行工作树里干活，一个评审者把关，一个测试者验证。SWE-AF 的工厂架构、MetaGPT 的基于角色的提示、AutoGen 0.4 的类型化 actor 图、Cognition 的 Devin，以及 Factory 的 Droids，都独立地落到了同一个形态上。并行工作树把墙钟时间转化为吞吐量。共享状态和交接协议则成为失效面。项目要搭建这支团队，在 SWE-bench Pro 上评估，并报告哪些交接会出问题、频率如何。

**Type:** Capstone
**Languages:** Python / TypeScript (agents), Shell (worktree scripts)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 16 (multi-agent), Phase 17 (infrastructure)
**Phases exercised:** P11 · P13 · P14 · P15 · P16 · P17
**Time:** 40 hours

## 问题

单 Agent 编码框架在大任务上会撞到天花板。原因不是单个 Agent 太弱，而是 200k token 的上下文装不下架构计划、四份并行的代码库切片、评审意见和测试输出。多 Agent 工厂把问题拆开：架构师拥有计划，编码者在并行工作树里各自实现，评审者把关，测试者验证。SWE-AF 的「工厂」架构、MetaGPT 的角色、AutoGen 的类型化 actor 图——这三种表述描述的是同一个形态。

失效面在交接上。架构师规划了编码者无法实现的东西。编码者产出互相冲突的 diff。评审者批准了一个幻觉出来的修复。测试者和还在写代码的编码者抢跑。你会搭建这样一支团队，在 50 个 SWE-bench Pro issue 上运行，追踪每一次交接，并公布复盘报告。

## 概念

角色就是类型化的 Agent。**架构师**（Claude Opus 4.7）阅读 issue，写出计划，并拆成带显式接口的子任务。**编码者**（Claude Sonnet 4.7，N 个并行实例，各自位于一个 `git worktree` + Daytona 沙箱中）独立实现子任务。**评审者**（GPT-5.4）阅读合并后的 diff，要么批准，要么要求具体修改。**测试者**（Gemini 2.5 Pro）在隔离环境中跑测试套件，并带着产物报告通过/失败。

沟通通过共享任务板（文件后端或 Redis）进行。每个角色消费自己有权限处理的任务。交接是 A2A 协议类型化的消息。协调关注点包括：合并冲突解决（协调者角色或自动三方合并）、共享状态同步（编码者一旦开工，计划就冻结；重新规划是独立事件），以及评审把关（评审者不能批准自己做出的改动或自己提出的改动）。

token 放大是隐藏成本。每个角色边界都会增加摘要提示和交接上下文。一次 40 轮的单 Agent 运行，在四个角色之间会变成总共 160 轮。评分标准特意对比 token 效率与单 Agent 基线，因为问题不是「多 Agent 能不能用」，而是「它在每美元上是否更划算」。

## 架构

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## 技术栈

- 编排：LangGraph，带共享状态 + 每个 Agent 一个子图
- 消息：A2A 协议（Google 2025），用于类型化的 Agent 间消息
- 模型：Opus 4.7（架构师）、Sonnet 4.7（编码者）、GPT-5.4（评审者）、Gemini 2.5 Pro（测试者）
- 工作树隔离：每个编码者一个 `git worktree add` + Daytona 沙箱
- 合并协调者：自定义三方合并 + LLM 调解的冲突解决
- 评估：SWE-bench Pro（50 个 issue）、SWE-AF 场景、HumanEval++ 用于单元测试
- 可观测性：Langfuse，带角色标记的 span，逐 Agent 的 token 记账
- 部署：K8s，每个角色一个独立 Deployment，按积压量做 HPA

```figure
ce-team-handoff
```

## 构建它

1. **任务板。** 文件后端的 JSONL，带类型化消息：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。Agent 按标签订阅。

2. **架构师。** 读 GitHub issue，用 Opus 4.7 跑一个计划模板，该模板要求显式的子任务接口（涉及哪些文件、公开函数、测试影响）。发出一个 `plan_request`，内含子任务 DAG。

3. **编码者。** N 个并行 worker，各自从任务板认领一个子任务。每个都新开一个 `git worktree add` 分支，外加一个 Daytona 沙箱。实现子任务。发出 `diff_ready`，附上 patch 和测试变化。

4. **合并协调者。** 所有编码者完成后，把 N 个分支三方合并进一个暂存分支。只有当文件级出现重叠时，才启用 LLM 调解的冲突解决。

5. **评审者。** GPT-5.4 读合并后的 diff。不能批准自己写的 diff。发出 `approved`（无操作）或 `review_feedback`，带具体修改要求，并路由回相关的编码者。

6. **测试者。** Gemini 2.5 Pro 在干净的沙箱里跑测试套件。捕获产物。发出 `test_passed` 或 `test_failed`，附堆栈痕迹。失败的测试会回环给拥有该失败子任务的编码者。

7. **交接记账。** 每一条跨越角色边界的消息都在 Langfuse 里有一个 span，记录载荷大小和所用模型。计算每个子任务的 token 放大比（coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens）。

8. **评估。** 在 50 个 SWE-bench Pro issue 上运行。对比 pass@1 和每解决一个 issue 的美元成本，与单 Agent 基线（一个 Sonnet 4.7 在单个工作树里）比较。

9. **复盘。** 对每个失败的 issue，找出断掉的交接（计划太含糊、合并冲突、评审者误批、测试者抖动）。产出一张交接失败直方图。

## 使用它

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## 交付它

`outputs/skill-multi-agent-team.md` 是交付物。给定一个 issue URL 和并行度，团队产出一个可合并的 PR，并附上逐角色的 token 记账。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 匹配的 50 issue 子集，pass@1 |
| 20 | 并行加速比 | 墙钟时间对比单 Agent 基线 |
| 20 | 评审质量 | 注入缺陷探针上的误批率 |
| 20 | token 效率 | 每个已解决 issue 的总 token 对比单 Agent |
| 15 | 协调工程 | 合并冲突解决、交接失败直方图 |
| **100** | | |

## 练习

1. 在运行中途往 diff 里注入一个明显的 bug（在主逻辑前多一句 `return None`）。测量评审者的误批率。调评审者提示，直到误批率低于 5%。

2. 缩减到两个编码者（架构师 + 编码者 + 评审者 + 测试者，编码者顺序跑两个子任务）。对比墙钟时间和通过率。

3. 用单写者约束替代合并协调者（子任务触及互不相交的文件集）。测量架构师承担的规划负担。

4. 把评审者从 GPT-5.4 换成 Claude Opus 4.7。测量误批率和 token 成本差值。

5. 加第五个角色：文档员（Haiku 4.5）。评审之后，它产出一条 changelog。测量文档质量是否值得那份额外的 token 开销。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Parallel worktree | 「隔离分支」 | `git worktree add` 为每个编码者产出全新工作树 |
| Task board | 「共享消息总线」 | 文件或 Redis 存储的类型化消息，Agent 订阅 |
| Handoff | 「角色边界」 | 任何从一个角色上下文跨到另一个角色上下文的消息 |
| Token amplification | 「多 Agent 开销」 | 同一任务下所有角色消耗的总 token / 单 Agent token |
| A2A protocol | 「Agent 到 Agent」 | Google 2025 年的规范，定义类型化的 Agent 间消息 |
| Merge coordinator | 「集成者」 | 执行三方合并并调解冲突的组件 |
| False approval | 「评审幻觉」 | 评审者批准了含已知缺陷的 diff |

## 延伸阅读

- [SWE-AF 工厂架构](https://github.com/Agent-Field/SWE-AF) —— 2026 年多 Agent 工厂的参考
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) —— 基于角色的多 Agent 框架
- [AutoGen v0.4](https://github.com/microsoft/autogen) —— 微软的类型化 actor 框架
- [Cognition AI（Devin）](https://cognition.ai) —— 参考产品
- [Factory Droids](https://www.factory.ai) —— 另一个参考产品
- [Google A2A 协议](https://a2a-protocol.org/latest/) —— Agent 间消息规范
- [git worktree 文档](https://git-scm.com/docs/git-worktree) —— 隔离底座
- [SWE-bench Pro](https://www.swebench.com) —— 评估目标
