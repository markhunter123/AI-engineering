# 项目 05 — 自主研究智能体（AI-Scientist 级别）

> Sakana 的 AI-Scientist-v2 发表了完整论文。Agent Laboratory 跑通了实验。Allen AI 公开了 trace。2026 年的形态是：对实验做计划—执行—验证的树搜索、带预算的成本、沙箱化的代码执行、带视觉反馈的 LaTeX 写作器，以及自动化的 NeurIPS 风格评审团。这个项目要求你构建一个，把每篇论文的成本控制在 30 美元内端到端跑通，并通过 Sakana 记录的沙箱逃逸红队测试。

**Type:** Capstone
**Languages:** Python (agent + sandbox), LaTeX (output)
**Prerequisites:** Phase 2 (ML), Phase 3 (deep learning), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 14 (agents), Phase 15 (autonomous), Phase 16 (multi-agent), Phase 18 (safety)
**Phases exercised:** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**Time:** 40 hours

## 问题

2026 年，自主研究智能体跨过了一道门槛。Sakana AI 的 AI-Scientist-v2 登上 Nature，其生成的论文通过了 workshop 同行评审。ShinkaEvolve（ICLR 2026）把这条路线扩展到可进化的假设。AMD 的 Agent Laboratory 发布了可复现的 trace。这些智能体并不神奇 —— 它们是运行在候选实验树上的计划—执行—验证循环，配有成本上限、种子绑定的沙箱和自动评审。功夫都在循环、预算和安全方案里。

你要通过在一个狭窄领域里围绕一个种子想法实现这套循环来学习它（例如，在 1 亿参数的 Transformer 上做注意力稀疏性消融）。价值不在于第一次运行就发现新东西，而在于基础设施：树搜索、实验沙箱、写作—评审循环、红队报告。Sakana 团队记录了沙箱逃逸失败；你的智能体必须通过同样的红队测试。

## 概念

这个智能体是一个最佳优先树搜索。节点是实验规格：(假设、配置、代码、预期结果)。expand 步骤通过小幅修改提出子节点（换优化器、调整 batch size、消融某个组件）。每个子节点都在全新的沙箱中运行，并受硬性资源上限约束。结果回馈给评分函数，按（新颖性 × 质量 × 剩余预算）给节点排序。树一直生长到预算耗尽，然后把最佳分支写成论文。

写作器是多模态的。它生成 LaTeX 草稿，编译，渲染图表，再把渲染后的 PDF 送回 Claude Opus 4.7 的视觉模式，就版式、图表可读性和论断—证据对齐给出批评。由五位 LLM 评审组成的评审团给出 NeurIPS 风格评分（新颖性、严谨性、清晰度、可复现性、影响力）；如果平均分低于阈值，论文就带着批评意见退回给写作器。

安全是承重结构。每个实验都在 E2B 或 Daytona 沙箱中运行，无网络出口、有墙钟时间上限、资源限制固定。智能体的代码生成步骤会经过一个策略层，拦截可能逃逸沙箱的系统调用。红队报告复现 Sakana 记录的攻面（fork 炸弹、文件系统逃逸、LLM 编写的网络调用）。

## 架构

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## 技术栈

- 编排：带 checkpoint 和人工审批门禁的 LangGraph
- 树搜索：在实验节点上自研的最佳优先搜索（Sakana v2 的 AB-MCTS 风格）
- 沙箱：每个实验一个 E2B，备选 Docker-in-Docker；通过 cgroups 限制资源
- 文献：Semantic Scholar Graph API + OpenAlex + 本地摘要 FAISS 缓存
- 写作器：LaTeX 模板 + Claude Opus 4.7（视觉模式），负责图表批评和版式
- 评审：5 位评审组成的评审团（Opus 4.7、GPT-5.4、Gemini 3 Pro、DeepSeek R1、Qwen3-Max），加权聚合
- 实验框架：真实实验用 PyTorch 2.5，日志用 W&B
- 可观测性：智能体 trace 用 Langfuse，每篇论文硬预算 30 美元

```figure
ce-experiment-tree
```

## 构建它

1. **种子与领域划定。** 取一个种子想法（例如「研究 10 亿参数以下 Transformer 注意力图中的稀疏模式」）。定义搜索空间：模型、数据集、算力预算。

2. **文献扫描。** 向 Semantic Scholar + OpenAlex 查询 50 篇被引最多的相关论文；在本地缓存摘要；生成一页领域摘要。

3. **树骨架。** 用种子假设初始化根节点。实现 `expand(node) -> children`，用小幅修改提案（每个子节点只改一项配置）。把 `score(node)` 实现为新颖性 × 质量 × 预算的加权项。

4. **沙箱封装。** 每个实验都运行 `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only`（或等效的 E2B 策略）。种子写入沙箱；输出以只读方式挂载取出。

5. **计划—执行—验证循环。** `plan` 提出子节点。`execute` 运行沙箱，捕获日志和指标。`verify` 对指标运行单元检查（损失下降了吗？消融是否隔离出了效应？）。失败的节点会在树上记录失败原因。

6. **写作器。** 预算耗尽后，选出最佳分支。用 matplotlib 渲染图表。把分支 trace 放进上下文，通过 Claude Opus 4.7 生成 LaTeX 草稿。编译。把编译后的 PDF 送回 Opus 4.7 视觉模式做批评。迭代。

7. **评审团。** 五位评审按 NeurIPS 风格评分标准，从（新颖性、严谨性、清晰度、可复现性、影响力）五个维度给草稿打分。如果平均分低于 4.0/5，就带着批评意见退回写作器。重写 3 次后硬性停止。

8. **红队测试。** 构建或集成一组针对沙箱的对抗性任务：fork 炸弹、网络外泄尝试、文件系统逃逸、LLM 编写的 shell 元字符。确认全部被拦截。撰写发现报告。

9. **可复现性。** 每篇论文都附带树搜索 trace JSON、随机种子、W&B run 链接、沙箱配置，以及一份端到端复现的 README。

## 使用它

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## 交付它

交付物是 `outputs/skill-ai-scientist.md`。给定一个种子想法 + 一个领域 + 30 美元预算，它运行完整 pipeline，产出一篇可评审的论文以及一个可复现性包。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 论文质量 | 与已发表的 workshop 论文做盲评对照 |
| 20 | 实验严谨性 | 基线、种子、消融；每条论断都有结果表中的一个单元格支撑 |
| 20 | 成本与算力纪律 | 执行每篇 30 美元的上限，并有 Langfuse 追踪 |
| 20 | 安全性 | 通过沙箱红队测试；验证网络策略和 kill switch |
| 15 | 可复现性 | 用相同种子一条命令重跑即可复现论文 |
| **100** | | |

## 练习

1. 在同一领域内用三个不同的种子想法运行 pipeline。比较树搜索的哪些部分重叠。找出重复浪费的算力。

2. 对预估成本超过 5 美元的节点，在实验执行前加入人工审批门禁。测量总成本下降多少。

3. 把评审团换成单一评审。在由已知劣质论文组成的留出集上测量误接受率。

4. 引入一个网络外泄红队测试：让智能体编写尝试 `curl` 外部地址的代码。确认 `--network=none` 策略将其拦截。记录这次尝试。

5. 把你的树搜索与平坦随机基线（相同预算、没有扩展策略）对比。报告新颖性 × 质量的提升。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 树搜索 | 「AB-MCTS 式扩展」 | 以新颖性×质量×预算为评分，在实验节点上做最佳优先探索 |
| 沙箱 | 「实验隔离」 | 无网络、CPU/内存受限、种子固定、输入只读的容器 |
| 视觉批评 | 「先渲染再阅读」 | 把论文编译成 PDF，再把 PDF 送回 VLM，就版式和论断—证据一致性提出批评 |
| 评审团 | 「自动同行评审」 | 多位 LLM 评审按 NeurIPS 评分标准打分；加权聚合结果作为 pipeline 的门禁 |
| 新颖性评分 | 「这新吗？」 | 对与 50 篇文献缓存接近程度进行惩罚的启发式评分 |
| 成本上限 | 「美元预算」 | 每篇论文总花费的硬性上限；Langfuse 计数器 + 运行前预估 |
| 红队测试 | 「沙箱逃逸审计」 | 如果策略有误就会逃逸沙箱的对抗性任务 |

## 延伸阅读

- [Sakana AI-Scientist-v2 仓库](https://github.com/SakanaAI/AI-Scientist-v2) —— 生产级研究智能体的参考
- [Sakana AI-Scientist-v1 论文（arXiv:2408.06292）](https://arxiv.org/abs/2408.06292) —— 原始方法论
- [ShinkaEvolve（Sakana，ICLR 2026）](https://sakana.ai) —— 进化式扩展
- [Agent Laboratory（AMD）](https://github.com/SamuelSchmidgall/AgentLaboratory) —— 多角色研究实验室框架
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) —— 参考编排层
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) —— 文献搜索
- [E2B 沙箱](https://e2b.dev) —— 实验隔离参考
- [NeurIPS 评审指南](https://neurips.cc/Conferences/2026/ReviewerGuidelines) —— 评审团所编码的评分标准
