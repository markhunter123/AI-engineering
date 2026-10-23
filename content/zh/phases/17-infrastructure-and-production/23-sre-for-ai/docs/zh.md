# 面向 AI 的 SRE —— 多智能体事件响应、运行手册与预测性检测

> AI SRE 使用 LLM，通过 RAG 把基础设施数据（日志、运行手册、服务拓扑）作为依据，自动化调查、文档和协调阶段。2026 年的架构模式是多智能体编排——由 supervisor（监督者）协调专用智能体（日志、指标、运行手册）；AI 提出假设和查询，人类批准判断性决策。Datadog Bits AI 和 Azure SRE Agent 以托管产品形式提供这套能力。运行手册正在演进：NeuBird Hawkeye 使用对抗式评估（两个模型分析同一事件；一致 = 置信，分歧 = 不确定）；运营记忆在团队更替中得以保留。自动修复保持谨慎：AI 建议，人类批准。完全自主行动的范围很窄（重启 pod、回滚特定部署），并配有严格防护栏——任何兜售「设置后就不用管」的人都在过度营销。新兴前沿：事件前预测。MIT 研究显示，一个在历史日志 + GPU 温度 + API 错误模式上训练的 LLM，提前 10-15 分钟预测了 89% 的故障。预测：到 2026 年底，95% 的企业 LLM 具备自动故障转移。

**Type:** Learn
**Languages:** Python (stdlib, toy multi-agent incident triage simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 24 (Chaos Engineering)
**Time:** ~60 minutes

## 学习目标

- 画出多智能体 AI SRE 架构：supervisor + 专用智能体（日志、指标、运行手册）+ 人类审批闸门。
- 解释为什么自动修复范围窄（重启 pod、还原部署），而不是宽泛（重构服务）。
- 说出对抗式评估模式（NeuBird Hawkeye）：两个模型一致 = 置信；不一致 = 升级。
- 引用 MIT 89% 提前检测结果及其运营约束：没有执行动作的预测只是仪表盘。

## 问题

凌晨 3 点，值班工程师被叫醒：「结账链路错误率过高。」

他们查 Datadog、Loki、三份运行手册和部署日志。30 分钟后才意识到根因是 KV cache 激增导致的 vLLM OOM。他们重启 pod，错误消失。

2026 年，这段调查的前 20 分钟已经可以自动化。按服务对日志分组、关联最近的部署、与运行手册匹配——这些都是 RAG + 工具使用。一个受监督的智能体可以在人类打开 Datadog 之前完成首轮研判，并给出假设。

全自主修复是另一个问题。重启 pod：安全。扩缩 GPU 池：策略允许则安全。重构服务：绝对不行。这门纪律的关键在于画出那条窄线。

## 概念

### 多智能体架构

```
          Incident
             │
             ▼
        Supervisor
        /    |    \
       ▼     ▼     ▼
  Log agent  Metric agent  Runbook agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        Hypothesis + evidence
             │
             ▼
        Human approval
             │
             ▼
        Action (narrow set)
```

Supervisor 把事件拆分为子查询。专用智能体拥有工具访问权限（日志搜索、PromQL、文档检索）。Supervisor 综合结果，把假设 + 证据呈现给人类。人类批准或改向。

### 自动修复的边界

**安全（窄）**：重启 pod、还原特定部署、在预批准范围内扩缩池、启用预批准的 feature flag。

**不安全（宽）**：更改服务拓扑、修改资源限制、部署新代码、更改 IAM、改动数据库。

任何兜售「设置后不用管」的人都在过度营销。随着 AI SRE 成熟，安全集合会扩大，但边界是真实存在的。

### 对抗式评估（NeuBird Hawkeye）

两个模型各自独立分析同一事件。若它们对根因判断一致，置信度高。若不一致，则升级给人类，并同时展示两种假设。模式简单，却是对抗幻觉根因的有效过滤器。

### 运营记忆

团队更替是传统 SRE 的隐形杀手——部落知识随之流失。AI SRE 把运行手册 + 事后复盘存进向量数据库；智能体在每次新事件中检索。新工程师加入时，AI 拥有完整历史。

### 事件前预测

MIT 2025 年的研究：一个在历史日志、GPU 温度和 API 错误模式上训练的 LLM，在测试集上提前 10-15 分钟预测了 89% 的故障。

现实检验：没有执行动作的预测只是仪表盘。运营问题是「当我们预测到问题时，我们做什么？」提前排空？叫醒值班？自动扩缩？答案取决于具体策略。

### 2026 年的产品

- **Datadog Bits AI** —— Datadog 内部的托管 SRE 副驾驶。
- **Azure SRE Agent** —— Azure 原生。
- **NeuBird Hawkeye** —— 对抗式评估 + 运营记忆。
- **PagerDuty AIOps** —— 研判 + 去重。
- **Incident.io Autopilot** —— 事件指挥官 + 协调。

### 运行手册即代码

运行手册从 Confluence 页面演进为带结构化章节的版本化 markdown（症状、假设、验证、行动）。结构化的运行手册能喂出更好的 RAG 检索。任何 AI SRE 落地都应从把非结构化运行手册转成结构化开始。

### 应该记住的数字

- MIT 提前检测：89% 的故障，提前 10-15 分钟。
- 多智能体研判：supervisor +（日志、指标、运行手册）+ 人类。
- 安全自动修复集合：重启 pod、还原部署、在边界内扩缩。
- 对抗式评估：两个模型独立；一致 = 置信。

```figure
i4-incident-agents
```

## 使用它

`code/main.py` 模拟一次多智能体研判：日志智能体发现错误，指标智能体发现 CPU 飙升，运行手册智能体匹配到已知问题。Supervisor 对假设排序。

## 交付它

本节课产出 `outputs/skill-ai-sre-plan.md`。给定当前值班方式、事件量和团队成熟度，设计一套 AI SRE 落地方案。

## 练习

1. 运行 `code/main.py`。如果日志智能体和指标智能体意见不一致会怎样？Supervisor 如何解决？
2. 为你的服务定义三个「安全」自动修复动作。逐一给出理由。
3. 写一个结构化运行手册模板：章节、必填字段、验证命令。
4. 预测性检测提前 12 分钟报警。你的策略是什么——叫醒值班、提前排空，还是两者都做？
5. 论证一个 3 人团队在 2026 年应该采用 AI SRE 还是观望。考虑成熟度、事件量和风险。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| AI SRE | 「值班智能体」 | 由 LLM 驱动的事件调查 + 协调 |
| Supervisor agent | 「编排者」 | 把事件拆成子查询的顶层智能体 |
| Specialized agent | 「领域智能体」 | 有工具访问权限的子智能体（日志、指标、运行手册） |
| Auto-remediation | 「AI 来修」 | 窄范围预批准动作；不是宽泛重构 |
| Operational memory | 「向量化运行手册」 | 事后复盘 + 运行手册存进向量数据库供 RAG |
| Adversarial eval | 「双模型校验」 | 独立分析；一致 = 置信 |
| NeuBird Hawkeye | 「对抗式那个」 | 采用对抗评估 + 记忆模式的产品 |
| Bits AI | 「Datadog 的 SRE 智能体」 | Datadog 托管的 AI SRE |
| Pre-incident prediction | 「提前检测」 | 故障预测提前 10-15 分钟 |

## 延伸阅读

- [incident.io — AI SRE Complete Guide 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — Human-Centred AI for SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — AI in SRE 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)
