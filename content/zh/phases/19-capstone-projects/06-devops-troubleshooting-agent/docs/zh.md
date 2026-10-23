# 项目 06 — Kubernetes 上的 DevOps 故障排查 Agent

> AWS 的 DevOps Agent 正式发布（GA），Resolve AI 公开了其 K8s playbook，NeuBird 演示了语义监控，Metoro 把 AI SRE 与逐服务 SLO 绑定。生产形态已经确定：告警 webhook 触发，Agent 读取遥测，遍历 K8s 对象图，对根因假设排序，并向 Slack 发送带审批按钮的简报。默认只读。每一次修复都由人工把关。这个项目就是这样一个 Agent，在 20 个合成事故上评测，并在三个共享案例上与 AWS 的 Agent 对比。

**Type:** Capstone
**Languages:** Python (agent), TypeScript (Slack integration)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools and MCP), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P11 · P13 · P14 · P15 · P17 · P18
**Time:** 30 hours

## 问题

2025–2026 年的 SRE 叙事变成了：「AI Agent 分诊事故，人类批准修复。」AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 都在生产环境交付这一形态。Agent 读取 Prometheus 指标、Loki 日志、Tempo trace、kube-state-metrics，以及一张 K8s 对象知识图谱。它在五分钟内产出一份带遥测引用的排好序的根因假设，并且绝不在没有通过 Slack 获得明确人工批准的情况下执行破坏性命令。

大部分硬功夫在范围界定和安全，而不在推理。Agent 需要默认只读的 RBAC 面、加固的 MCP 工具服务器，以及记录每条「考虑过」与「实际执行」命令的审计日志。它需要知道自己何时力有不逮并升级（escalate）。它还得跑得足够便宜，避免 OOM-kill 级联产生 5000 美元的 Agent 账单。

## 概念

Agent 在一张知识图谱上工作。节点是 K8s 对象（Pod、Deployment、Service、Node、HPA、PVC）以及遥测来源（Prometheus series、Loki stream、Tempo trace）。边编码归属关系（Pod -> ReplicaSet -> Deployment）、调度关系（Pod -> Node）和观测关系（Pod -> Prometheus series）。图谱通过 kube-state-metrics 同步保持新鲜，并在每次告警时重新采样。

告警触发时，Agent 从受影响的对象出发做根因分析。它沿边遍历，拉取相关的遥测切片（最近 15 分钟），并起草假设。假设按证据排序：有多少遥测引用支撑、有多新、有多具体。前 3 个假设会连同图路径可视化和修复操作的审批按钮一起发到 Slack。

修复是受门禁控制的。默认允许的操作都是只读的。破坏性操作（缩容、回滚、删除 Pod）需要 Slack 批准；ArgoCD 回滚 hook 所需的 auth token Agent 永远不持有。审计日志记录 Agent 考虑过的每一条命令 —— 不只是执行过的 —— 这样评审流程才能抓住险些酿成事故的操作。

## 架构

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## 技术栈

- 可观测性来源：Prometheus、Loki、Tempo、kube-state-metrics
- 知识图谱：K8s 对象 + 遥测边的 Neo4j（托管）或 kuzu（嵌入式）
- Agent：带逐工具 allow-list 的 LangGraph，默认只读
- 工具传输：FastMCP over StreamableHTTP；破坏性工具放在审批门禁之后的独立服务器
- 模型：根因推理用 Claude Sonnet 4.7，日志摘要用 Gemini 2.5 Flash
- 修复：ArgoCD 回滚 webhook、PagerDuty 升级、Slack 审批卡片
- 审计：仅追加的结构化日志（considered、executed、approved、outcome）
- 部署：自身只有狭窄 RBAC 角色的 K8s deployment；独立 namespace

```figure
ce-rootcause-walk
```

## 构建它

1. **图谱摄入。** 每 30 秒把 kube-state-metrics 同步进 Neo4j/kuzu。节点：Pod、Deployment、Node、Service、PVC、HPA。边：OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。遥测叠加边：OBSERVED_BY（一个 Pod 被某个 Prometheus series 观测）。

2. **告警接收器。** 一个接受 PagerDuty 或 Alertmanager webhook 的 FastAPI 端点。抽取受影响的对象和 SLO 违约信息。

3. **只读工具面。** 通过 FastMCP 封装 kubectl、Prometheus query、Loki logql、Tempo traceql。每个工具只对应狭窄的 RBAC 动词（「get」「list」「describe」）。默认服务器里没有「delete」「exec」「scale」。

4. **根因 Agent。** 一个含三个节点的 LangGraph：`sample` 拉取最近 15 分钟的遥测切片，`walk` 查询图谱中的邻近对象，`hypothesize` 起草带遥测引用的排好序的根因候选。

5. **证据打分。** 每个假设的分数 = 新鲜度 * 具体度 * 图路径长度的倒数 * 引用数量。返回前 3 个。

6. **Slack 简报。** 发布一条附件消息，包含假设、图路径可视化（服务端渲染的子图图像），以及至多一个修复操作的审批按钮。

7. **修复门禁。** 破坏性工具（缩容、回滚、删除）位于第二个 MCP 服务器上，需要审批 token 才能访问。只有在 Slack 卡片获得人工批准后，Agent 才能调用它们。

8. **审计日志。** 仅追加的 JSONL：对每条候选命令，记录是否被考虑、是否被执行、由谁批准。每天发送到 S3。

9. **合成事故套件。** 构建 20 个场景：OOMKill 级联、DNS 抖动、HPA 反复伸缩、PVC 写满、吵闹邻居、故障 sidecar、错误的 ConfigMap 发布、证书轮换、镜像拉取退避等。按根因准确率和假设耗时给 Agent 打分。

## 使用它

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## 交付它

交付物是 `outputs/skill-devops-agent.md`。给定一个 K8s 集群和告警源，该 Agent 产出排好序的根因假设和一套由 Slack 门禁控制的修复流程。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 场景套件上的 RCA 准确率 | 在 20 个合成事故中，根因正确率 ≥80% |
| 20 | 安全性 | 审计日志显示，任何破坏性操作都未曾在缺少 Slack 批准的情况下执行 |
| 20 | 假设耗时 | 从告警到 Slack 简报的 p50 低于 5 分钟 |
| 20 | 可解释性 | 每个假设都有图路径和遥测引用 |
| 15 | 集成完备性 | PagerDuty、Slack、ArgoCD、Prometheus 端到端可用 |
| **100** | | |

## 练习

1. 在 AWS DevOps Agent 演示所用的同样三个事故上运行你的 Agent。发布并排对比。报告 Agent 在哪些地方出现分歧。

2. 增加「险肇」审计，标记 Agent 考虑过、若未经批准就会造成破坏的任何命令。测量一周内的险肇率。

3. 把假设模型从 Claude Sonnet 4.7 换成自托管的 Llama 3.3 70B。测量 RCA 准确率差值和每事故美元成本。

4. 构建因果过滤器：把相关的遥测尖峰与真正的根因区分开。用 20 个场景的标签训练一个小型分类器。

5. 增加回滚演练：用相同 manifest 对 staging 集群执行 ArgoCD 回滚。在 Slack 审批按钮出现之前，先在真实集群中验证回滚方案。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| K8s 知识图谱 | 「集群图」 | 节点 = K8s 对象 + 遥测 series；边 = 归属、调度、观测 |
| 默认只读 | 「限定范围的 RBAC」 | Agent 的 service account 只有 get/list/describe 动词；破坏性动词位于需审批的独立服务器 |
| 审计日志 | 「考虑过 vs 已执行」 | 对每条候选命令的仅追加记录：是否运行、由谁批准 |
| 假设排序 | 「证据分」 | 新鲜度 × 具体度 × 图路径长度倒数 × 引用数量 |
| Slack 审批卡片 | 「HITL 门禁」 | 带修复按钮的交互式 Slack 消息；在人工点击前 Agent 无法继续 |
| 遥测引用 | 「证据指针」 | 支撑某条论断的 Prometheus 查询、Loki 选择器或 Tempo trace URL |
| MTTR | 「解决耗时」 | 从告警触发到 SLO 恢复的实际耗时 |

## 延伸阅读

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) —— 2026 年的权威参考
- [Resolve AI K8s 故障排查](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) —— 竞品参考
- [NeuBird 语义监控](https://www.neubird.ai) —— 语义图谱路线
- [Metoro AI SRE](https://metoro.io) —— SLO 优先的生产视角
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) —— 集群状态来源
- [LangGraph](https://langchain-ai.github.io/langgraph/) —— 参考 Agent 编排器
- [FastMCP](https://github.com/jlowin/fastmcp) —— Python MCP 服务器框架
- [ArgoCD 回滚](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) —— 受门禁控制的修复目标
