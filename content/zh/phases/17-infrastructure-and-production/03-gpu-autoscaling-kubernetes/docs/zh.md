# Kubernetes 上的 GPU 自动扩缩容——Karpenter、KAI Scheduler、Gang Scheduling

> 三层，而不是一层。Karpenter 动态供给节点（不到一分钟，比 Cluster Autoscaler 快 40%）。KAI Scheduler 负责 gang scheduling、拓扑感知和层级队列——它阻止「8 缺 1」的部分分配陷阱：七个节点干等，并因缺一块 GPU 持续烧钱。应用层自动扩缩器（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）依据推理专用信号扩缩——队列深度、KV cache 利用率——而不是 CPU/DCGM 占空比。经典 HPA 陷阱在于 `DCGM_FI_DEV_GPU_UTIL` 是一种占空比度量：100% 可能是 10 个请求，也可能是 100 个。vLLM 会预分配 KV cache 内存，所以内存使用量永远不会触发缩容。本节课教你组合这三层，并避开 Karpenter 默认的 `WhenEmptyOrUnderutilized` 策略——它会在推理中途终止正在运行的 GPU 作业。

**Type:** Learn
**Languages:** Python (stdlib, toy queue-depth autoscaler simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 04 (Serving Engine Internals)
**Time:** ~75 minutes

## 学习目标

- 画出三层自动扩缩容（节点供给、gang scheduling、应用层），并说出每层使用的工具。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 对 vLLM 来说是错误的 HPA 信号，并说出两个替代信号（队列深度、KV cache 利用率）。
- 描述 gang scheduling 以及 KAI Scheduler 所阻止的部分分配故障模式（8 块 GPU 中 7 块空闲）。
- 说出会终止运行中 GPU 作业的 Karpenter 整合策略（`WhenEmptyOrUnderutilized`），并说明 2026 年的安全替代方案。

## 问题

你的团队在 Kubernetes 上交付了一个 LLM 推理服务。你用 `DCGM_FI_DEV_GPU_UTIL` 作为信号配置了 HPA。营业时间内，服务利用率固定在 100%。HPA 从不扩容——它已经认为你满了。你手动加了一个副本；TTFT 下降。HPA 仍然不扩缩。这个信号在骗你。

另一件事：你用 Cluster Autoscaler 管理节点。凌晨 2 点来了一个 100 万 token 的 prompt；集群花了 3 分钟供给节点，请求超时了。

还有一件事：你部署一个 70B 模型，需要跨 2 个节点的 8 块 GPU。集群有 7 块 GPU 空闲，另有 1 块分散在 3 个节点上。Cluster Autoscaler 为缺失的那 1 块 GPU 供给一个节点。七个节点等待 4 分钟、持续烧钱，直到 Kubernetes 把最后一块 GPU 拉起来。

三层，三种不同的故障模式。2026 年 GPU 感知的自动扩缩容不是「打开 HPA」。它要组合节点供给、gang scheduling 和应用信号驱动的扩缩。

## 概念

### 第 1 层——节点供给（Karpenter）

Karpenter 监视 pending 的 pod，并在约 45-60 秒内供给节点（对于 GPU 节点，Cluster Autoscaler 通常要 90-120 秒）。它依据 `NodePool` 约束动态选择实例类型——如果你的 pod 需要 8 块 H100 而集群没有匹配的节点，Karpenter 会直接供给一个，而不是扩缩现有的节点组。

**整合陷阱**：Karpenter 默认的 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU 池很危险。它会终止一台正在运行的 GPU 节点，把 pod 迁移到更便宜的合适实例上。对推理工作负载而言，这意味着驱逐运行中的请求，并在新节点上重新加载 70B 模型。损失是数分钟的容量加上请求失败。

GPU 池的安全设置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

这让 Karpenter 在一小时后整合真正为空的节点，但绝不会驱逐正在运行的作业。

### 第 2 层——gang scheduling（KAI Scheduler）

KAI Scheduler（项目最初叫「Karp」，后改名）处理默认 kube-scheduler 不处理的事情：

**Gang scheduling**——全有或全无地调度。一个需要 8 块 GPU 的分布式推理 pod，要么 8 块一起启动，要么都不启动。没有它，就会遇到部分分配陷阱：8 个 pod 启动了 7 个，它们无限等待，持续烧钱。

**拓扑感知**——知道哪些 GPU 共享 NVLink、哪些在同一机架、哪些之间是 InfiniBand。据此放置 pod。DeepSeek-V3 67B 的张量并行工作负载必须留在同一个 NVLink 域内；KAI Scheduler 会遵守这一点。

**层级队列**——多个团队竞争同一个 GPU 池，带优先级和配额。只有当优先级规则允许时，A 团队的生产作业才会被 B 团队的训练作业抢占。

KAI 作为次级调度器与 kube-scheduler 一起部署；你给工作负载加注解来使用它。Ray 和 vLLM production-stack 都已集成。

### 第 3 层——应用层信号

**HPA 陷阱**：`DCGM_FI_DEV_GPU_UTIL` 是占空比指标——它衡量每个采样区间内 GPU 是否在做工。100% 利用率可能意味着 10 个并发请求，也可能意味着 100 个；无论如何 GPU 都是忙的。按占空比扩缩等于盲扩。

更糟的是，vLLM 及类似引擎会预分配 KV cache 内存（上限由 `--gpu-memory-utilization` 决定）。即使只有一个请求，内存使用量也维持在 90% 附近。基于内存的 HPA 永远不会缩容。

**2026 年的替代信号**：

- 队列深度（等待 prefill 的请求数）。
- KV cache 利用率（分配给活跃序列的 block 占比）。
- 每副本 P99 TTFT（你的 SLA 信号）。
- Goodput（每秒满足所有 SLO 的请求数）。

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 消费这些信号并扩缩副本。对 LLM 服务而言，它们完全取代 HPA。

### 何时用什么

| 扩缩决策 | 工具 |
|----------------|------|
| 增加/移除节点 | Karpenter |
| 调度多 GPU 作业 | KAI Scheduler |
| 增加/移除副本 | Dynamo Planner / llm-d WVA（或基于队列深度的自定义 HPA） |
| 选择 GPU 类型 | Karpenter NodePool |
| 抢占低优先级 | KAI Scheduler 队列 |

### 分离式 prefill/decode 会让一切更复杂

如果你运行分离式 prefill/decode（Phase 17 · 17），就会有两类 pod，扩缩触发条件不同：prefill pod 按队列深度扩缩，decode pod 按 KV cache 压力扩缩。llm-d 把它们暴露为独立的 `Services`，每个角色各有自己的 HPA。不要试图用一个 HPA 同时管住两者。

### 冷启动在这里也很重要

冷启动缓解（Phase 17 · 10）正是节点供给时间变得用户可见的地方。Karpenter 的 45-60 秒预热，加上 20GB 模型加载，再加引擎初始化，意味着一个从零开始的请求要 2-5 分钟。对 SLO 关键路径保留一个热池（`min_workers=1`），或者在应用层使用 Modal 风格的 checkpointing。

### 应该记住的数字

- Karpenter 节点供给：约 45-60 秒，Cluster Autoscaler（GPU 节点）约 90-120 秒。
- KAI Scheduler 阻止部分分配浪费——「8 缺 1」陷阱。
- `DCGM_FI_DEV_GPU_UTIL` 作为 HPA 信号：失效；改用队列深度或 KV 利用率。
- Karpenter `WhenEmptyOrUnderutilized`：会终止运行中的 GPU 作业。推理场景用 `WhenEmpty + consolidateAfter: 1h`。

```figure
autoscaling
```

## 使用它

`code/main.py` 在一个突发性 GPU 工作负载上模拟三层自动扩缩器。比较朴素 HPA（占空比）、队列深度 HPA 和 KAI gang 调度扩缩。报告未满足的请求、GPU 空闲分钟数和一个综合评分。

## 交付它

本节课产出 `outputs/skill-gpu-autoscaler-plan.md`。给定集群拓扑、工作负载形态和 SLO，它设计一份三层自动扩缩容方案。

## 练习

1. 运行 `code/main.py`。在突发工作负载下，朴素的占空比 HPA 会丢掉多少队列深度 HPA 能接住的请求？差异从何而来？
2. 为一个在 H100 SXM5 上以 FP8 服务 Llama 3.3 70B 的集群设计 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及一个让非 GPU 工作负载避开这些节点的 taint。
3. 你的团队报告部署卡在 Pending，因为「GPU 明明有空，pod 却调度不上」。诊断——这是 Karpenter、kube-scheduler 还是 KAI Scheduler 的问题？哪些指标可以确认？
4. 为分离式 prefill pod 选一个扩缩信号，再为 decode pod 选一个不同的信号。为两者提供理由。
5. 计算 `WhenEmptyOrUnderutilized` 整合陷阱在一个 7x24 生产服务上的成本：该服务平均每天有 60 次请求被丢弃的事件，P99 TTFT > 10 秒。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Karpenter | 「节点供给器」 | Kubernetes 节点自动扩缩器；亚分钟级供给 |
| Cluster Autoscaler | 「旧版扩缩器」 | Kubernetes 节点自动扩缩器的前身；更慢，基于节点组 |
| KAI Scheduler | 「GPU 调度器」 | 用于 gang + 拓扑 + 队列的次级调度器 |
| Gang scheduling | 「全有或全无」 | 原子地调度 N 个 pod，否则全部推迟 |
| Topology awareness | 「机架感知」 | 依据 NVLink/IB/机架位置放置 pod |
| `DCGM_FI_DEV_GPU_UTIL` | 「GPU 利用率」 | 占空比指标；对 LLM 来说不是扩缩信号 |
| Queue depth | 「等待中的请求」 | prefill 受限扩缩的正确 HPA 信号 |
| KV cache utilization | 「内存压力」 | decode 受限扩缩的正确 HPA 信号 |
| Consolidation | 「Karpenter 整合」 | 为迁往更便宜的实例类型而终止节点 |
| `WhenEmpty + 1h` | 「安全整合」 | 不驱逐运行中 GPU 作业的策略 |

## 延伸阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler)——设计文档与配置示例。
- [Karpenter 中断控制](https://karpenter.sh/docs/concepts/disruption/)——整合策略语义与 GPU 安全默认值。
- [NVIDIA——Kubernetes 上的分离式 LLM 推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)——Dynamo Planner 的扩缩信号。
- [Ray 文档——RayCluster 的 KAI Scheduler](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html)——Ray 集成模式。
- [AWS EKS 计算与自动扩缩最佳实践](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html)——托管 Kubernetes 专属指南。
- [llm-d GitHub](https://github.com/llm-d/llm-d)——Workload Variant Autoscaler 设计。
