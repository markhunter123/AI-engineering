# Serverless LLM 的冷启动缓解

> 一个 20 GB 的模型镜像从冷态到开始服务需要 5-10 分钟（7B）到 20 分钟以上（70B）。在真正的 serverless 世界里，这不是热身——这是故障。缓解手段在五个层面展开：预热节点镜像（AWS 上的 Bottlerocket、双卷架构）、模型流式加载（NVIDIA Run:ai Model Streamer，vLLM 原生支持）、GPU 内存快照（Modal checkpoint，重启最快提升 10 倍）、热池（`min_workers=1`）、分层加载（ServerlessLLM 的 NVMe→DRAM→HBM pipeline，延迟降低 10-200 倍），以及迁移输入 token（KB）而非 KV cache（GB）的在线迁移。Modal 公布 2-4 秒冷启动为下限；Baseten 默认 5-10 秒，预热后可小于 1 秒。本课教你测量、预算并叠加这五个层面。

**Type:** Learn
**Languages:** Python (stdlib, toy cold-start path simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~60 minutes

## 学习目标

- 列举冷启动缓解的五个层面，并在每一层说出一个工具或模式。
- 对 70B 模型，把总冷启动时间计算为（节点供给）+（权重下载）+（权重载入 HBM）+（引擎初始化）之和。
- 解释为什么在线迁移传输的是输入 token（KB）而不是 KV cache（GB），以及代价是什么（重计算）。
- 说出热池的权衡（为空闲 GPU 付费，或接受冷启动长尾），以及在什么 SLA 阈值下 `min_workers > 0` 成为强制要求。

## 问题

你的 serverless LLM 端点在夜间缩容到零。早上 8 点流量飙升。第一个请求要等待：

1. Karpenter 供给一个 GPU 节点：45-60 秒。
2. 容器拉取带权重的 30 GB 镜像：120-300 秒。
3. 引擎把权重载入 HBM：视模型规模和存储速度，45-120 秒。
4. vLLM 或 TRT-LLM 初始化 CUDA graph、KV cache 池、tokenizer：10-30 秒。

总计：在返回第一个 token 之前需要 220-510 秒（大约 3-8 分钟）。你的 SLA 是 2 秒。你上了热池（`min_workers=1`），问题似乎消失了——但你现在要为一块空闲 GPU 7×24 付费。如果你的服务有 5 个产品，每个各留一个热副本，那就是每月 5 × 24 × 30 = 3,600 GPU 小时，无论有没有用户调用。

冷启动缓解就是要在保持 serverless 经济性的同时，逼近常开服务的延迟。

## 概念

### 第 1 层——预热节点镜像（Bottlerocket）

在 AWS 上，Bottlerocket 的双卷架构将操作系统与数据分离。为数据卷创建快照，并把容器镜像预先拉取进去；在 `EC2NodeClass` 中引用该快照 ID。新节点启动时权重已经在本地 NVMe 上——步骤 2 以及步骤 3 的一部分随之消失。它原生配合 Karpenter。典型节省：大模型每次冷启动省 2-4 分钟。

GCP 上的等价做法：预烘焙容器层的自定义 VM 镜像。Azure 上：使用同一模式的托管磁盘快照。

### 第 2 层——模型流式加载（Run:ai Model Streamer）

与其在回答第一个请求前加载完整文件，不如把权重逐层流式载入 GPU 内存，并在第一个 transformer block 常驻后立即开始处理。NVIDIA Run:ai Model Streamer 在 2026 年已原生集成于 vLLM。它支持 S3、GCS 和本地 NVMe。通过把 I/O 与计算初始化重叠，大模型的权重加载时间大致减半。

### 第 3 层——GPU 内存快照（Modal）

Modal 在首次加载后对 GPU 状态（权重、CUDA graph、KV cache 区域）取 checkpoint。后续重启直接反序列化进 HBM——比重新初始化快 10 倍。这是最接近「2 秒内启动一块热的 GPU」的东西。权衡：快照与 GPU 拓扑绑定，如果 Karpenter 把你迁到不同的 SKU，就需要重新做 checkpoint。

### 第 4 层——热池（min_workers=1）

最简单的缓解：始终保持一个副本就绪。成本是一块 GPU 的时租 7×24。这笔账对小模型很残酷（为避免 30 秒冷启动而支付每小时 0.85-1.50 美元），对大模型则友好（为避开 5 分钟冷启动而支付每小时 4 美元）。热池成为强制要求的 SLA 阈值：通常在 70B+ 模型上为 TTFT P99 < 60 秒。

### 第 5 层——分层加载（ServerlessLLM）

ServerlessLLM 把存储视为层级结构：NVMe（快但容量大）、DRAM（中等但分层）、HBM（小但即时）。权重预加载到 DRAM，再按需载入 HBM。论文报告冷加载相对朴素的磁盘到 HBM 路径延迟降低 10-200 倍。生产采用尚早，但已有与 vLLM 的集成。

### 第 6 层——在线迁移（附加模式）

当节点不可用时（spot 抢占、节点排空），传统做法是冷启动另一个副本并排干请求队列。在线迁移则把输入 token（千字节级）转移到已加载模型的目标节点，并在目标节点上重算 KV cache。重计算比在网络上传输 GB 级 KV cache 便宜得多。适用于分离式部署。

### 热池的算术

对于一个 TTFT P99 SLA 为 2 秒的服务，问题不是「要不要热池」，而是「要几个热副本，哪些路径得到它们」。

- 高价值交互路径（实时聊天、语音 Agent）：`min_workers=1-2`。
- 后台批处理路径（夜间分类）：可以接受缩容到零，5-10 分钟冷启动可容忍。
- 高级层：按租户配置 `min_workers`，并分配专属容量。

### 先测量，再优化

70B 模型在新节点上的冷启动解剖（示意）：

| 阶段 | 时间 | 缓解 |
|-------|------|-----------|
| 节点供给 | 50s | Bottlerocket + 预热镜像、热池 |
| 镜像拉取 | 180s | 预热数据卷（消除） |
| 权重载入 HBM | 75s | 模型 streamer（减半）；GPU 快照（消除） |
| 引擎初始化 | 20s | 持久化 CUDA graph 缓存 |
| 首次前向 | 3s | 固有延迟下限 |
| **冷启动总计** | **328s** | |
| **缓解后总计** | **~15s** | 22 倍降低 |

### 应该记住的数字

- Modal 冷启动：2-4 秒（配合 GPU 快照）。
- Baseten 默认冷启动：5-10 秒；预热后可小于 1 秒。
- 70B 原始冷启动：3-8 分钟。
- Run:ai Model Streamer：权重加载约 2 倍加速。
- ServerlessLLM 分层加载：延迟降低 10-200 倍（论文数字）。

```figure
cold-start-pipeline
```

## 使用它

`code/main.py` 对带与不带各项缓解的冷启动路径建模。它报告总冷启动时间、热池成本，以及热池自身盈亏平衡所需的请求率。

## 交付它

本课产出 `outputs/skill-cold-start-planner.md`。给定 SLA、模型规模和流量形态，它选择要叠加哪些缓解手段。

## 练习

1. 运行 `code/main.py`。计算热副本比在 SLO 处通过额外丢弃请求来承受冷启动税更便宜的盈亏平衡请求率。
2. 你要部署一个 13B 模型，P99 TTFT SLA 为 3 秒。选出能达成目标的最小缓解组合（层数最少）。
3. Bottlerocket 预置消除了镜像拉取，但权重仍需从快照载入 HBM。如果快照支撑的 NVMe 读取速度为 7 GB/s，计算 70B 模型的墙钟时间。
4. 你的 serverless 提供商提供 GPU 快照（Modal），但团队以「快照会泄漏 PII」为由拒绝。论证双方观点——现实风险是什么，缓解手段是什么（临时快照、加密、命名空间隔离）？
5. 设计一套分层热池策略：付费用户、试用用户和批处理工作负载各需要几个热副本？给出计算过程。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 冷启动 | 「大停顿」 | 新副本上从请求到首个 token 的时间 |
| 热池 | 「常开下限」 | `min_workers >= 1`，保持至少一个副本就绪 |
| 预热镜像 | 「烘焙 AMI」 | 容器权重已预置在节点镜像中 |
| Bottlerocket | 「AWS 节点 OS」 | AWS 面向容器的 OS，支持双卷快照 |
| Model streamer | 「流式加载」 | 让权重 I/O 与计算初始化重叠 |
| GPU 快照 | 「checkpoint 到 HBM」 | 序列化加载后的 GPU 状态；重启时反序列化 |
| 分层加载 | 「NVMe + DRAM + HBM」 | 存储层级；按需加载 |
| 在线迁移 | 「搬运 token」 | 传输输入（KB），在目标端重算 KV |
| `min_workers` | 「热副本」 | serverless 的最小保活数量 |
| 缩容到零 | 「完全 serverless」 | 空闲时零成本；接受完整冷启动税 |

## 延伸阅读

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start)——Modal 公布的基准和 checkpoint 架构。
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket)——预热数据卷快照模式。
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer)——让权重加载与计算初始化重叠。
- [Baseten — Cold starts](https://docs.baseten.co/deployment/autoscaling/cold-starts)——预热操作手册。
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu)——分层加载设计。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)——分离式部署的在线迁移。
