# 分离式 Prefill/Decode——NVIDIA Dynamo 与 llm-d

> Prefill 受算力约束，decode 受内存带宽约束。把两者放在同一块 GPU 上，总有一种资源被浪费。分离式架构把它们拆到不同的池子上，并通过 NIXL（RDMA/InfiniBand，回退到 TCP）在池间传输 KV cache。NVIDIA Dynamo（GTC 2025 发布，1.0 GA）位于 vLLM/SGLang/TRT-LLM 之上——它的 Planner Profiler + SLA Planner 会自动调整 prefill:decode 比例以满足 SLO。NVIDIA 公布过这个量级的吞吐提升——developer.nvidia.com（2025-06）显示，在中等延迟区间，DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo 上有约 6 倍提升；Dynamo 产品页（developer.nvidia.com，未标注日期）宣称在 GB300 NVL72 + Dynamo 上，MoE 吞吐可达 Hopper 的 50 倍。「30 倍」这个数字是社区对 Blackwell + Dynamo + DeepSeek-R1 全栈报告的汇总；我们没有找到任何单一原始来源明确说 30 倍，因此应把它当作方向性说法。llm-d（Red Hat + AWS）是 Kubernetes 原生的：prefill / decode / router 各自是独立的 Service，按角色配置 HPA。llm-d 0.5 增加了分层 KV offload、缓存感知的 LoRA 路由、UCCL 网络、缩容到零。经济性：对多份客户披露的内部汇总显示，从同址部署（colocated）切换到使用 Dynamo 的分离式部署、并保持 SLA 不变时，$2M 级别的推理支出可节省 30–40%（即每年 $600-800K）；具体的 $2M→$600-800K 数字是内部合成值，不是某一篇发表的案例研究——请把它当作数量级锚点，而不是引用来源。短 prompt（<512 token，短输出）不值得支付传输成本。

**Type:** Learn
**Languages:** Python (stdlib, toy disaggregated-vs-colocated simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 08 (Inference Metrics)
**Time:** ~75 minutes

## 学习目标

- 解释为什么 prefill 和 decode 的最优 GPU 分配不同，并量化同址部署下的浪费。
- 画出分离式架构图：prefill 池、decode 池、通过 NIXL 的 KV 传输、router。
- 说出分离式架构不划算的条件（短 prompt、短输出）。
- 区分 NVIDIA Dynamo（栈上层）与 llm-d（Kubernetes 原生），并为各自匹配运营场景。

## 问题

你在 8 块 H100 上运行 Llama 3.3 70B。在混合负载下（长 prompt + 短输出），GPU 在 decode 期间空闲，因为大部分算力花在了 prefill 上。换一种负载（短 prompt + 长输出），情况正好相反。prefill + decode 同址部署意味着你两者都要过度配置。

预算影响：20-40% 的 GPU 时间浪费在了错误的资源上。你花钱买 H100 算力去跑受内存带宽约束的 decode，或者买 H100 的 HBM 带宽去跑受算力约束的 prefill。两者都是昂贵的浪费。

分离式架构把 prefill 和 decode 拆到各自按瓶颈定容的池子。KV cache 通过高带宽互连从 prefill 池传输到 decode 池。

## 概念

### 为什么瓶颈不同

**Prefill**——对完整输入 prompt 做一次前向计算。矩阵乘法占主导；受算力约束。H100 FP8 可提供约 2000 TFLOPS 的有效吞吐。批处理效率好——一次前向处理许多 token。

**Decode**——一次生成一个 token，每次迭代都要读取全部权重。受内存带宽约束。HBM3 提供约 3 TB/s。只有在高并发下批处理效率才好——权重读取能摊销到整个 batch 上。

把它们放在一起：你买的 GPU 要同时对两者都优化。H100 两者都不错，但成本相同。规模化之后，你会想让 prefill 池用 H100 / 偏算力，decode 池用 H200 / 偏内存，或者配合激进量化。

### 架构

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL 是 NVIDIA 的节点间传输层。有 RDMA/InfiniBand 时使用它，否则回退到 TCP。传输延迟是真实存在的——70B FP8 上，4K token prompt 的 KV cache 通常需要 20-80 ms。这就是短 prompt 不值得分离的原因：传输税超过了节省。

### Dynamo vs llm-d

**NVIDIA Dynamo**（GTC 2025 发布，1.0 GA）：
- 作为编排器位于 vLLM、SGLang、TRT-LLM 之上。
- Planner Profiler 测量负载，SLA Planner 自动配置 prefill:decode 比例。
- Rust 核心，Python 可扩展。
- 吞吐提升：NVIDIA 报告在中等延迟区间，DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo 上有 6 倍提升（developer.nvidia.com，2025-06）；社区关于完整 Blackwell + Dynamo + DeepSeek-R1 栈「最高 30 倍」的说法缺乏单一原始来源，应视为方向性结论。
- GB300 NVL72 + Dynamo：根据 Dynamo 产品页（developer.nvidia.com，未标注日期），MoE 吞吐可达 Hopper 的 50 倍。

**llm-d**（Red Hat + AWS，Kubernetes 原生）：
- Prefill / decode / router 作为独立的 Kubernetes Service。
- 按角色的 HPA，使用队列深度（prefill）/ KV 利用率（decode）信号。
- `topologyConstraint packDomain: rack` 把 prefill+decode 组打包在同一机架上，以获得高带宽 KV 传输。
- llm-d 0.5（2026）：分层 KV offload、缓存感知的 LoRA 路由、UCCL 网络、缩容到零。

如果你想要一个栈上层的托管式编排器，用 Dynamo。如果你想要 Kubernetes 原生原语、并且坚定投入 CNCF 生态，用 llm-d。

### 经济性

内部合成值（不是单篇发表的案例研究——只是数量级锚点）：

- 同址部署的推理支出：每年 $2M。
- 切换到使用 Dynamo 的分离式部署。
- 请求量不变，P99 延迟 SLA 不变。
- 报告节省：每年 $600K–$800K（降低 30–40%）。
- 没有新增硬件。

这个数字是我们综合多份客户披露得出的，而非某一篇可引用的案例研究；最接近的公开数据点是 Baseten 使用 Dynamo KV 路由实现的 TTFT 快 2 倍 / 吞吐高 61%（baseten.co，2025-10），以及 VAST + CoreWeave 在 40–60% KV 命中率下每美元 token 数多 60–130% 的预测（vastdata.com，2025-12）。节省来自给每个池子做正确的容量规划；prefill 重的负载（前缀 8K+ 的 RAG）比均衡负载受益更多。

### 什么情况下不要做分离

- prompt < 512 token 且输出 < 200 token：传输税压过收益。
- 小集群（< 4 块 GPU）：池子多样性不足。
- 团队无法运营两个 GPU 池并做按角色扩缩容：Dynamo 有帮助，但不是轻而易举。
- 没有 RDMA 网络结构：TCP 传输税更重。

### Router 与 Phase 17 · 11 的集成

分离式 router 是 KV cache 感知的（Phase 17 · 11）。请求会落到持有其前缀的 decode 池——如果没有匹配，就走 prefill → decode。命中率和分离式部署会叠加——缓存感知的 router 决定是否需要做新的 prefill。

### Blackwell 上的 MoE 才是真实数字所在

GB300 NVL72 + Dynamo 相比 Hopper 基线显示出 50 倍的 MoE 吞吐。MoE 专家路由在 prefill 阶段偏算力，在 decode 阶段偏内存（专家缓存），因此分离式是双赢。2026 年的前沿模型服务以 MoE 为主（DeepSeek-V3、未来的 GPT-5 变体）。

### 你应该记住的数字

基准数字会漂移——NVIDIA 和推理栈每季度都会发布更新结果。引用前请重新核实。

- DeepSeek-R1 on GB200 NVL72 + Dynamo：中等延迟区间约 6 倍吞吐提升（developer.nvidia.com，2025-06）；社区关于完整 Blackwell + Dynamo 栈「最高 30 倍」的说法是方向性汇总，没有单一原始来源。
- GB300 NVL72 + Dynamo：MoE 吞吐最高为 Hopper 的 50 倍（developer.nvidia.com，未标注日期）。
- 节省锚点（内部合成值，非单一案例研究）：$2M 年支出在 SLA 不变下每年省 $600-800K。
- 分离式阈值：prompt > 512 token + 输出 > 200 token。
- 通过 NIXL 的 KV 传输：70B FP8、4K prompt 的 KV 需 20-80 ms。

```figure
prefill-decode-split
```

## 使用它

`code/main.py` 模拟同址与分离式服务。报告吞吐、每请求成本和 prompt 长度的交叉点。

## 交付它

本课产出 `outputs/skill-disaggregation-decider.md`。给定负载和集群，判断是否要分离式部署。

## 练习

1. 运行 `code/main.py`。在什么 prompt 长度上，分离式开始胜过同址部署？
2. 为一个 P99 前缀长度 8K、输出 300 的 RAG 服务设计 prefill 池和 decode 池。
3. Dynamo vs llm-d：为一个纯 Kubernetes 团队选一个，该团队对 Python 运行时没有偏好。
4. 计算 KV 传输成本：70B FP8 上 4K prefill ≈ 500 MB KV。在 100 GB/s 的 RDMA 下，传输 = 5 ms。在 10 GB/s 的 TCP 下 = 50 ms。哪一个对你的 SLA 有影响？
5. MoE 专家路由会改变 KV 访问模式。对于每个 token 激活不同专家的 MoE，分离式部署表现如何？

## 关键术语

| 术语 | 人们怎么说 | 它实际指什么 |
|------|----------------|------------------------|
| 分离式服务 | 「prefill/decode 拆分」 | 为每个阶段使用独立的 GPU 池 |
| NIXL | 「NVIDIA 传输层」 | Dynamo 的节点间 KV 传输（RDMA/TCP） |
| NVIDIA Dynamo | 「编排器」 | 面向 vLLM/SGLang/TRT-LLM 的栈上层协调器 |
| llm-d | 「Kubernetes 原生」 | Red Hat + AWS 的 K8s 分离式栈 |
| Planner Profiler | 「Dynamo 自动配置」 | 测量负载，配置池比例 |
| SLA Planner | 「Dynamo 策略」 | 自动按比例匹配 prefill:decode 以满足 SLO |
| `packDomain: rack` | 「llm-d 拓扑」 | 把 prefill+decode 打包到同一机架以获得快速 KV 传输 |
| UCCL | 「统一集合通信」 | llm-d 0.5 为缩容到零提供的网络层 |
| MoE 专家路由 | 「每个 token 选专家」 | DeepSeek-V3 的模式；分离式有帮助 |

## 延伸阅读

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
