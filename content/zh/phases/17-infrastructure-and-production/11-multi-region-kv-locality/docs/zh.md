# 多区域 LLM 服务与 KV cache 局部性

> 轮询负载均衡对带缓存的 LLM 推理有害无益。一个请求如果没落到持有其前缀的节点上，就要支付完整的 prefill 成本——长 prompt 下 P50 约 800 ms，而缓存命中时约 80 ms。2026 年的生产模式是缓存感知路由器（Rust 写的 vLLM Router、llm-d router），它消费 KV-cache 事件，并按前缀哈希匹配来路由。近期研究（GORGO）把跨区域网络延迟作为路由目标中的显式项。商业「跨区域推理」产品（Bedrock cross-region inference、GKE 多集群网关）把推理视为不透明的——它们负责可用性，而不是 TTFT。JPMorgan 和 Mayo Clinic 在 2024 年 11 月做了 us-east-1 故障转移，耗时约 22 分钟。DR 的残酷现实是：32% 的 LLM DR 失败，是因为团队备份了权重，却忘了 tokenizer 文件或量化配置。

**Type:** Learn
**Languages:** Python (stdlib, toy prefix-cache-aware router simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## 学习目标

- 解释为什么轮询负载均衡会破坏带缓存的推理，并量化 TTFT 惩罚。
- 画出缓存感知路由器：输入（KV-cache 事件）、算法（前缀哈希匹配）、决胜规则（GPU 利用率）。
- 说出 LLM 场景下 32% DR 失败的驱动因素（缺失 tokenizer 文件 / 量化配置），并给出三文件 DR 清单。
- 区分商业跨区域产品（Bedrock CRI、GKE Multi-Cluster Gateway）与 KV 感知路由。

## 问题

你的服务运行在 us-east-1、us-west-2 和 eu-west-1。你在前面放了一个 ALB 做轮询。生产环境的前缀缓存命中率跌到 8%。TTFT P50 翻了三倍。vLLM 日志显示每个请求都在支付完整的 prefill 成本。

轮询对无状态服务是最优的。LLM 推理在设计上是有状态的——KV cache 编码了模型见过的一切。盲路由就是把请求路由进错误的缓存。

另外，你的团队有 DR 计划。你把模型权重备份到跨区域 S3。区域故障发生，你尝试故障转移，副本却拒绝启动。你忘了 tokenizer.json、量化配置和 RoPE 缩放配置放在另一个没有同步的桶里。

多区域 LLM 服务是缓存问题、路由问题和 DR 卫生问题——不是负载均衡器问题。

## 概念

### 缓存感知路由

请求带着 prompt 到达。路由器对前缀（比如前 512 个 token）做哈希，然后问每个副本「这个前缀你缓存了吗？」副本在分配和驱逐 block 时，会通过 pub/sub 通道发布 KV-cache 事件。路由器选出匹配的副本；如果无人匹配，则退回到基于 GPU 利用率的决胜规则。

**vLLM Router**（Rust，2026 生产栈）：订阅 `kv.cache.block_added` 事件，维护前缀哈希 → 副本索引，以 $O(1)$ 查找完成路由。无匹配时退回到队列深度最小的副本。

**llm-d router**：同样的模式，Kubernetes 原生。通过 ControlPlane API 发布事件。

**SGLang RadixAttention**（Phase 17 · 06）是副本内的等价机制。跨副本路由严格位于其上游。

### 数字

2K token prompt、Llama 3.3 70B FP8、H100 上的 TTFT P50：
- 缓存命中（同一副本，前缀常驻）：约 80 ms。
- 缓存未命中（冷 prefill）：约 800 ms。

10 倍差距。如果你的路由器在副本之间达到 60-80% 的前缀缓存命中，你就能以 N 副本容量逼近单副本性能。如果只有 10%，你逼近的就是朴素扩展。

### 跨区域有一个新约束——网络延迟

区域间 RTT：
- us-east-1 ↔ us-west-2：约 65 ms。
- us-east-1 ↔ eu-west-1：约 75 ms。
- us-east-1 ↔ ap-southeast-1：约 220 ms。

如果路由把请求从 us-east-1 送到 ap-southeast-1 的一个热前缀，节省的 prefill（800 → 80 ms）会被 440 ms 的往返吃掉。GORGO（2026 年研究）把这一点显式化——联合最小化 `prefill_time + network_latency`，而不是只看 prefill。通常的答案是保持区域内路由，除非是 prefill 占主导的超大 MB 级前缀。

### 商业「跨区域推理」在这里帮不上忙

AWS Bedrock cross-region inference 会在容量压力下自动把请求路由到其他区域。它优化的是可用性，不是 TTFT，并且把推理视为不透明。GKE Multi-Cluster Gateway 同理——服务级故障转移，完全不感知 KV cache。

即使使用这些产品，你仍然需要应用层的缓存感知路由器。它们处理「us-east-1 着火了」的情形；缓存感知路由处理 TTFT 的情形。

### DR 卫生——32% 的缺文件问题

2026 年被广泛引用的统计：32% 的 LLM DR 失败，是因为团队备份了权重却忘了：

- `tokenizer.json` 或 `tokenizer.model`
- 量化配置（`quantize_config.json`、AWQ scales、GPTQ zero-points）
- 模型特定配置（RoPE scaling、attention mask、chat template）
- 引擎配置（`vllm_config.yaml`、采样默认值、LoRA adapter manifest）

解决办法是最小化的三文件 DR manifest：

1. HF 模型仓库下的所有文件（权重 + 配置 + tokenizer）。
2. 引擎特定的服务配置。
3. 部署 manifest（K8s YAML、Dockerfile、依赖锁文件）。

另外：每季度做一次 DR 演练。JPMorgan 在 2024 年 11 月的 us-east-1 演练之所以能在 22 分钟内恢复，只是因为这个 playbook 被反复排练过。

### 数据驻留是正交问题

欧盟客户的 PHI 不能离开欧盟。如果你的缓存感知路由器把一个源自巴黎的请求送到 us-east-1 去做前缀匹配，无论 TTFT 收益多大，你都违反了 GDPR。优化缓存之前，先按驻留边界对路由器分区。

### 应该记住的数字

- 缓存命中与未命中的 TTFT 差距：约 10 倍（2K prompt 上 80 ms 对 800 ms）。
- 美欧区域间 RTT：约 75 ms。
- DR 失败：32% 缺少 tokenizer/量化配置。
- JPMorgan 2024 年 11 月 us-east-1 故障转移：22 分钟（30 分钟 SLA）。

```figure
cache-aware-router
```

## 使用它

`code/main.py` 在多区域工作负载上模拟三种路由策略（轮询、区域缓存感知、全局缓存感知）。它报告缓存命中率、TTFT P50/P99 以及跨区域账单。

## 交付它

本课产出 `outputs/skill-multi-region-router.md`。给定区域、驻留约束和 SLA，它设计一套路由方案。

## 练习

1. 运行 `code/main.py`。在 RTT 为 75 ms 时，prompt 长度达到多少，跨区域路由才会胜过仅本地路由？
2. 你的缓存命中率从 70% 掉到 12%。诊断三种可能原因，以及能分别证实每种原因的可观测项。
3. 为一个在 vLLM 中服务、带 5 个 LoRA 适配器的 70B AWQ 量化模型设计 DR manifest。列出每个文件和配置。
4. 论证 Bedrock cross-region inference 对一家有严格 TTFT SLO 的金融科技公司是否「足够」。引用具体行为作为依据。
5. 一个源自巴黎的请求在 us-east-1 匹配到了前缀。你会路由它吗？写出策略。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 缓存感知路由 | 「智能 LB」 | 依据前缀哈希匹配，路由到持有 KV cache 的副本 |
| KV-cache 事件 | 「缓存 pub-sub」 | 副本发布 block 添加/驱逐；路由器建立索引 |
| 前缀哈希 | 「缓存键」 | 前 N 个 token 的哈希，用作路由查找键 |
| GORGO | 「跨区域路由研究」 | arXiv 2602.11688；把网络延迟作为显式项 |
| 跨区域推理 | 「Bedrock CRI」 | AWS 产品；可用性故障转移，不感知 TTFT |
| DR manifest | 「备份清单」 | 恢复所需的每个文件——不只是权重 |
| 数据驻留 | 「GDPR 边界」 | 哪些区域能看到用户数据的法律约束 |
| RTT | 「往返时间」 | 网络延迟；美欧 75 ms，美亚 220 ms |
| LLM 感知 LB | 「缓存命中 LB」 | 作为产品类别的缓存感知路由器 |

## 延伸阅读

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1)——带网络延迟项的跨区域 KV-cache 复用。
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html)——可用性故障转移文档。
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack)——缓存感知路由器源码。
