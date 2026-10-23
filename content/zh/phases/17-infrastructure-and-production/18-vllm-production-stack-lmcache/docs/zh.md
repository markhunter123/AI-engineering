# 生产级服务栈——KV Offload 与缓存感知路由

> 生产级服务栈把 router、引擎和可观测性编进一个 Kubernetes 部署——并把 KV cache 视为一种可以离开 GPU 的资源。KV offload 把 KV cache 从 GPU 显存中提取出来，在查询和引擎之间复用（先 CPU DRAM，再磁盘/Ceph）。vLLM 的 production-stack 是参考部署；LMCache 是 offload 层。vLLM 0.11.0 的 KV Offloading Connector（2026 年 1 月）通过 Connector API（v0.9.0+）让这一过程异步化、可插拔。offload 路径通常对请求路径隐藏，但缓存未命中和提升（promotion）可能增加端到端延迟。即使没有共享前缀，LMCache 也很有价值——当 GPU 的 KV 槽位耗尽时，被抢占的请求可以从 CPU 恢复，而不必重算 prefill。在 4 台 a3-highgpu-4g 上的 16 块 H100（80GB HBM）已发表基准显示：当 KV cache 超出 HBM 时，原生 CPU offload 和 LMCache 都显著提升吞吐；KV 占用较低时，所有配置都与基线持平，仅有小幅开销。

**Type:** Learn
**Languages:** Python (stdlib, toy KV-spill simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 06 (SGLang/RadixAttention)
**Time:** ~60 minutes

## 学习目标

- 画出 vLLM production-stack 的层次：router、引擎、KV offload、可观测性。
- 解释 KV Offloading Connector API（v0.9.0+），以及 0.11.0 的异步路径如何隐藏 offload 延迟。
- 量化 LMCache 的 CPU-DRAM 何时有帮助（KV > HBM）、何时只是增加开销（KV 小到能装进 HBM）。
- 在部署约束下，在 vLLM 原生 CPU offload 和 LMCache connector 之间做选择。

## 问题

你的 vLLM 服务显示，只要并发上来，GPU 的 HBM 就跑到 100%，并出现抢占事件。请求被驱逐、重新入队，一分钟内对同一个 2K token prompt 重做了四次 prefill。GPU 算力花在了冗余 prefill 上；goodput 远低于原始吞吐。

加 GPU 是线性成本。加 HBM 不可能。但 CPU DRAM 很便宜——一个插槽就有 512 GB 以上，延迟比 HBM 差几个数量级，但对于「暂时保温」的 KV cache 来说足够好。

LMCache 把 KV cache 提取到 CPU DRAM，让被抢占的请求快速恢复；跨引擎的重复前缀也能共享缓存，无需每个引擎重新 prefill。

## 概念

### vLLM production-stack

`github.com/vllm-project/production-stack` 是参考 Kubernetes 部署：

- **Router**——缓存感知（Phase 17 · 11）。消费 KV 事件。
- **引擎**——vLLM worker。每块 GPU 一个，或每个 TP/PP 组一个。
- **KV cache offload**——LMCache 部署或原生 connector。
- **可观测性**——Prometheus 抓取、Grafana 仪表盘、OTel trace。
- **控制面**——服务发现、配置、滚动更新。

以 Helm chart + operator 的形式交付。

### KV Offloading Connector API（v0.9.0+）

vLLM 0.9.0 引入了 Connector API，用于可插拔的 KV cache 后端。你的引擎把块 offload 到 connector；connector 负责存储（RAM、磁盘、对象存储、LMCache）。请求需要某个块时，connector 把它加载回来。

vLLM 0.11.0（2026 年 1 月）增加了异步 offload 路径——offload 可以在后台进行，因此在常见情况下引擎不会阻塞。端到端延迟和吞吐仍取决于负载形态、KV cache 命中率和系统压力；vLLM 自己的说明指出，自定义 kernel 的 offload 在低命中率下可能降低吞吐，而且异步调度与投机解码存在已知的交互问题。

### 原生 CPU offload vs LMCache

**vLLM 原生 CPU offload**：引擎本地。把 KV 块存到主机 RAM。实现快，零网络跳数。不跨引擎。

**LMCache connector**：集群规模。把块存到共享的 LMCache 服务器（CPU DRAM + Ceph/S3 层）。任何引擎都能访问这些块。已发表 16 块 H100 的基准。

当单个引擎有 HBM 压力时选原生。当多个引擎共享前缀时选 LMCache（有公共系统 prompt 的 RAG、共享模板的多租户）。

### 基准表现

跨 4 台 a3-highgpu-4g 的 16 块 H100（80 GB HBM）测试：

- KV 占用低（短 prompt、低并发）：所有配置与基线持平，LMCache 增加约 3-5% 开销。
- 中等占用：LMCache 开始因跨引擎的前缀复用而获益。
- KV 超出 HBM：原生 CPU offload 和 LMCache 都显著提升吞吐；LMCache 收益更大，因为跨引擎共享。

### LMCache 起决定作用的情况

- 多租户服务，系统 prompt 在租户间共享。
- RAG，文档分块在查询间重复。
- 同一基座上的微调变体（LoRA），基座模型的 KV 复用能减少冗余工作。
- 抢占频繁的负载：从 CPU 恢复比重新 prefill 便宜。

### 不要启用的情况

- HBM 压力小——你付出开销却得不到收益。
- 短上下文（<1K token）——传输时间 > 重新 prefill。
- 单租户单 prompt 负载——没有可捕获的复用。

### 与分离式服务的集成

Phase 17 · 17 的分离式服务 + LMCache 会叠加：从 prefill 池到 decode 池的 KV 传输如果没被使用，会落到 LMCache；后续查询直接从 LMCache 拉取。Phase 17 · 11 的缓存感知 router 可以把请求路由到本地缓存或 LMCache 共享缓存匹配的引擎。

### 你应该记住的数字

- vLLM 0.9.0：Connector API 发布。
- vLLM 0.11.0（2026 年 1 月）：异步 offload 路径；端到端延迟影响取决于负载、KV 命中率和系统压力（不是绝对保证）。
- 16 块 H100 基准：KV 占用超出 HBM 时 LMCache 有帮助。
- HBM 压力小时：3-5% 开销且无收益。

```figure
zero-sharding
```

## 使用它

`code/main.py` 模拟抢占频繁的负载，分别启用和不启用 LMCache。报告避免了多少次重新 prefill、吞吐增益以及盈亏平衡的 HBM 利用率。

## 交付它

本课产出 `outputs/skill-vllm-stack-decider.md`。给定负载形态和 vLLM 部署，决定用原生、LMCache 还是都不用。

## 练习

1. 运行 `code/main.py`。在什么 HBM 利用率上 LMCache 开始划算？
2. 一个租户每小时 200 次查询共享同一个 6K token 系统 prompt。计算每个租户预期的 LMCache 节省。
3. LMCache 服务器是单点故障。设计高可用策略（副本、回退到原生）。
4. LMCache 把数据存到机械盘上的 Ceph。70B FP8 下 4K token 的 KV（500 MB），读取时间与重新 prefill 相比如何？
5. 论证 vLLM 0.11.0 的异步路径是否「免费」——开销藏在哪里？

## 关键术语

| 术语 | 人们怎么说 | 它实际指什么 |
|------|----------------|------------------------|
| Production-stack | 「参考部署」 | vLLM 的 Kubernetes Helm chart + operator |
| Connector API | 「KV 后端接口」 | vLLM 0.9.0+ 的可插拔 KV 存储接口 |
| 原生 CPU offload | 「引擎本地溢写」 | 把 KV 存到同一引擎的主机 RAM |
| LMCache | 「集群 KV 缓存」 | 基于 CPU DRAM + 磁盘的跨引擎 KV cache 服务器 |
| 0.11.0 异步 | 「非阻塞 offload」 | offload 隐藏在引擎流之后 |
| 抢占 | 「腾空间驱逐」 | HBM 满时的 KV cache 腾挪 |
| 前缀复用 | 「相同系统 prompt」 | 多个查询共享开头；缓存命中 |
| Ceph 层 | 「磁盘层」 | 缓存层次中位于 DRAM 之下的持久存储 |

## 延伸阅读

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack)——Helm chart + operator。
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache)——Connector 实现。
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases)——异步路径细节。
