# 生产环境中的 EAGLE-3 投机解码

> 投机解码把一个快速的草稿模型与目标模型配对。草稿提出 K 个 token；目标模型在单次前向中验证；被接受的 token 相当于免费获得。2026 年，EAGLE-3 是生产级变体——它在目标模型的隐藏状态而非原始 token 上训练草稿头，把通用聊天上的接受率 alpha 推入 0.6-0.8 区间。正确的问题不是「草稿有多快」，而是「在我的流量上 alpha 是多少」。如果 alpha 低于约 0.55，在高并发下投机解码就是净负收益，因为每个被拒绝的草稿都要多一次目标前向。本节课教你先测 alpha，再决定是否打开开关。

**Type:** Learn
**Languages:** Python (stdlib, toy acceptance-rate simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 10 · 18 (Multi-Token Prediction)
**Time:** ~60 minutes

## 学习目标

- 说出投机解码的三代方案，并解释 EAGLE-3 相对 EAGLE-2 和经典草稿模型改了什么。
- 定义接受率 alpha，由 alpha 和 K（草稿长度）计算预期加速比，并找出在你目标并发度下的盈亏平衡 alpha。
- 解释为什么投机解码在 2026 年的 vLLM 中是可选开启（而非默认），以及为什么不测 alpha 就打开它是生产反模式。
- 写一份测量方案：用哪个 benchmark、哪种 prompt 分布、哪个并发点、以哪个指标作为放行门槛。

## 问题

解码是内存受限的。在 H100 上运行 Llama 3.3 70B FP8 时，每解码一个 token 要读取约 140 GB/s 的权重，并产出一个 token。解码期间 GPU 算力几乎闲置——瓶颈是 HBM 带宽，而不是 matmul 吞吐量。

投机解码利用这个空档。用廉价的草稿模型生成 K 个候选 token，然后让目标模型在单次前向中验证全部 K 个。每个通过验证的 token 实际上不额外付费（成本摊销进目标模型本来就要做的一次 K 批量前向）。

经典草稿模型方案使用同系列的较小模型（用 Llama 3.2 1B 为 Llama 3.3 70B 做草稿）。它有效，但接受率平庸——小模型的分布偏离目标模型。EAGLE、EAGLE-2、再到 EAGLE-3，直接在目标模型的内部状态上训练一个轻量草稿头，因此草稿分布更贴近目标模型。这就是为什么 alpha 从草稿模型的 0.4 提升到 EAGLE-3 的 0.6-0.8。

关键是：EAGLE-3 在 2026 年的 vLLM 中需要显式开启。`speculative_config` 必须明确设置。没有 flag，就没有加速。不测真实流量上的 alpha 就打开它的团队，常常看到尾部延迟变差，而不是变好。

## 概念

### 投机解码到底买到了什么

不用投机解码时，每 token 成本是一次目标前向。使用草稿长度 K、接受率 alpha 的投机解码时，每次目标前向的期望 token 数是 `1 + K * alpha`。加速比为 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是草稿加验证的开销。对 K=5、alpha=0.7：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。真实世界的数字聚集在 2-3 倍，因为生产流量上 alpha 很少有那么高，而且 epsilon 会随 batch 增大而增长。

### 为什么 alpha 是唯一重要的指标

被拒绝的 token 不会消失——它们会强制对第一个被拒 token 再做一次目标前向。在 alpha 跌到 0.4 的工作负载上，你要付草稿开销、验证开销，再加重新生成的开销。在高并发下（比如 256 并发），解码 batch 已经足够大，「仅目标」与「目标加验证」之间的内存带宽差距会缩小。在 2026 年的大多数硬件上，alpha 低于 0.55 时，投机解码就是净负的。

alpha 因工作负载而异。在 ShareGPT 风格的通用聊天上，用 ShareGPT 训练的 EAGLE-3 能达到 0.6-0.8。在领域特定流量（代码、医疗、法律）上，用通用数据训练的草稿头会掉到 0.4-0.6。训练一个领域特定的草稿头可以找回 alpha——相比目标模型微调，这是一次轻量、快速的训练任务。

### EAGLE 各代一览

- **经典草稿模型**：同系列的小模型。Alpha 0.3-0.5。基础设施简单——加载两个模型，每次目标前向时草稿跑 K 次前向。
- **EAGLE-1（2024）**：在目标隐藏状态（最后一层）上训练的单一草稿头。Alpha 约 0.5-0.6。在目标模型之上增加少量参数开销。
- **EAGLE-2（2025）**：自适应草稿长度和基于树的草稿（在一次目标前向中验证多个分支）。Alpha 约 0.6-0.7。草稿调度器更复杂。
- **EAGLE-3（2025-2026）**：在目标模型的多个层（而不只是最后一层）上训练草稿头，对齐更好。通用聊天上 alpha 约 0.6-0.8。

### 2026 年的生产配方

1. 先以原样交付目标模型。在目标并发度测出基线 TTFT、ITL、吞吐量。
2. 通过 vLLM `speculative_config` 启用 EAGLE-3 草稿。重跑基准。
3. 记录接受率 alpha。vLLM V1 把它报告为 `spec_decode_metrics.accepted_tokens_per_request`。除以请求的草稿长度即得 alpha。
4. 如果生产流量分布上的 alpha < 0.55，关掉投机解码，或训练领域特定的 EAGLE-3 草稿。
5. 在生产并发度下重跑，确认 P99 ITL 没有变差。

### 生产陷阱：P99 尾部

投机解码会降低平均 ITL，但如果不调优，P99 可能变差。被拒的草稿会触发两趟序列（草稿 + 验证失败 + 重新生成）。在满 batch 下，这两趟会串行化。要盯 P99 ITL，而不是 P50。

### EAGLE-3 已经部署在哪里

Google 于 2025 年在 AI Overviews 中部署了投机解码（质量相同，响应更快）。vLLM V1 把 `speculative_config` 作为有文档的接口交付，其特性矩阵将投机解码标记为与分块预填充兼容。SGLang 支持 EAGLE-3，把它作为前缀密集工作负载的推荐草稿路径。

### 一行盈亏平衡数学

期望加速比：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。令 `S = 1` 解出 alpha：`alpha_breakeven = verify_overhead / K`。对典型的 verify_overhead ~0.15 和 K=5：`alpha_breakeven = 0.03`。但那是纯粹的裸解码数学。在高并发下，验证开销上升，且解码 batch 已经跨序列摊薄了内存读取，所以实践中有效 alpha_breakeven 会升到约 0.45-0.55。

### 何时不要用投机解码

- 延迟无关紧要的 batch-1 离线生成。直接使用目标模型。
- 极短输出（少于 50 token）。草稿开销和验证成本占主导。
- 没有领域训练草稿头的专用领域。alpha 太低。
- 假设任意两个特性都能组合。检查你所装版本的 vLLM 兼容性矩阵：v0.18.0 把投机解码标记为与分块预填充兼容。

```figure
mx-speculative-tree
```

## 使用它

`code/main.py` 在一系列 alpha 值和草稿长度 K 上模拟带与不带投机解码的解码循环。它打印盈亏平衡 alpha、实测加速比和尾部行为。在若干 (alpha, K) 组合上运行，看看投机解码究竟在何处停止划算。

## 交付它

本节课产出 `outputs/skill-eagle3-rollout.md`。给定目标模型、流量分布描述和并发目标，它产出一份分阶段的 EAGLE-3 上线计划——基准基线、启用配置、测量 alpha、以 alpha >= 0.55 为放行门槛、盯住 P99 ITL。

## 练习

1. 运行 `code/main.py`。在 K=5 时，要实现 2 倍加速需要多大的 alpha？3 倍呢？这对 verify_overhead 有多敏感？
2. 假设生产流量中 70% 是通用聊天，30% 是代码。通用聊天用 ShareGPT 训练的 EAGLE-3 达到 alpha 0.7；代码只有 alpha 0.4。混合后的 alpha 是多少？投机解码是净正的吗？
3. 阅读 vLLM `speculative_config` 文档。说出三种模式（草稿模型、EAGLE、N-gram），并检查在你所用的 vLLM 版本中，各自能与哪些特性组合。
4. 启用 EAGLE-3 后，你看到平均 ITL 下降 25%，但 P99 ITL 上升 15%。诊断原因并提出缓解措施。
5. 计算 Llama 3.3 70B 上 EAGLE-3 草稿头的内存成本。与把 Llama 3.2 1B 作为经典草稿运行相比如何？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Speculative decoding | 「草稿加验证」 | 用廉价模型提出 K 个 token，在一次目标前向中验证全部 K 个 |
| Acceptance rate alpha | 「投机接受率」 | 草稿 token 被目标模型接受的比例；唯一重要的指标 |
| Draft length K | 「spec k」 | 每次目标前向中草稿提出的 token 数；典型 4-8 |
| Verify overhead epsilon | 「投机开销」 | 验证并重新生成相对普通目标前向的额外成本；随 batch 增长 |
| EAGLE-3 | 「最新 EAGLE」 | 2025-2026 变体；在多个目标层上训练草稿头；通用聊天 alpha 0.6-0.8 |
| `speculative_config` | 「vLLM 投机配置」 | vLLM V1 中显式开启的配置；没有默认值就没有加速 |
| N-gram spec decode | 「N-gram 草稿」 | GPU 侧草稿，利用 prompt 中的 N-gram 查找；与分块预填充兼容 |
| Break-even alpha | 「无效 alpha」 | 投机解码零加速时的 alpha；生产并发下要盯住它 |
| Rejected-draft two-pass | 「重生成成本」 | 草稿被拒时的两次目标前向；驱动 P99 尾部 |

## 延伸阅读

- [vLLM——投机解码文档](https://docs.vllm.ai/en/latest/features/spec_decode/)——关于 V1 中 `speculative_config` 与分块预填充兼容性的权威来源。
- [vLLM 投机配置 API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/)——确切的字段集。
- [EAGLE 论文（arXiv:2401.15077）](https://arxiv.org/abs/2401.15077)——最初的 EAGLE 草稿头形式化。
- [EAGLE-2 论文（arXiv:2406.16858）](https://arxiv.org/abs/2406.16858)——自适应草稿与树。
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html)——带投机解码的高效 LLM 系统。
- [BentoML——投机解码](https://bentoml.com/llm/inference-optimization/speculative-decoding)——生产上线检查清单。
