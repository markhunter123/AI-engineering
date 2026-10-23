# 投机解码与 EAGLE-3

> 第 7 阶段 · 第 16 课证明了数学基础：Leviathan 拒绝规则能精确保留验证器的输出分布。本课从 2026 年生产环境的训练栈视角来看投机解码。EAGLE-3 把草稿模型从廉价的近似器，变成专门构建的小型网络：它在验证器自身的隐藏状态上训练，并加入训练时测试循环，让训练分布与推理分布对齐。结果：端到端提速 3× 到 6.5×，聊天场景下每 token 接受率超过 0.9，且没有分布上的折衷。2026 年每一个生产推理栈都默认启用它。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 16 (speculative decoding math), Phase 10 · 12 (inference optimization)
**Time:** ~75 minutes

## 学习目标

- 用一句话陈述 Leviathan 定理，并证明投机循环产出的样本与验证器直接采样同分布。
- 梳理从原始投机解码（Leviathan 2023）经 EAGLE、EAGLE-2 到 EAGLE-3 的两年演进，并指出每一步消除的具体限制。
- 根据接受率 `α` 和草稿与验证器的成本比 `c` 计算期望加速比，并为每种情形选择最优草稿长度 `N`。
- 从零实现完整的投机循环：起草、验证、从残差分布拒绝采样、拒绝时回滚 KV cache、全部接受时输出奖励 token。

## 问题

在 70B 模型上做自回归解码，在 H100 上大约每秒 35 个 token。GPU 远未饱和。内存带宽是天花板：每个 token 都要从 HBM 加载 70B 的权重，做一步算术，产出一个浮点数。计算单元大部分时间闲着。

投机解码把这个问题变成可以真正解决的吞吐量问题。廉价的草稿模型用 `N` 次小规模前向传播提出 `N` 个 token。验证器在「前缀 + 全部 `N` 个草稿」上运行一次。如果验证器在位置 `i` 的分布与草稿一致（一种我们即将精确化的统计意义），就接受；否则拒绝，并从残差分布中采样一个修正 token。一次大模型前向最多能产出 `N+1` 个被接受的 token，而不是一个。

真正重要的定理来自 Leviathan、Kalman、Matias（ICML 2023）：输出分布与直接从验证器采样完全相同。不是近似，而是完全相同。这正是投机解码能在生产中成立的全部理由：它纯粹是延迟优化，没有质量代价。

第 7 阶段 · 第 16 课给你的是数学。本课给你的是训练栈。一个好的草稿模型带来的加速比一个廉价草稿模型多一倍以上。EAGLE、EAGLE-2 和 EAGLE-3（Li 等，2024–2025）把「草稿 = 同一模型的缩小版」变成一门精确的工程学科。2026 年的生产推理服务器默认使用 EAGLE-3。

## 概念

### 不变量：Leviathan 拒绝采样

设 `p(t)` 是草稿模型在给定前缀下对下一个 token 的分布，`q(t)` 是验证器的分布。采样一个草稿 token `d ~ p`。以 `min(1, q(d) / p(d))` 的概率接受。若拒绝，从残差分布 `(q - p)_+ / ||(q - p)_+||_1` 中采样。最终样本服从 `q` 分布。这与 `p` 有多差无关——`p` 越差，拒绝越频繁，但输出始终精确。

把 `N` 次这样的调用背靠背堆叠起来，只需在 `prefix + d_1 + ... + d_N` 上做一次验证器前向。验证器同时返回 `q_1, q_2, ..., q_{N+1}`。从左到右遍历。在位置 `j` 第一次拒绝时，从 `residual(q_j, p_j)` 采样并停止。若全部接受，则从 `q_{N+1}` 采样一个奖励 token。

### 什么决定加速比

设 `α` 为每个草稿 token 的期望接受率。设 `c = cost(draft) / cost(verifier)` 为成本比。每次验证器前向的期望接受 token 数为：

```math
\mathbb{E}[\text{accepted}] = \frac{1 - \alpha^{N+1}}{1 - \alpha}
```

每接受一个 token 的期望总墙钟时间为 `(N * c + 1) / E[accepted]`。对 `N` 最小化该值即可得到甜点。对 `α = 0.8, c = 0.05`：最优 `N` 约为 5–7，加速 3.2×。对 `α = 0.95, c = 0.02`：最优 `N` 约为 8–10，加速逼近 5×。

最大的杠杆是 `α`。固定 `N = 5`，从 `α = 0.6`（原始草稿）提升到 `α = 0.9`（EAGLE-3），每次验证器前向的期望接受 token 数从 2.2 提升到 4.1。同样的验证器，吞吐量几乎翻倍。

### 两年的演进

**原始投机解码（Leviathan，2023）。** 草稿模型是同家族中独立训练的较小 LLM。接入简单，`α ≈ 0.6`，最好情况下加速约 2×。

**EAGLE-1（Li 等，2024）。** 草稿是一个微型 Transformer——通常一到两层——以验证器最后一层的隐藏状态为输入，直接预测下一个 token。由于草稿看到了验证器的特征表示，其分布与验证器接近得多。`α` 提升到 0.7–0.8。

**EAGLE-2（Li 等，2024）。** 加入动态草稿树：不再提出单条 `N` 个 token 的序列，而是提出一棵候选树，用验证器在一次前向中为每个候选打分（树注意力），然后走概率最高的路径。草稿长度按步自适应。被接受路径上每个 token 的 `α` 超过 0.85。

**EAGLE-3（Li 等，2025，NeurIPS）。** 又有两处改动。第一，彻底去掉特征预测损失——EAGLE-1/2 训练草稿去匹配验证器的隐藏状态，这限制了更多数据能带来的收益。EAGLE-3 直接在 token 预测上训练。第二，训练时测试（TTT）：训练草稿时，把草稿自己之前的预测按多步回馈作为输入，与推理时的运行方式一致。这让训练分布与测试分布对齐，阻止误差累积。实测加速：聊天场景最高 6.5×，在 H100 的 SGLang 上 batch 64 时吞吐量提升 38%。

### KV cache 回滚

验证会用一次 pass 向验证器的 KV cache 追加 `N` 个条目。如果在位置 `j` 发生拒绝，位置 `j-1` 之后的缓存内容就错了。两种常见实现：写入暂存缓冲区、接受时提交（vLLM、TensorRT-LLM）；或保留物理 KV cache 加一个逻辑长度，拒绝时截断。两种方式的回滚成本都是每层每头的字节数，相对前向传播成本可以忽略。

对 EAGLE-2 的树搜索，验证器使用遵循树拓扑的非因果掩码运行注意力。工程实现繁琐，但计算只是一次带自定义掩码的标准 flash-attention 调用。

### 2026 年的草稿架构

| 策略 | 草稿类型 | `α` | 加速比 | 训练成本 |
|----------|-----------|-----|---------|---------------|
| 原始 | 独立的小 LLM | 0.55-0.70 | 1.8-2.3× | 无（复用现有小模型） |
| Medusa | 验证器上的额外 LM 头 | 0.65-0.75 | 2-3× | ~1B SFT token |
| EAGLE-1 | 隐藏状态上的 1 层 Transformer | 0.70-0.80 | 2.5-3× | ~60B token |
| EAGLE-2 | EAGLE-1 + 动态草稿树 | 0.80-0.88 | 3-4× | ~60B token |
| EAGLE-3 | 多层特征融合 + TTT | 0.88-0.92 | 3.5-6.5× | ~60-200B token |
| Lookahead | 无草稿（Jacobi 迭代） | N/A | 1.3-1.6× | 无 |

2026 年的生产现状：有 EAGLE-3 时 vLLM 和 SGLang 默认用它，否则用 EAGLE-2。TensorRT-LLM 为 Meta 和 NVIDIA 的公开模型提供了最快的 Medusa 路径。llama.cpp 为 CPU 部署提供原始草稿。

```figure
l5-spec-decode-eagle
```

## 构建它

见 `code/main.py`。这是完整的 Leviathan 投机循环，包含所有部件：N 个草稿、验证器并行 pass、逐位置拒绝、残差采样、奖励 token、KV 回滚，以及输出分布与直接从 `q` 采样一致的实证验证。

### 第 1 步：拒绝规则

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### 第 2 步：残差分布

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### 第 3 步：一个完整的投机步

`spec_step` 函数从 `p` 起草 `N` 个 token，然后在一次并行的 `q` 求值中验证全部。对每个草稿 token 应用拒绝规则，在第一次拒绝时从残差采样修正。如果全部接受，则从 `q_{N+1}` 输出一个奖励 token。

### 第 4 步：KV 回滚记账

模拟器为每个 worker 跟踪一个逻辑 `kv_length`。接受 `k` 个草稿时，`kv_length += k`。在位置 `j` 拒绝时，缓存虽已写入 `j` 之后，但逻辑长度被设成 `prefix_length + j + 1`——修正 token 的后一位。后续读取按逻辑长度截断。

### 第 5 步：Leviathan 检验

运行 50,000 个投机步。统计接受 token 的经验分布。与 50,000 个直接来自 `q` 的样本比较。卡方统计量应远低于临界值。定理在实践中成立。

### 第 6 步：加速比 vs. α

通过以不同幅度把 `p` 推离 `q` 来扫描草稿质量。测量 `α`，然后把每次验证器调用的期望 token 数画成 `α` 和 `N` 的函数。代码打印一张表，展示 EAGLE-3 级草稿质量（`α ≈ 0.9`）如何解锁每次验证器调用 4–5 个 token。

## 使用它

生产级 `vllm serve` 配 EAGLE-3：

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

SGLang 在 H100 上以 batch 64 运行 EAGLE-3：据 EAGLE-3 论文，吞吐量约比 batch-64 原始解码高 1.38×。

何时该用投机解码：

- 任何交互式聊天负载，p50 延迟比峰值吞吐量更重要。
- 代码生成和结构化输出（JSON、SQL）。目标分布高度可预测，`α` 超过 0.9。
- 长文本生成（数千 token）。摊销的加速持续生效。

何时不该用：

- 很小的模型（< 3B）。草稿并不比验证器便宜多少。
- 极小的 batch-1 CPU 部署。草稿模型的内存开销可能不划算。
- 温度极高的创意采样，`α` 会崩溃。

## 交付它

本课产出 `outputs/skill-eagle3-tuner.md`。给定推理负载（模型、batch 大小、目标延迟、任务画像），它推荐投机解码策略和调参（草稿家族、`N`、树深度、温度感知切换）。

## 练习

1. 运行 `code/main.py`。确认 50,000 个样本上 Leviathan 分布检验的卡方统计量保持在 95% 临界值以下。

2. 在 `α` 固定为 0.9、`c` 固定为 0.04 的情况下，把 `N` 从 1 扫描到 10。绘制每次验证器调用的期望 token 数和实际的每 token 墙钟时间。找到使墙钟时间最小的 `N`。解释曲线的形状。

3. 修改代码模拟 EAGLE-2 树搜索：每一步草稿提出形状为 `[2, 2, 2]` 的树（八条候选路径）。验证器运行一次，概率最高的被接受路径获胜。计算每个叶子的 `α` 和每次验证器调用的总 token 数。与等量计算下的线性链投机解码比较。

4. 为两条并发序列实现批量 KV 回滚模拟器。序列 A 的全部草稿被接受；序列 B 在位置 2 拒绝。证明 `kv_length` 按序列正确更新，且没有浪费工作。

5. 阅读 EAGLE-3 论文第 4 节（Training-Time Test）。用两句话解释为什么没有 TTT 的朴素草稿训练会遭受曝光偏差，以及为什么训练时把草稿自己的预测喂回给它能修复这个问题。将其与 seq2seq 中的 scheduled-sampling 文献联系起来。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Leviathan 规则 | 「min(1, q 除以 p)」 | 以概率 `min(1, q(d)/p(d))` 做伯努利接受/拒绝，拒绝时从残差采样即可精确保留验证器分布 |
| 残差分布 | 「(q 减 p) 取正再归一化」 | `(q - p)_+` 截断到零后重新归一化——拒绝时应当采样的正确分布 |
| 接受率 α | 「草稿有多常是对的」 | 拒绝规则下每 token 的期望伯努利成功概率，支配所有加速比计算 |
| EAGLE-1 | 「隐藏状态草稿」 | 以验证器最后一层隐藏状态为条件的微型 Transformer 草稿（Li 等，2024） |
| EAGLE-2 | 「动态草稿树」 | EAGLE-1 加一棵候选续写树，用一次验证器 pass 的树注意力打分 |
| EAGLE-3 | 「训练时测试」 | 去掉特征预测损失，在直接 token 预测上训练，并在训练中把草稿自己的输出喂回 |
| 训练时测试（TTT） | 「曝光偏差修复」 | 训练时以自回归方式运行草稿，使训练与测试的输入分布一致——与 scheduled sampling 直接对应 |
| KV 回滚 | 「撤销被拒的草稿」 | 拒绝后把验证器 KV cache 重置到已接受前缀长度的记账操作 |
| 奖励 token | 「白送的那个」 | 当 `N` 个草稿全部接受时，额外从 `q_{N+1}` 采样一个，不增加验证器成本 |
| 树注意力 | 「一次验证多个候选」 | 使用遵循草稿树拓扑的非因果掩码的注意力；一次前向算出树中每个节点的 `q_i` |

## 延伸阅读

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) —— 奠基论文与等价性定理
- [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) —— 同期独立提出，附有简洁证明
- [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) —— EAGLE-1，以隐藏状态为条件的草稿
- [Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) —— 动态树搜索
- [Li et al. — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) —— 2026 年的生产默认选择
- [Cai et al. — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) —— 无需草稿模型的替代方案
- [vLLM Speculative Decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html) —— 接入所有策略的权威生产参考
