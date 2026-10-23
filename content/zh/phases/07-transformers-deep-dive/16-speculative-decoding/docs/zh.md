# 投机解码——草稿、验证、重复

> 自回归解码是串行的：每个 token 都要等上一个。投机解码打破这条链：廉价模型起草 N 个 token，昂贵模型一次前向验证全部 N 个。草稿正确时，你为 N 个生成只付了一次大前向。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 07 (GPT Causal LM), Phase 7 · 12 (KV Cache & Flash Attention)
**Time:** ~60 minutes

## 问题

70B LLM 在 H100 上采样一个 token 约需 30 ms。3B 草稿模型约需 3 ms。如果让 3B 提前起草 5 个 token，再用 70B *一次*验证全部 5 个，总计 $5 \times 3 + 30 = 45$ ms，最多可接受 5 个 token——而直线生成需要 `5×30 = 150 ms`。这就是投机解码的完整卖点：用少量额外 GPU 内存（草稿模型）换取 2–4 倍更低的解码延迟。

这个技巧必须保持分布不变。投机采样（speculative sampling）由 Leviathan 等人（2023）以及 Chen 等人同期提出，保证输出序列与大模型独自生成的序列**同分布**。没有质量折衷，只是更快。

四类草稿-验证器组合主导了 2026 年的推理：

1. **原生投机（Leviathan 2023）。** 独立的草稿模型（如 Llama 3 1B）+ 验证器（如 Llama 3 70B）。
2. **Medusa（Cai 2024）。** 在验证器上加多个解码头，并行预测位置 `t+1..t+k`。没有独立草稿模型。
3. **EAGLE 家族（Li 2024、2025）。** 轻量草稿复用验证器的隐藏状态；接受率比原生更高；通常 3–4 倍。
4. **前瞻解码（Lookahead decoding，Fu 2024）。** Jacobi 迭代；完全不需要草稿模型。自投机。小众但无依赖。

2026 年每个生产推理栈默认都带投机解码。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 都至少支持原生 + EAGLE-2。

## 概念

### 核心算法

给定验证器 $M_q$ 和更便宜的草稿 $M_p$：

1. 设 $x_1 \ldots x_k$ 为已解码的前缀。
2. **起草**：用 $M_p$ 自回归地提出 $d_{k+1}, d_{k+2}, \ldots, d_{k+N}$，草稿概率为 $p_1 \ldots p_N$。
3. **并行验证**：在 $x_1 \ldots x_k, d_{k+1}, \ldots, d_{k+N}$ 上运行一次 $M_q$，得到位置 $k+1 \ldots k+N+1$ 的验证器概率 $q_1 \ldots q_{N+1}$。
4. **从左到右逐个接受/拒绝草稿 token**：对每个 $i$，以概率 $\min(1, q_i(d_i) / p_i(d_i))$ 接受。
5. 在位置 $j$ 首次拒绝时：从归一化的「残差」分布 $(q_j - p_j)_+$ 中采样 $t_j$。$j$ 之后的所有草稿丢弃。
6. 若全部 $N$ 个都被接受：从 $q_{N+1}$ 再采样一个 token $t_{N+1}$（免费的奖励 token）。

残差分布的技巧，是让输出分布与 $M_q$ 从头采样完全一致的数学关键。

### 什么决定加速比

设 $\alpha$ = 每个草稿 token 的期望接受率。设 $c$ = 草稿与验证器的成本比。每一步：

- 朴素生成每 token 一次大模型调用。
- 投机在 $\alpha$ 较高时每 $(1 - \alpha^{N+1}) / (1 - \alpha) \approx 1/(1-\alpha)$ 个 token 一次大模型调用。

在 $\alpha = 0.75$、$N = 5$ 时的典型经验法则：大模型调用少 3 倍。草稿成本便宜 5 倍。总实际耗时下降约 2.5 倍。

**$\alpha$ 取决于：**

- 草稿逼近验证器的程度。同一家族 / 同一训练数据会显著提升 $\alpha$。
- 解码策略。贪心草稿对贪心验证器：$\alpha$ 高。温度采样：更难匹配；接受率下降。
- 任务类型。代码和结构化输出接受率更高（可预测）；自由创作接受率更低。

### Medusa——没有草稿模型的草稿

Medusa 用验证器上的额外输出头替代草稿模型。在位置 `t`：

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

每个头输出自己的 logit。推理时从每个头采样得到候选序列，然后用一次前向、通过树注意力方案同时考虑所有候选续写来验证。

优点：没有第二个模型。缺点：增加可训练参数；需要监督微调阶段（约 1B token）；接受率比配好草稿的原生投机略低。

### EAGLE——复用隐藏状态获得更好的草稿

EAGLE-1/2/3（Li 等人，2024–2025）把草稿模型做成一个小 Transformer（通常 1 层），输入是验证器最后一层的隐藏状态。由于草稿看到了验证器的特征表示，其预测与验证器输出分布高度相关。接受率从约 0.6（原生）升到 0.85+。

EAGLE-3（2025）增加了对候选续写的树搜索。vLLM 和 SGLang 将 EAGLE-2/3 作为 Llama 3/4 和 Qwen 3 的默认投机路径。

### KV cache 的舞蹈

验证会把 $N$ 个草稿 token 一次前向送入验证器。这会把验证器的 KV cache 扩展 $N$ 个条目。如果某些草稿被拒绝，必须把缓存回滚到已接受前缀的长度。

生产实现（vLLM 的 `--speculative-config`、TensorRT-LLM 的 LookaheadDecoder）用草稿 KV 缓冲区处理这一点。先写入，接受才提交。概念上不难，但很琐碎。

```figure
draft-verify-tokens
```

## 构建它

见 `code/main.py`。我们实现核心投机采样算法（拒绝步骤 + 残差分布），包含：

- 一个「大模型」，对手工编码的分布做确定性 softmax（这样能解析地验证接受数学）。
- 一个「草稿模型」，是大模型的扰动。
- 一个接受 / 拒绝循环，产生与直接采样相同的边缘分布。

### 第 1 步：拒绝步骤

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是均匀随机数。`q_prob` 是验证器对草稿 token 的概率。`p_prob` 是草稿模型的概率。Leviathan 定理指出：这个 Bernoulli 决策，加上拒绝时从残差采样，能精确保持验证器的分布。

### 第 2 步：残差分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

从 `q` 逐元素减去 `p`，负值截为零，再归一化。任何拒绝时从中采样。

### 第 3 步：一个投机步骤

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

五个接受 → 一个奖励 → 一次验证器前向产出六个 token。

### 第 4 步：测量接受率

在不同草稿质量水平下运行 10,000 个投机步骤。画出接受率 vs 草稿与验证器分布之间的 KL 散度。你应该看到干净的单调关系。

### 第 5 步：验证分布等价

经验上：投机循环生成的 token 直方图应与直接从验证器采样的直方图匹配。这是实践中的 Leviathan 定理。卡方检验会在采样误差内确认。

## 使用它

生产：

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-config '{"method": "eagle", "model": "/models/llama-3.1-eagle-70b", "draft_tensor_parallel_size": 1, "num_speculative_tokens": 5}'

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-config '{"method": "draft_model", "model": "meta-llama/Llama-3.2-1B-Instruct", "num_speculative_tokens": 5}'
```

截至 2026 年中，TensorRT-LLM 的 Medusa 路径最快。`faster-whisper` 用一个小草稿为 Whisper-large 包装了投机解码。

**挑选草稿：**

| 策略 | 何时选择 | 加速 |
|----------|--------------|---------|
| 原生草稿（1B/3B Llama 家族） | 快速原型，无需训练 | 1.8–2.3× |
| Medusa 头 | 可以微调验证器 | 2–3× |
| EAGLE-2 / 3 | 生产，最大速度 | 3–4× |
| 前瞻 | 无草稿、无训练、无额外参数 | 1.3–1.6× |

**什么时候不要投机解码：**

- 单序列生成 1–5 个 token。开销占主导。
- 高度创意 / 高温采样（$\alpha$ 下降）。
- 内存受限的部署（草稿模型增加显存）。

## 交付它

见 `outputs/skill-spec-decode-picker.md`。该技能为新的推理工作负载挑选投机解码策略（原生 / Medusa / EAGLE / 前瞻）和调参（N、草稿温度）。

## 练习

1. **简单。** 运行 `code/main.py`。确认投机 token 分布在 50,000 个 token 上与验证器直接采样分布在卡方 p > 0.05 内匹配。
2. **中等。** 画出 $\alpha = 0.5, 0.7, 0.85$ 时加速比（每大模型前向的 token 数）随 $N$ 变化的曲线。找出每个 $\alpha$ 的最优 $N$。（提示：每次验证调用的期望 token 数 = $(1 - \alpha^{N+1}) / (1 - \alpha)$。）
3. **困难。** 实现一个微型 Medusa：取第 14 课项目中的 GPT，加 3 个额外 LM 头预测位置 t+2、t+3、t+4。用联合多头损失在 tinyshakespeare 上训练。比较接受率，对照截断同一模型得到的原生草稿。
4. **困难。** 实现回滚：从 10 token 前缀的 KV cache 开始，喂入 5 个草稿 token，模拟在位置 3 的拒绝。验证下一次迭代时缓存读取正确匹配「前缀 + 前 2 个已接受草稿」。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 草稿模型 | 「便宜的那个」 | 提出候选 token 的较小模型；通常比验证器便宜 10–50 倍。 |
| 验证器 | 「大的那个」 | 我们要保持其分布的目标模型；每个投机步骤运行一次。 |
| 接受率（$\alpha$） | 「草稿有多常正确」 | 验证器接受草稿的每 token 概率。通常 0.7–0.9。 |
| 残差分布 | 「拒绝时的兜底」 | 归一化的 `(q - p)_+`；拒绝时从中采样可保持验证器分布。 |
| 奖励 token | 「免费的那个」 | 全部 N 个草稿都被接受时，从验证器的下一步分布再采一个。 |
| Medusa | 「无草稿投机」 | 验证器上的多个 LM 头并行预测位置 t+1..t+k。 |
| EAGLE | 「隐藏状态草稿」 | 以验证器最后一层隐藏状态为条件的小 Transformer 草稿。 |
| 前瞻解码 | 「Jacobi 迭代」 | 用不动点迭代做自投机；无需草稿模型。 |
| 树注意力 | 「一次验证多个候选」 | 分支式验证，同时考虑多条草稿续写。 |
| KV 回滚 | 「撤销被拒绝的草稿」 | 草稿 KV 缓冲区；接受时提交，拒绝时丢弃。 |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) —— 核心算法与等价性定理。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) —— 同期提出；干净的 Bernoulli 拒绝证明。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) —— Medusa 论文；树注意力验证。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) —— EAGLE-1；以隐藏状态为条件的草稿。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) —— EAGLE-2；动态树深度。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) —— EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) —— 前瞻解码，无草稿方案。
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) —— 接入全部四种策略的权威生产参考。
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) —— EAGLE-1/2/3 的参考代码。
