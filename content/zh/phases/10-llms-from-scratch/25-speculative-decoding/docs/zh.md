# 投机解码与 EAGLE

> 前沿 LLM 每生成一个 token 都需要对数十亿参数做一次完整前向。这次前向被大大过度配置了：大多数时候，一个小得多的模型就能正确猜出接下来的 3-5 个 token，大模型只需要*验证*这个猜测。猜对时，你用一次计算的代价得到了 5 个 token。投机解码（Leviathan 等，2023）把这套机制做得精确，EAGLE-3（2025）把接受率推到每次验证约 4.5 个 token——在输出分布一致的前提下实现 4-5 倍加速。

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10 Lesson 12 (Inference Optimization), Phase 10 Lesson 04 (Pre-training Mini-GPT)
**Time:** ~75 minutes

## 问题

70B 级模型在 H100 上的解码吞吐通常是 40-80 token/秒。每个 token 都需要一次完整前向，从 HBM 读取全部模型权重。不改变输出就无法让模型变小。批大小受内存限制也无法再增大。你被卡住了——除非你能让模型每次前向输出不止一个 token。

自回归生成看起来天然串行：`x_{t+1} = sample(p(· | x_{1:t}))`。但这里存在并发机会。如果你有一个廉价的预测器说「接下来 4 个 token 大概是 [a, b, c, d]」，你就可以在**大模型的单次前向**中验证全部 5 个位置，并接受最长的匹配前缀。

Leviathan、Kalai、Matias（2023，「Fast Inference from Transformers via Speculative Decoding」）通过一个巧妙的接受/拒绝规则把这套机制做得精确，同时保持目标模型的采样分布不变。相同的输出分布，2-4 倍更快。

## 概念

### 双模型设置

- **目标模型** `M_p`：你真正想从中采样的大而慢、高质量的模型。分布为 `p(x)`。
- **草稿模型** `M_q`：小而快、质量较低。分布为 `q(x)`。体积小 5-30 倍。

每一步：

1. 草稿模型自回归地提出 `K` 个 token：`x_1, x_2, ..., x_K ~ q`。
2. 目标模型对所有 `K+1` 个位置并行执行**一次**前向，为每个候选 token 产生 `p(x_k)`。
3. 按下面的修正拒绝采样规则，从左到右逐个接受/拒绝。接受最长的匹配前缀。
4. 如果有 token 被拒绝，从修正后的分布中采样替代 token 并停止。否则从 `p(· | x_1...x_K)` 采样一个奖励 token。

如果草稿与目标完全一致，每次目标前向你得到 K+1 个 token。如果草稿在第 1 个位置就错了，你只得到 1 个 token。

### 精确性规则

投机解码**可证明在分布上等价于从 p 采样**。拒绝规则：

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

其中 `(p - q)+` 表示逐点差值的正部。当草稿与目标一致（`p ≈ q`）时，接受率接近 1。当两者不一致时，残差分布经过构造，使得整体采样仍然精确等于 `p`。

**贪心情形。** 对 temperature=0 采样，只需检查 `argmax(p) == x_t`。是则接受；否则输出 `argmax(p)` 并停止。

### 期望加速比

如果草稿模型的 token 级接受率为 `α`，每次目标前向产生的期望 token 数为：

```math
\mathbb{E}[\text{tokens}] = \frac{1 - \alpha^{K+1}}{1 - \alpha} \qquad K = \text{draft length}, \ \alpha \in [0, 1]
```

在 `α = 0.8, K = 4` 时：`(1 - 0.8^5)/(1 - 0.8) = 3.36` 个 token 每次前向。一次目标前向的成本大致为 `cost_q * K + cost_p`（K 个草稿步加一次目标验证）。如果 `cost_p >> cost_q * K`，吞吐加速比约为 `3.36× / 1 = 3.36×`。

唯一真正的参数是 `α`，它完全取决于草稿与目标的对齐程度。好的草稿就是一切。

### 训练草稿：蒸馏

随机的小模型是糟糕的草稿。标准配方是从目标模型蒸馏：

1. 选择一个小架构（70B 目标配约 1B，7B 目标配约 500M）。
2. 用目标模型跑一个大型文本语料，存下它的下一 token 分布。
3. 用 KL 散度训练草稿去拟合目标分布（不是拟合真实 token）。

结果：`α` 在代码上通常为 0.6-0.8，自然语言聊天上 0.7-0.85。生产中加速 2-3 倍。

### EAGLE：树状草稿 + 特征复用

Li、Wei、Zhang、Zhang（2024，「EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty」）观察到标准投机解码中的两处低效：

1. 草稿要做 K 个串行步骤，每一步都是全栈。但草稿本可以复用最近一次验证中目标的特征（隐藏状态）——目标已经计算了丰富的表示，草稿却在从头重新推导。
2. 草稿输出一条线性链。如果草稿能输出一棵*树*状的候选（每个节点多个猜测），目标的单次前向就能通过树注意力掩码并行验证多条候选路径，挑出最长被接受的枝。

EAGLE-1 的改动：
- 草稿输入 = 目标在位置 t 的最终隐藏状态，而非原始 token。
- 草稿架构 = 1 个 Transformer 解码器层（不是独立的小模型）。
- 输出 = 每个深度 K = 4-8 个候选组成的树，深度 4-6。

EAGLE-2（2024）加入动态树拓扑：草稿不确定的地方树长得更宽，有把握的地方保持窄。在不增加验证成本的前提下提高 `α_effective`。

EAGLE-3（Li 等 2025，「EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test」）去掉了对固定顶层特征的依赖，用一种新的「测试时模拟」损失训练草稿——草稿训练时面对的输出匹配目标在测试时的分布，而非教师强制的训练分布。接受率从 0.75（EAGLE-2）升到 0.82（EAGLE-3），平均每次验证的 token 数从 3.0 升到 4.5。

### 树注意力验证

当草稿输出一棵树时，目标模型用**树注意力掩码**在单次前向中验证它——这是一个编码树拓扑而非纯线性的因果掩码。每个 token 只关注它在树中的祖先。验证仍然是一次前向、一次 matmul；拓扑掩码只多花几个 KV 条目。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是竞争的首 token 候选，`c, d, e, f` 是第二 token 候选，六个位置都在一次前向中验证。输出是任何被接受路径上最长的前缀。

### 何时赢，何时不赢

**赢：**
- 文本可预测的聊天/补全（代码、常见英语、结构化输出）。`α` 很高。
- 解码期间 GPU 计算有空闲的场景（内存受限阶段）。树状草稿利用可用 FLOPs。

**输 / 无收益：**
- 高度随机的输出（高温度下的创意写作）。`α` 会降到接近 `1/|vocab|`。
- 极高并发的批处理服务——批处理已经填满 FLOPs，树验证没多少空间。
- 目标模型非常小、草稿小不了多少的场景。

生产环境通常报告：聊天场景 2-3 倍墙钟加速，代码生成 3-5 倍，创意写作几乎为零。

```figure
speculative-decoding
```

## 构建它

`code/main.py`：

- 参考实现 `speculative_decode(target, draft, prompt, K, temperature)`，实现精确拒绝规则，并验证它保持目标分布（与普通目标采样相比经验 KL < 0.01）。
- 一个 EAGLE 风格的树状草稿器，用 top-p 分支构建深度为 K 的树。
- 一个树注意力掩码构建器，为验证器生成正确的因果模式。
- 一个接受率测试装置，在一个微型 LM 上运行两者（从 GPT-2-medium 目标中蒸馏一个 GPT-2-small）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 使用它

- **vLLM** 和 **SGLang** 原生支持投机解码。在 vLLM 中，给 `--speculative-config` 传一个含 `method`、`model`、`num_speculative_tokens` 的 JSON 对象；EAGLE-3 是 `"method": "eagle3"`。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE 树。
- **参考草稿模型**：`Qwen/Qwen3-0.6B`（用于 Qwen3-32B 的草稿），`meta-llama/Llama-3.2-1B-Instruct`（用于 Llama 3.x 70B 的草稿）。
- **Medusa 头**（Cai 等 2024，「Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads」）：不用草稿模型，而是在目标模型上直接加 K 个并行预测头。部署更简单，接受率略低于 EAGLE。

## 交付它

本课产出 `outputs/skill-speculative-tuning.md`——一个分析目标模型工作负载并做出选择的技能：草稿模型、K（草稿长度）、树宽、温度，以及何时回退到普通解码。

## 练习

1. 实现精确拒绝规则并经验性验证它。通过 `speculative_decode` 和普通目标采样各跑 10K 个样本；计算两个输出分布之间的 TV 距离。应 < 0.01。

2. 推导加速公式。在给定 `α` 和 `K` 下，绘制每次目标前向的期望 token 数。找出 $\alpha \in \lbrace 0.5, 0.7, 0.9 \rbrace$ 时的最优 $K$。

3. 训练一个微型草稿。取一个 124M 的 GPT-2 目标，用 KL 损失在 100M token 上蒸馏一个 30M 的 GPT-2 草稿。在留出文本上测量 `α`。预期：0.6-0.7。

4. 实现 EAGLE 风格的树状草稿。草稿输出不再是链，而是在每个深度输出 top-3 分支。构建树注意力掩码。验证目标模型接受最长的正确分支。

5. 测量失败模式。在 temperature=1.5（高随机性）下运行投机解码。展示 α 崩塌，而且由于草稿开销，算法比普通解码还慢。

## 关键术语

| 术语 | 常见的说法 | 实际含义 |
|------|-----------------|------------------------|
| 目标模型 | 「大模型」 | 你希望从中采样的慢而高质量的模型（p 分布） |
| 草稿模型 | 「猜测器」 | 小而快的预测器（q 分布）；小 5-30 倍 |
| K / 草稿长度 | 「前瞻量」 | 每次验证所猜测的 token 数 |
| α / 接受率 | 「命中率」 | 草稿提议被接受的逐 token 概率 |
| 精确拒绝规则 | 「接受测试」 | r < p/q 的比较，保持目标分布不变 |
| 残差分布 | 「修正后的 p-q」 | (p - q)+ / ||(p - q)+||_1，被拒绝时用于采样的分布 |
| 树状草稿 | 「分支投机」 | 草稿输出候选树，用树结构注意力掩码一次验证 |
| 树注意力掩码 | 「拓扑掩码」 | 编码树拓扑的因果掩码，使每个节点只关注其祖先 |
| Medusa 头 | 「并行头」 | 在目标模型上直接加 K 个预测头；没有独立草稿模型 |
| EAGLE 特征复用 | 「隐藏状态草稿」 | 草稿输入是目标的最后隐藏状态而非原始 token，从而缩小草稿 |
| 测试时模拟损失 | 「EAGLE-3 训练」 | 让草稿在匹配目标测试时分布的输出上训练，而非教师强制 |

## 延伸阅读

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192)——精确拒绝规则与理论加速分析
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318)——DeepMind 的并发投机采样论文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774)——替代草稿模型的并行头方案
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077)——特征复用与树状草稿
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858)——动态树拓扑
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840)——训练时与测试时匹配
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057)——Jacobi/前瞻解码，一种无猜测器的替代方案
