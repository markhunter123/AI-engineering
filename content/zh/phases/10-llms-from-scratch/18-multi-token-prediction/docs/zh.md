# 多 token 预测（MTP）

> 从 GPT-2 到 Llama 3，每个自回归 LLM 在每个位置只训练一个损失：预测下一个 token。DeepSeek-V3 增加了每个位置的第二个损失：预测再下一个 token。额外 14B 参数（在 671B 模型上）通过梯度流蒸馏回主模型，训练好的 MTP 头在推理时被改用作投机解码的草稿器，接受率 80% 以上。1.8× 生成吞吐量白送。本课从 DeepSeek 技术报告构建顺序式 MTP 模块，计算损失和共享头参数布局，并解释为什么 MTP 保持了因果链，而 Gloeckle 等人最初的并行 MTP 破坏了它。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 04 (pre-training a mini GPT), Phase 10 · 15 (speculative decoding)
**Time:** ~60 minutes

## 学习目标

- 陈述 MTP 训练目标，并推导跨预测深度的联合损失。
- 解释 Gloeckle 等人的并行 MTP 头（2024）与 DeepSeek-V3 顺序式 MTP 模块的区别，以及顺序式设计为何保持因果链。
- 计算向预训练运行加入 MTP 模块的参数和内存开销。
- 从零实现一个 MTP 模块：共享 embedding、逐深度 Transformer 块、投影和共享输出头。

## 问题

下一 token 预测是标准的 LLM 训练目标。每个隐藏状态只被监督预测一件事：紧跟其后的 token。这是一个出奇弱的信号。序列中的大部分信息都跨越不止一个 token——结构、连贯性、事实性、算术流程。模型必须通过在数万亿 token 上累积许多单 token 信号来学习这些。

MTP 问：如果每个隐藏状态同时被监督预测多个未来 token 会怎样？Gloeckle 等人（Meta，2024）表明这有帮助。他们的实现在主干上放置几个独立的输出头，每个预测一个不同的偏移。并行、简单，但各头看到的是同一个隐藏状态，没有层级细化——而且预测之间没有因果链接，因此不能用于投机解码。

DeepSeek-V3（2024 年 12 月）把 MTP 重新设计成顺序式模块，在每个预测深度保持因果链。模型从 `h_i^(0)` 预测 `t+1`，再从结合了 `h_i^(0)` 与 `E(t+1)` embedding 的新隐藏状态 `h_i^(1)` 预测 `t+2`，以此类推。每个深度有自己的小型 Transformer 块。共享 embedding 和共享输出头让参数开销保持适度。在 DeepSeek-V3 的规模下，MTP 模块在 671B 主模型权重之上增加了 14B 参数。这 2% 的开销换来了更密集的训练信号，以及推理时现成的投机解码草稿。

本课从零构建单个 MTP 模块和 D 深度损失。数学简洁，实现约 150 行。

## 概念

### 顺序式 MTP 配方

DeepSeek-V3 在主模型之上增加 `D` 个 MTP 模块。每个模块 `k`（`k = 1..D`）预测深度 `k` 的 token——即给定经过位置 `i` 的前缀，预测 `t_{i+k}`。

模块 `k` 由以下组成：

- 一个 Transformer 块 `T_k`，有自己的注意力和 MLP。
- 一个投影矩阵 `M_k`，把上一深度的隐藏状态与下一深度真值 token 的 embedding 结合。
- 共享 embedding `E`（与主模型相同）。
- 共享输出头 `Out`（与主模型相同）。

训练时，对经过位置 `i` 的前缀，逐深度隐藏状态为：

```math
\begin{aligned}
h_i^{(0)} &= \text{main model backbone at position } i \\
h_i^{(k)} &= T_k\Big( M_k \cdot \mathop{\mathrm{concat}}\big(\mathop{\mathrm{RMSNorm}}(h_i^{(k-1)}), \mathop{\mathrm{RMSNorm}}(E(t_{i+k}))\big) \Big) \quad \text{for } k \ge 1
\end{aligned}
```

逐深度预测为：

```math
\text{logits}_{i+k} = \mathop{\mathrm{Out}}(h_i^{(k-1)}) \quad \text{for } k = 1, \ldots, D
```

逐深度损失是对真值 `t_{i+k}` 的交叉熵：

```math
L_k = \mathop{\mathrm{CE}}(\text{logits}_{i+k}, t_{i+k})
```

跨深度的联合损失：

```math
L_{\text{MTP}} = \frac{\lambda}{D} \sum_{k=1}^{D} L_k
```

`lambda` 是一个小的加权因子——DeepSeek-V3 在训练前 10% 用 0.3，之后用 0.1。总训练损失是 `L_main + L_MTP`。

### 为何顺序而非并行

Gloeckle 最初的并行 MTP 有 D 个输出头，每个直接作用于 `h_i^(0)`。每个头从同一个主干隐藏状态预测 `t_{i+k}`。这样训练没问题，但预测之间不互为条件。你无法用 `head_1` 的输出帮助 `head_2`——各头并行触发。

DeepSeek-V3 的顺序式设计从 `h_i^(k-1)` 加上实际的下一 token embedding `E(t_{i+k})` 构造 `h_i^(k)`。这保持了因果链：要预测 `t_{i+k+1}`，深度 `k+1` 的模块能看到 `t_{i+k}` 处的内容。这在结构上等同于自回归解码器消费自身输出的方式——使 MTP 模块可以直接用作投机解码草稿器。

推理时：把 `h_i^(k-1)` 和起草的 `t_{i+k}` 送入模块 `k+1`，得到 `t_{i+k+1}` 的预测。重复。这正是 EAGLE 式草稿，用训练好的 MTP 模块作为草稿网络。DeepSeek-V3 报告第一个 MTP 模块接受率 80% 以上，加速约 1.8×。

### 参数核算

对隐藏维度为 `h`、词表为 `V` 的模型：

- 主模型：数十亿参数，外加一个大小为 `V * h` 的输出头。
- 共享输出头：复用主模型的头。无额外参数。
- 共享 embedding：复用主模型的 embedding。无额外参数。
- 每个 MTP 模块：
  - 投影 `M_k`：`(2h) * h = 2h^2`。
  - Transformer 块 `T_k`：注意力（MHA 为 `4h^2`）加 MLP（SwiGLU 比例 8/3 时通常为 `8h^2`）。每块约 `12h^2`。

每模块额外总计：`~14h^2`。对 DeepSeek-V3 的 `h = 7168`、D = 1 个模块：纸面约 `14 * 7168^2 = ~720M` 参数。DeepSeek-V3 报告 14B——差异主要是 MTP 模块中的专家层同样是 MoE。

### 投机解码的回报

预训练期间，MTP 模块让训练慢约 10%（更多前向计算、额外损失）。回报有两方面：

1. 更密集的训练信号。每个隐藏状态看到 D+1 个监督目标。在 MMLU、GSM8K、MATH、HumanEval 上的测量效果：DeepSeek-V3 的消融中有一致的几个百分点提升。

2. 推理时免费的投机解码草稿。MTP 模块已经训练好预测接下来的几个 token。改作草稿网络后，它给出 80% 以上的接受率。在这个水平上，N=3 或 N=5 的投机解码带来 1.8× 吞吐量。10% 的训练时成本在你第一次运行推理时就收回。

### 与 EAGLE 的关系

EAGLE 在预训练之后单独训练一个小草稿模型。MTP 把草稿烘焙进预训练。两种方法收敛到相近的接受率，但 pipeline 不同：

| 维度 | EAGLE-3 | MTP (DeepSeek-V3) |
|-----------|---------|------------------|
| 训练时机 | 预训练之后 | 预训练期间 |
| 与现有权重向后兼容 | 是 | 否（需重训） |
| 草稿参数 | 1-2 个 Transformer 层 | 1 个 Transformer 块 + 投影 |
| 接受率 | 0.88-0.92 | 深度 1 时 0.80+ |
| 加速之外的收益 | 仅投机解码 | 更密集训练信号 + 加速 |

```figure
multi-token-predict
```

## 构建它

`code/main.py` 端到端构建单个 MTP 模块：共享 embedding、投影、Transformer 块、共享输出头。然后在短合成序列上计算逐深度交叉熵损失，并按组件打印参数数量。32 个 token 的玩具词表让数字保持可读。

### 第 1 步：共享 embedding 表

一张 `vocab_size x hidden` 表由主模型以及每个深度的每个 MTP 模块共享。不是第二份副本——就是同一个张量。

### 第 2 步：逐深度组合

```python
def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected
```

真实的 DeepSeek-V3 把两个 RMSNorm 后的向量拼接成 `[2h]`，再用 `h x 2h` 矩阵投影。玩具实现为 stdlib 简洁使用向量加法。

### 第 3 步：深度 k 的 Transformer 块

自注意力加 MLP。玩具中，一层线性注意力块和 SwiGLU MLP 在不使用 numpy 的情况下保持结构可见。

### 第 4 步：共享输出头

复用主模型的输出投影。得到词表上的 logits。

### 第 5 步：逐深度损失

softmax(logits) 对偏移 `k` 处真值 token 的交叉熵。用 `lambda / D` 缩放因子跨深度聚合。

### 第 6 步：参数核算

打印总参数数、共享（embedding、头）参数数、每模块额外参数数。展示 MTP 额外参数相对主模型规模的比例。

## 使用它

MTP 已集成到 DeepSeek-V3（2024 年 12 月）和 DeepSeek-R1 系列。推理时：

- DeepSeek 自己的服务栈开箱即用地把 MTP 模块当作投机解码器使用。
- 截至 2026 年 4 月，vLLM 和 SGLang 已有 DeepSeek-V3 MTP 的集成路径。
- AMD 的 ROCm SGLang 教程展示了一个具体的 MTP 投机解码配置，在 V3 checkpoint 上实测 1.8× 加速。

何时在新的预训练中使用 MTP：

- 你控制完整预训练 pipeline，想储备更密集的训练信号。
- 你知道会大规模服务该模型，想要免费的投机解码。
- 你的隐藏维度至少为 4096。在 1B 规模，开销的伤害大于收益的帮助。

何时不该用：

- 微调现有的预训练稠密模型。MTP 模块没有被训练。
- 需要干净基线做对比的研究模型。MTP 改变了架构。

## 交付它

本课产出 `outputs/skill-mtp-planner.md`。给定预训练运行规格（模型规模、数据、计算），它返回集成 MTP 的计划：深度数量 D、`lambda` 调度、内存开销，以及推理时的投机解码接线。

## 练习

1. 运行 `code/main.py`。展示逐深度损失随合成信号增强而单调下降。修改合成数据使用固定模式，验证深度 1 和深度 2 的损失都收敛。

2. 计算稠密 70B 模型（hidden 8192, 80 层）加 D=1 MTP 模块的参数开销。与 DeepSeek-V3 报告的 14B 开销比较。解释为什么 DeepSeek 的数字更高：MTP 的 Transformer 块继承了相同的 MoE 结构，抬高了每模块参数数。

3. 在玩具中实现 D=2：加入第二个 MTP 模块，接收 $h^{(1)}$ 并预测 `t_{i+2}`。验证联合损失和参数核算与 DeepSeek 论文的等式 19-21 一致。

4. 把玩具改成并行 MTP（Gloeckle 风格）：在主隐藏状态之上加 D 个输出头，各预测不同偏移。在相同合成信号上测量各深度损失与顺序版本的对比。顺序版本对 k > 1 应产生更低的深度 k 损失，因为它以中间预测为条件。

5. 把训练好的 MTP 模块用作 EAGLE 式草稿：推理时调用模块 k 提出 `t_{i+k}`。在留出序列上测量这些草稿 token 相对主模型预测的接受率。如果在玩具上达到 50% 以上，你就复现了 MTP 作为草稿的实证性质。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| MTP 模块 | 「额外损失块」 | 一个小型 Transformer 块加投影，预测主模型前方 `k` 个位置的 token |
| 预测深度 | 「哪个偏移」 | 整数 `k`，使模块 `k` 从经过位置 `i` 的前缀预测 `t_{i+k}` |
| 并行 MTP | 「Gloeckle 风格」 | 同一主干隐藏状态上的 D 个独立头，无条件链 |
| 顺序式 MTP | 「DeepSeek-V3 风格」 | 每个模块以上一深度的隐藏状态加下一 token 的 embedding 为条件；保持因果链 |
| 共享输出头 | 「复用主头」 | MTP 模块调用主模型的 LM 头，而非单独的输出投影 |
| 共享 embedding | 「复用主表」 | 同一张词表 embedding 表处处使用；无重复参数 |
| 投影矩阵 $M_k$ | 「组合隐藏状态 + 下一 token」 | 一个 `h x 2h` 线性层，把上一隐藏状态与目标 token embedding 折叠为下一深度的输入 |
| 联合损失 $L_{\text{MTP}}$ | 「平均额外损失」 | 逐深度交叉熵损失的算术平均，按 `lambda` 缩放 |
| 深度 1 接受率 | 「MTP 草稿有多常是对的」 | D=1 时 MTP 模块的 top-1 预测等于主模型 top-1 预测的比率；DeepSeek-V3 上 80%+ |
| Lambda 加权 | 「额外损失的重要性」 | 逐深度缩放因子；DeepSeek-V3 训练初期 0.3，后期 0.1 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) —— 完整的顺序式 MTP 描述（第 2.2 节），包括联合损失公式和推理时 1.8× 加速
- [Gloeckle et al. — Better & Faster Large Language Models via Multi-token Prediction (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) —— DeepSeek 设计所改进的并行 MTP 基线
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) —— 685B 总量（671B 主模型 + 14B MTP），部署说明
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) —— MTP 所融入的投机解码框架
- [Li et al. — EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) —— EAGLE 2025 年的草稿架构，MTP 的竞争对照
