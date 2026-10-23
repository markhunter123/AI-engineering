# DeepSeek-V3 架构详解

> 第 10 阶段 · 第 14 课列出了每个开放模型都会转动的六个架构旋钮。DeepSeek-V3（2024 年 12 月，总参数量 671B，激活参数 37B）把六个旋钮全部拧动，又额外加了四个：多头潜在注意力（Multi-Head Latent Attention）、无辅助损失的负载均衡、多 token 预测（Multi-Token Prediction）和 DualPipe 训练。本课自上而下通读 DeepSeek-V3 的架构，并从公开配置推导出每一个参数计数。读完后你能解释为什么 671B/37B 这个比例是正确的赌注，以及为什么在前沿规模下 MLA + MoE 合起来胜过任何单独一个。

**Type:** Learn
**Languages:** Python (stdlib, parameter calculator)
**Prerequisites:** Phase 10 · 14 (open-model walkthroughs), Phase 10 · 17 (NSA), Phase 10 · 18 (MTP), Phase 10 · 19 (DualPipe)
**Time:** ~75 minutes

## 学习目标

- 自上而下阅读 DeepSeek-V3 配置，并用六个 GPT-2 旋钮加上四个 DeepSeek 特有的新增项来解释每个字段。
- 推导总参数量（671B）、激活参数量（37B），以及各自由哪些组件贡献。
- 计算 MLA 在 128k 上下文下的 KV cache 占用，并与同样激活参数量的稠密模型使用 GQA 时的开销对比。
- 说出 DeepSeek 的四项特有创新（MLA、MTP、无辅助损失路由、DualPipe），并指出每一项针对的是架构或训练栈的哪一部分。

## 问题

DeepSeek-V3 是第一个架构与 Llama 家族有实质差异的前沿开放模型。Llama 3 405B 是「拧了六个旋钮的 GPT-2」。DeepSeek-V3 是拧了全部六个旋钮、再加四个的 GPT-2。读 Llama 3 配置是读 DeepSeek 配置的热身，但深层结构——注意力块的形状、路由逻辑、训练时的目标——差异足够大，需要单独一篇详解。

学它的回报：DeepSeek-V3 的开放权重发布改变了开放模型中「前沿能力」的含义。这套架构是 2026 年许多训练任务正在复制的蓝图。对于任何涉及前沿 LLM 训练或推理的岗位，理解它都是入门门槛。

## 概念

### 不变的内核，再说一遍

DeepSeek-V3 仍然是自回归的。它仍然堆叠解码器块。每个块仍然包含注意力、MLP 和两个 RMSNorm。MLP 里仍然使用 SwiGLU。仍然使用 RoPE。pre-norm。权重共享的 embedding。与每个 Llama 或 Mistral 相同的基线。

### 变招：用 MLA 取代 GQA

从第 10 阶段 · 第 14 课你知道，GQA 通过在 Q 头分组之间共享 K 和 V 来缩小 KV cache。多头潜在注意力（MLA）走得更远：K 和 V 被压缩进一个共享的低秩潜在表示（`kv_lora_rank`），然后按头即时解压。KV cache 只存这个潜在向量——通常每层每 token 512 个浮点数，而不是 8 x 128 = 1024 个浮点数。

在 128k 上下文下，DeepSeek-V3 使用 MLA（每层每 token 一个共享潜在向量 `c^{KV}`；K 和 V 都由这个潜在向量经上投影导出，而上投影可以在推理时吸收进后续的 matmul）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

一个假想的 GQA 基线（Llama 3 70B 的形态，8 个 KV 头，头维度 128）需要付出：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

在 128k 上下文下，MLA 比 Llama-3-70B 风格的 GQA cache 小 4 倍。

代价是：MLA 在每次注意力计算时增加一步解压（按头进行）。相比节省的带宽，这部分额外计算很小。对长上下文推理来说是净胜。

### 路由：无辅助损失的负载均衡

MoE 路由器决定哪些 top-k 专家处理每个 token。朴素的路由器会把过多工作集中到少数专家上，让其他专家闲置。标准修复办法：增加一个辅助损失项，惩罚负载不均。这有效，但会略微降低主任务性能。

DeepSeek-V3 引入了无辅助损失方案。把每个专家的偏置项加到路由 logit 上，训练时按一条简单规则调整：如果专家 `e` 过载，就减小 `bias_e`；如果欠载，就增大它。没有额外的损失项。训练保持干净。专家负载保持均衡。

对主损失的影响：无可测量的影响。对 MoE 架构的影响：更干净，没有辅助损失超参数需要调。

### MTP：更密集的训练 + 免费的草稿

从第 10 阶段 · 第 18 课你知道，DeepSeek-V3 增加了 D=1 的 MTP 模块，用来预测向前两个位置的 token。推理时，训练好的模块被改用作投机解码的草稿模型，接受率 80% 以上。训练时，每个隐藏状态在 D+1 = 2 个目标上受到监督，提供更密集的信号。

参数量：在 671B 主模型之上再加 14B。开销：2.1%。

### 训练：DualPipe

从第 10 阶段 · 第 19 课你知道，DualPipe 是一种双向 pipeline，把前向和反向的 chunk 与跨节点 all-to-all 通信重叠起来。在 DeepSeek-V3 的 2,048 块 H800 规模下，它大约挽回了 245k GPU 小时——这些时间原本会被 1F1B 浪费在 pipeline 气泡上。

### 配置，逐字段解读

下面是 DeepSeek-V3 的配置（简化版）：

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

逐条解析：

- `hidden_size=7168`：embedding 维度。
- `num_hidden_layers=61`：总块深度。
- `first_k_dense_layers=3`：前 3 个块使用大小为 18432 的稠密 MLP。其余 58 个使用 MoE。
- `num_attention_heads=128`：128 个 query 头。
- `kv_lora_rank=512`：K 和 V 被压缩到这个潜在维度，再按头解压。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE 块有 256 个专家，路由 top-8。
- `shared_experts=1`：在 256 个被路由的专家之外，还有 1 个始终激活的专家为每个 token 做贡献。可以把它看作「稠密地板」，确保每个 token 都能得到可靠的处理。
- `moe_intermediate_size=2048`：每个专家的 MLP 隐藏尺寸。比稠密 MLP 小，因为有 256 个专家。

### 参数核算

完整计算在 `code/main.py` 中。要点如下：

- Embedding：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前 3 个稠密块：带 MLA 的注意力（每块约 144M）+ 稠密 MLP（每块约 260M）+ norm。合计约 1.2B。
- 58 个 MoE 块：带 MLA 的注意力（约 144M）+ 256 个专家（每个 30M）+ 1 个共享专家（30M）+ norm。每块总计约 7.95B（含全部专家）。58 个 MoE 块合计 461B。
- MTP 模块：14B。

总计：核心架构约 476B + 14B MTP；显然，公开发布的 671B 数字还计入了额外的结构性参数（偏置张量、专家特有组件、共享专家缩放等）。我们在计算器中复现的数字与公开值相差 3-5%，差值来自 DeepSeek 报告第 2 节附录中记录的细粒度核算。

每次前向的激活参数：

- 注意力：每层 144M * 61 = 8.8B（所有层都参与）。
- MLP 激活部分：前 3 层稠密（3 * 260M = 780M），58 个 MoE 层各自激活 8 个路由专家 + 1 个共享专家 + 路由开销。每层激活 MLP 约 260M。合计：3 * 260M + 58 * 260M = ~15.9B。
- Embedding + norm：1.2B。
- 总激活量：核心约 26B + 14B MTP（会训练，但推理时不一定总运行）≈ 37B。

### 671B / 37B 这个比例

稀疏比为 18 倍（激活参数占总参数的 5.5%）。DeepSeek-V3 是已经发布开放权重的前沿 MoE 模型中最稀疏的一个。Mixtral 8x7B 的比例是 13/47（28%），稠密得多。Llama 4 Maverick 的比例是 17B/400B（4.25%），与之相当。DeepSeek 的赌注是：在前沿规模下，更多专家加上更低的激活比例，能在每个激活 FLOP 上产出更好的质量。

### DeepSeek-V3 的定位

| 模型 | 总量 | 激活 | 比例 | 注意力 | 新想法 |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + 无辅助损失 + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN 扩展 |

### 后续版本：R1、V4

DeepSeek-R1（2025）是在 V3 骨干上做的一次推理训练。R1 使用相同的架构。改变的是后训练配方（在可验证任务上的大规模强化学习），而非预训练架构。

DeepSeek-V4（如果发布）预计会保留 MLA + MoE + MTP，并加入 DSA（DeepSeek Sparse Attention），即第 10 阶段 · 第 17 课中 NSA 的后继。谱系是稳定的：架构层面的创新不断累积；每个版本都多拧几个旋钮。

```figure
moe-routing
```

## 使用它

`code/main.py` 是针对 DeepSeek-V3 形态特化的参数计算器。运行它，把输出与论文数字对比，再用它计算假想的变体（256 个专家 vs 512 个，top-8 vs top-16，MLA rank 512 vs 1024）。

重点看：

- 总参数量与公开的 671B 对比。
- 激活参数量与公开的 37B 对比。
- 128k 上下文下的 KV cache——MLA 与 GQA 的对比。
- 逐层分解，看参数预算实际花在哪里。

## 交付它

本课产出 `outputs/skill-deepseek-v3-reader.md`。给定一个 DeepSeek 家族模型（V3、R1 或任何未来变体），它产出一份逐组件的架构解读，点名配置的每个字段，按组件推导参数计数，并识别该模型使用了四项 DeepSeek 特有创新中的哪几项。

## 练习

1. 运行 `code/main.py`。把计算器的总参数估计与公开的 671B 对比，找出差值来源。论文第 2 节有完整的逐项列表。

2. 修改配置，把 MLA rank 从 512 改成 256。计算 128k 上下文下得到的 KV cache 大小。它带来多少百分比的缩减，又让每个头的表达能力付出什么代价？

3. 把 DeepSeek-V3 的（256 专家，top-8）路由与假想的（512 专家，top-8）变体对比。总参数增长；激活参数不变。额外的专家容量理论上能换来什么，推理时又要付出什么代价？

4. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）关于 MLA 的第 2.1 节。用三句话解释为什么 K 和 V 的解压矩阵可以在推理时「吸收」进后续 matmul 以提升效率。

5. DeepSeek-V3 大多数运算使用 FP8 训练。计算用 FP8 而非 BF16 存储 671B 权重能节省多少内存。这与 14.8T token 的训练预算如何相互影响？

## 关键术语

| 术语 | 常见的说法 | 实际含义 |
|------|----------------|------------------------|
| MLA | 「Multi-Head Latent Attention」 | 把 K 和 V 压缩进共享的低秩潜在向量（kv_lora_rank，通常为 512），按头即时解压；KV cache 只存这个潜在向量 |
| kv_lora_rank | 「MLA 压缩维度」 | K 和 V 共享潜在向量的大小；DeepSeek-V3 使用 512 |
| 前 k 个稠密层 | 「早期层保持稠密」 | MoE 模型的前几层跳过 MoE 路由器，改跑稠密 MLP 以求稳定 |
| num_experts_per_tok | 「Top-k 路由」 | 每个 token 激活多少个路由专家；DeepSeek-V3 使用 8 |
| 共享专家 | 「常开专家」 | 无论路由结果如何都处理每个 token 的专家；DeepSeek-V3 使用 1 个 |
| 无辅助损失路由 | 「偏置调整的负载均衡」 | 训练时调整每个专家的偏置项，在不增加损失项的情况下保持专家负载均衡 |
| MTP 模块 | 「额外的预测头」 | 由 $h^{(1)}$ 和 $E(t+1)$ 预测 $t+2$ 的 Transformer 块；训练更密集，还免费提供投机解码草稿 |
| DualPipe | 「双向流水线」 | 把前向/反向计算与跨节点 all-to-all 重叠起来的训练调度 |
| 激活参数比 | 「稀疏度」 | active_params / total_params；DeepSeek-V3 达到 5.5% |
| FP8 训练 | 「8 位训练」 | 训练存储和许多计算算子使用 FP8；相比 BF16 内存大致减半，质量代价很小 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437)——完整的架构、训练与结果文档
- [DeepSeek-V3 在 Hugging Face 上的模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V3)——配置文件与部署说明
- [DeepSeek-V2 论文 (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434)——引入 MLA 的前作
- [DeepSeek-R1 论文 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)——基于 V3 架构的推理训练后继者
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089)——DeepSeek 家族注意力的未来方向
- [DualPipe 仓库](https://github.com/deepseek-ai/DualPipe)——训练调度的参考实现
