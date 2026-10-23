# 梯度检查点与激活重计算

> 反向传播会保留每一个中间激活。在 70B 参数和 128K 上下文下，每个 rank 有 3 TB 激活。检查点用 FLOP 换内存：重算而不是保存。问题是该丢弃哪些段，而答案并不是「全部」。

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 minutes

## 问题

训练 Transformer 时，每一层都要存储反向传播中被求导的每个算子的输入：注意力输入、Q/K/V 投影、softmax 输出、FFN 输入、norm 输出以及残差流。对于隐藏尺寸 `d`、序列长度 `L`、批大小 `B` 的一层，这大约是每层 `12 * B * L * d` 个浮点数。

对 `d=8192, L=8192, B=1`，即每层 800 MB（BF16）。64 层模型就是 51 GB 激活——而这还没乘以微批大小，还没加入注意力 softmax 中间量（每个头 `L^2`），也还没算张量并行带来的部分副本。

双面账单：BF16 权重加优化器状态也许能塞进 80GB，但激活会把你推过线。梯度检查点（又称激活重计算）是标准修复。丢弃大部分激活；在反向传播时重做前向以取回它们。代价：额外 FLOP。收益：内存按检查点段数与总层数之比下降。

朴素实现下，检查点大约让每步前向 FLOP 增加 33%。做好之后——按 Korthikanti 等人的「智能选择」做选择性检查点——你能在不到 5% 的 FLOP 开销下节省 5 倍内存。而在 FP8 matmul、FSDP offload 和专家并行 MoE 的时代，这一点非常关键：内存和浪费的计算你都负担不起。

## 概念

### 反向传播究竟需要什么

`output = layer(input)`。反向需要 `grad_input` 和 `grad_params`。为了计算它们，需要：

- `input`（对线性层计算 `grad_params = input.T @ grad_output`）
- 一些激活导数的中间量（ReLU/GELU/softmax 的导数依赖激活值）

前向过程会自动把它们存进 autograd 图。每个 `tensor.retain_grad()` 和每个需要其输入的算子都会保留一个引用。

### 朴素的全量检查点

把网络切分成 `N` 个段。前向时只存每个段的*输入*。反向需要中间量时，重跑该段的前向以物化它们，再做求导。

例子：32 层 Transformer 切成 32 个各含 1 层的段。

- 内存：存 32 个层输入（小）对比存 32 *（每层激活体积）（巨大）。
- 额外计算：每个段多一次前向，即前向 FLOP 总共多约 33%（因为反向是前向的 2 倍，整步从 1 + 2 = 3 个单位变成 1 + 1 + 2 = 4 个单位）。

这是 Chen 等人 2016 年的原始配方：每 `sqrt(L)` 层设一个检查点，以平衡内存和计算。对 $L=64$，就是 8 个检查点。

### 选择性检查点（Korthikanti 2022）

并非所有激活的成本都一样。注意力 softmax 输出是 `B*L*L*heads`，随序列长度*二次*增长。FFN 隐藏激活是 `B*L*4d`，线性增长。长序列下 softmax 占主导。

选择性检查点保留存储便宜的激活（线性投影、残差），只重算昂贵的那些（注意力）。你付出极小的 FLOP 来重算，却省下 $O(L^2)$ 的内存。

Megatron-Core 把它实现为「selective」激活重计算。2024 年以后的大多数前沿训练都在用。

### Offload

重算的替代方案：在前向与反向之间把激活运到 CPU RAM。需要 PCIe 带宽；当空闲带宽的收益超过重新物化的成本时有利。混合策略很常见：一些层做检查点，另一些层 offload。

FSDP2 把 offload 作为一等选项提供。当 GPU 受内存瓶颈限制而 CPU-GPU 传输还有余量时，offload 表现出色。

### 重算成本模型

每 `k` 层设检查点、共 `L` 层的每步 FLOP：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

使用选择性检查点时，只重算注意力核，而不是整层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层激活体积：`A`。对 `L` 层，总激活内存为 `L * A`。

全量检查点（段大小为 1）：只存 `L * input_volume`（对标准 Transformer 约为 `L * 1/10 A`）。节省约 `9 * L * A * 1/10`。

每 `k` 层检查点：存 `L/k * A`，再加上活动段内 `k-1` 层的份量。

取 `k = sqrt(L)` 时，内存和重算成本都随 `sqrt(L)` 缩放——这是均匀成本层的最优权衡。

### 何时不要做检查点

- 流水线阶段中最内层、已经在飞行中的层。它们无论如何都得跑完。
- 如果首尾层主导该阶段的计算（Transformer 中很少见）。
- 已经使用 FlashAttention 的注意力核——Flash 已经快速重算了 softmax，再叠加层级检查点收益很小。

### 实现模式

1. **函数包装：** 用 `torch.utils.checkpoint.checkpoint(fn, input)` 包装一个段。PyTorch 只存 `input`，反向时重算其余一切。

2. **装饰器方式：** 把层标记为可检查点；训练器在配置时决定哪些段被包装。

3. **手动显式重算：** 自己写反向传播，调用自定义的 `recompute_forward`，用保存的输入复制前向。

三种方式的功能结果相同。包装器是标准惯用法。

### 与 TP / PP / FP8 的交互

- **张量并行：** 检查点的输入在重算时必须 gather 或重新 scatter；要处理通信成本。
- **流水线并行：** 典型模式是对每个流水线阶段的前向做检查点，使逆序的微批能复用激活内存。
- **FP8 重算：** 重算期间更新的 amax 历史必须与原前向一致，否则 FP8 缩放会漂移。多数框架会快照该缩放值。

```figure
activation-recompute
```

## 构建它

### 第 1 步：一个带分段的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 第 2 步：需要全部激活的朴素反向

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 第 3 步：每 k 层检查点的内存

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 第 4 步：成本模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 第 5 步：内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 第 6 步：最优段大小

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 第 7 步：选择性检查点决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 使用它

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint`——PyTorch 中的标准包装器。包装一个函数；只存输入，反向时重算。
- **Megatron-Core 激活重计算**：支持 `selective`、`full` 和 `block` 模式。2024 年以后前沿训练的标配。
- **FSDP2 offload**：在 FSDP2 中配合 `offload_policy` 使用 `module.to_empty(device="cpu")`，把激活分片到 CPU，而不是重算。
- **DeepSpeed ZeRO-Offload**：对优化器状态和激活做 CPU offload，与检查点互补。

## 交付它

本课产出 `outputs/prompt-activation-recompute-policy.md`——一个提示，接收你的模型配置（层数、hidden、seq、batch）和可用 GPU 内存，输出逐层的重算策略（none / selective / full / offload）。

## 练习

1. 验证正确性。运行 `model_forward` + `model_backward`（全激活）对比 `model_forward_checkpointed` + `model_backward_checkpointed`（分段）。参数梯度必须精确到机器精度一致。

2. 把段大小 `k` 从 1 扫到 `L`。绘制 FLOP 开销和内存。找出曲线的拐点。

3. 实现选择性检查点：存注意力模块的输入，但不存其中间量。在 32 层、seq=8192 的模型上，测量相对整层检查点的 FLOP 开销。

4. 加入 offload。把段输入保存到模拟的「CPU 缓冲区」（一个单独的列表）。用字节/时间度量「PCIe 带宽」，找出 offload 与重算的盈亏平衡点。

5. 在真实 PyTorch Transformer 上对比启用和不启用 `torch.utils.checkpoint` 的基准。测量内存（通过 `torch.cuda.max_memory_allocated`）和步时间。

## 关键术语

| 术语 | 常见的说法 | 实际含义 |
|------|----------------|----------------------|
| 梯度检查点 | 「重做前向来省内存」 | 只存段输入；反向时重算中间量，以得到支撑梯度的张量 |
| 激活重计算 | 「与检查点相同」 | 同一技术的 HPC 叫法 |
| 段大小（k） | 「每个检查点包含多少层」 | 中间量一起被丢弃并重新物化的层数 |
| 选择性检查点 | 「Korthikanti 的技巧」 | 只重算存储昂贵的激活（注意力 softmax）；保留便宜的 |
| 全量检查点 | 「朴素版本」 | 每个段重算每一层的中间量 |
| 块检查点 | 「粗粒度」 | 以整个 Transformer 块为单位做检查点；粒度最大 |
| FLOP 开销 | 「计算税」 | 每步额外 FLOP =（重算 FLOP）/（前向 + 反向 FLOP）；朴素 33%，选择性 5% |
| 激活 offload | 「运到 CPU」 | 在前向→反向之间把激活移到 CPU RAM；重算的替代方案 |
| sqrt-L 规则 | 「经典最优解」 | 对均匀成本层，最优检查点间距是 $\sqrt{L}$ 层 |
| 注意力 softmax 体积 | 「O(L^2) 问题」 | $L^2 \cdot \text{heads} \cdot \text{batch}$ 个浮点数；长上下文下主导激活内存 |

## 延伸阅读

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174)--形式化梯度检查点的原始论文
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198)--选择性激活重计算与形式化成本分析
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645)--通过反向模式重新物化实现的恒定内存替代方案
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840)--规模化激活 offload
- [PyTorch torch.utils.checkpoint 文档](https://pytorch.org/docs/stable/checkpoint.html)--标准 API
- [Megatron Bridge 激活重计算文档](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/activation-recomputation.html)--selective、full 和 block 模式
