# 扩展：分布式训练、FSDP、DeepSpeed

> 你的 1.24 亿参数模型在单块 GPU 上训练完了。现在试试 70 亿参数。模型装不进内存。数据在单机上要跑几周。在大规模场景下，分布式训练不是可选项。它是唯一的出路。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 04 (Pre-Training a Mini GPT)
**Time:** ~120 minutes

## 学习目标

- 解释三种并行方式（数据并行、张量并行、pipeline 并行），以及根据模型和集群规模判断各自何时必要
- 使用 PyTorch DDP 实现数据并行训练，在多块 GPU 之间同步梯度
- 计算给定模型规模的内存预算（权重 + 优化器状态 + 梯度 + 激活值），确定最低硬件要求
- 配置 FSDP 或 DeepSpeed ZeRO 阶段，在多块 GPU 上分片模型状态，装下超出单 GPU 内存的模型

## 问题

一个 70 亿参数的模型，FP16 下仅权重就需要 14GB。Adam 优化器为每个参数额外存两份副本（一阶矩和二阶矩估计）。这又是 28GB。反向传播中的梯度再加 14GB。还没存一个激活值，你就已经用掉 56GB。

一块 NVIDIA A100 有 80GB 内存。

80GB 里用掉 56GB。剩下 24GB 给激活值——即前向传播中计算出来、必须保留到反向传播的中间值。对于 2048 个 token 的序列、4096 维的模型，单层激活值约用 64MB。32 层就是每个样本 2GB。batch size 为 8 需要 16GB。你有 24GB。batch size 到 12 就爆了。

现在试试 700 亿参数。仅权重：FP16 下 140GB。一块 GPU 装不下。至少需要 2 块 A100（2 × 80GB = 160GB）仅仅用来放权重。再加上优化器状态和梯度，你需要得多得多：至少 3 块以上 GPU，现实中根据分片策略要 8-16 块。

Llama 3 405B 是在 16,384 块 NVIDIA H100 GPU 上训练的。这次训练估计花费了 1 亿美元算力。DeepSeek V3 训练了一个规模相当的模型，只花了约 560 万美元，靠的是巧妙的架构设计（混合专家意味着每个 token 只有一小部分参数被激活）和训练效率。

本课覆盖让大规模训练成为可能的四种策略：数据并行、张量并行、pipeline 并行和全分片数据并行。你会在纯 Python 中逐一模拟，先理解机制，再去碰任何分布式训练框架。

## 概念

### 为什么必须分布式

下面是真实模型的内存账。每个数字都是算出来的，不是估的。

| 模型 | 参数 | 权重（FP16） | Adam 状态 | 梯度（FP16） | 总计（不含激活值） |
|-------|--------|----------------|-------------|------------------|----------------------|
| GPT-2 Small | 124M | 248 MB | 992 MB | 248 MB | 1.5 GB |
| Llama 3 8B | 8B | 16 GB | 64 GB | 16 GB | 96 GB |
| Llama 3 70B | 70B | 140 GB | 560 GB | 140 GB | 840 GB |
| Llama 3 405B | 405B | 810 GB | 3,240 GB | 810 GB | 4,860 GB |

「Adam 状态」这一列最致命。Adam 为每个参数存一个运行均值（m）和一个运行方差（v），都用 FP32。对 700 亿参数的模型，就是 70B × 4 字节 × 2 = 560GB。光优化器就需要七块 A100。

一块 H100 有 80GB。Llama 3 405B 至少需要 61 块 H100 来放权重、优化器和梯度。加上激活值，数字还要涨。Meta 用 16,384 块 GPU 不是因为他们想用——而是不得不用。

### 数据并行

最简单的分布式策略。把整个模型复制到 N 块 GPU。把每个训练 batch 切成 N 等份。每块 GPU 在自己的数据分片上跑前向和反向传播。反向传播后，在所有 GPU 之间对梯度求平均。每块 GPU 用同样的平均梯度更新自己的权重副本，保持所有副本同步。

**优点：** 吞吐量线性扩展。N 块 GPU 每步处理 N 倍的数据。通信仅限于梯度平均，且能与计算重叠。

**缺点：** 每块 GPU 都持有完整的模型、优化器状态和梯度副本。对 700 亿参数的模型，每块 GPU 需要 840GB。数据并行对降低单 GPU 内存毫无帮助。它只缩短训练时间。

**数学：** 有效 batch size = per_gpu_batch_size × N。N=64 块 GPU、每 GPU batch 为 16 时，有效 batch 是 1,024。Llama 3 使用的有效 batch size 是每步 1600 万个 token。

```mermaid
graph TD
    subgraph DataParallel["Data Parallelism (N=4 GPUs)"]
        B["Full Batch\n(1024 samples)"] --> S["Split"]
        S --> G1["GPU 1\nFull Model Copy\n256 samples"]
        S --> G2["GPU 2\nFull Model Copy\n256 samples"]
        S --> G3["GPU 3\nFull Model Copy\n256 samples"]
        S --> G4["GPU 4\nFull Model Copy\n256 samples"]
        G1 --> AR["AllReduce\nAverage Gradients"]
        G2 --> AR
        G3 --> AR
        G4 --> AR
        AR --> U["Update\n(identical on all GPUs)"]
    end

    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style G1 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G2 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G3 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G4 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style AR fill:#1a1a2e,stroke:#51cf66,color:#fff
    style U fill:#1a1a2e,stroke:#51cf66,color:#fff
```

### 张量并行

把单个层拆到多块 GPU 上。一次矩阵乘法被分给多块 GPU，每块计算一部分结果。

考虑前馈层里一个形状为 (8192, 8192) 的权重矩阵。用 4 路张量并行，每块 GPU 持有 (8192, 2048) 的分片。每块 GPU 用输入乘自己的分片，得到部分结果。部分结果再合并（通过 all-reduce 或 all-gather）得到完整输出。

**优点：** 降低每块 GPU 的模型权重内存。700 亿参数的模型分到 8 块 GPU 上，每块 GPU 大约持有相当于 87.5 亿参数的权重。

**缺点：** 每一层之后都需要快速的 GPU 间通信。每次矩阵乘法后的 all-reduce 会增加延迟。在 NVLink（同节点 GPU 之间 900 GB/s）下效果很好，但在通过 InfiniBand（400 Gb/s，约 50 GB/s）连接的跨节点上表现很差。张量并行几乎总是限于单个节点（8 块 GPU）之内。

**实际应用：** Megatron-LM 开创了张量并行。Llama 3 405B 在每个节点内使用 8 路张量并行。

### Pipeline 并行

按层拆分模型。GPU 1 跑第 1-8 层。GPU 2 跑第 9-16 层。GPU 3 跑第 17-24 层。GPU 4 跑第 25-32 层。数据在 pipeline 中流动：GPU 1 算完自己的层，把激活值发给 GPU 2，GPU 2 算完再发给 GPU 3，依此类推。

**优点：** GPU 间通信极少——只有层边界处的激活值，相比梯度或权重很小。带宽需求低，因此可以跨节点。

**缺点：** Pipeline 气泡（pipeline bubble）。当 GPU 4 在算 micro-batch 1 的前向传播时，GPU 1、2、3 处于空闲（它们已经完成了自己那部分前向）。反向传播时模式反转。朴素 pipeline 下，N 个流水级，GPU 利用率只有 1/N。

**GPipe 和 PipeDream** 通过把 batch 拆成 micro-batch 来解决气泡问题。GPU 1 一完成 micro-batch 1 的前向，就开始处理 micro-batch 2。这样流水级之间的计算得以重叠。有 M 个 micro-batch 和 N 个级时，气泡比例降到 (N-1)/M。用 M=16 个 micro-batch、N=4 个级，气泡是 3/16 = 18.75% 的空闲时间。

### FSDP：全分片数据并行

FSDP 结合了数据并行的可扩展性与分片的内存效率。每块 GPU 不再持有完整的模型副本，而是只持有 1/N 的参数、梯度和优化器状态。

在一层的前向传播之前，FSDP 执行一次 **all-gather**，从所有 GPU 收集完整参数到每块 GPU 的内存。前向之后，每块 GPU 丢弃非本地参数。反向传播时，all-gather 再次运行以重建参数用于梯度计算。反向之后，**reduce-scatter** 分发梯度分片，每块 GPU 只保存 1/N 的梯度。

**700 亿参数模型在 8 块 GPU 上的账：**

| 组件 | 不用 FSDP | 用 FSDP |
|-----------|-------------|-----------|
| 权重（FP16） | 每 GPU 140 GB | 每 GPU 17.5 GB |
| Adam 状态（FP32） | 每 GPU 560 GB | 每 GPU 70 GB |
| 梯度（FP16） | 每 GPU 140 GB | 每 GPU 17.5 GB |
| **总计** | **每 GPU 840 GB** | **每 GPU 105 GB** |

不用 FSDP，700 亿参数的模型装不进一块 80GB GPU。用 FSDP 分到 8 块 GPU，每块用 105GB——等等，这仍然装不下。你需要至少 16 块 GPU 才能降到每 GPU 80GB 以下，或者把 FSDP 与激活 checkpoint 结合（反向传播时重算激活值而不是存起来）。

通信开销比朴素数据并行更高，因为每层之前都要 all-gather。但内存节省让以前不可能的训练成为可能。

```mermaid
graph TD
    subgraph FSDP["FSDP: Fully Sharded Data Parallel (4 GPUs)"]
        direction TB
        S["Model: 4 layers, sharded"]

        subgraph GPU1["GPU 1"]
            G1S["Shard: 1/4 params\n1/4 optimizer\n1/4 gradients"]
        end
        subgraph GPU2["GPU 2"]
            G2S["Shard: 1/4 params\n1/4 optimizer\n1/4 gradients"]
        end
        subgraph GPU3["GPU 3"]
            G3S["Shard: 1/4 params\n1/4 optimizer\n1/4 gradients"]
        end
        subgraph GPU4["GPU 4"]
            G4S["Shard: 1/4 params\n1/4 optimizer\n1/4 gradients"]
        end

        AG["All-Gather\n(reconstruct full params\nbefore each layer)"]
        FW["Forward Pass\n(full params temporarily)"]
        RS["Reduce-Scatter\n(distribute gradient shards\nafter backward)"]

        S --> GPU1
        S --> GPU2
        S --> GPU3
        S --> GPU4
        GPU1 --> AG
        GPU2 --> AG
        GPU3 --> AG
        GPU4 --> AG
        AG --> FW
        FW --> RS
    end

    style G1S fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G2S fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G3S fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G4S fill:#1a1a2e,stroke:#0f3460,color:#fff
    style AG fill:#1a1a2e,stroke:#e94560,color:#fff
    style FW fill:#1a1a2e,stroke:#51cf66,color:#fff
    style RS fill:#1a1a2e,stroke:#e94560,color:#fff
```

### DeepSpeed ZeRO

DeepSpeed 的 ZeRO（Zero Redundancy Optimizer）在概念上与 FSDP 相同，但由 Microsoft 独立开发。它定义了三个阶段，每个阶段分片得更激进：

| 阶段 | 分片对象 | 内存节省 | 通信 |
|-------|--------|---------------|---------------|
| ZeRO-1 | 仅优化器状态 | 约 4 倍 | 与数据并行相同 |
| ZeRO-2 | + 梯度 | 约 8 倍 | 略多 |
| ZeRO-3 | + 参数 | 约 N 倍（N 块 GPU） | 每层 all-gather |

ZeRO-3 等价于 FSDP。名字不同，机制相同。DeepSpeed 证明这一概念之后，PyTorch 将其作为原生实现加入了 FSDP。

DeepSpeed 还引入了 ZeRO-Offload（把优化器状态卸载到 CPU 内存，更便宜也更大）和 ZeRO-Infinity（卸载到 NVMe SSD）。这些以计算速度换取内存容量——卸载的操作更慢，但腾出了 GPU 内存。

### 混合精度训练

现代训练同时使用多种浮点格式：

- **前向传播**：FP16 或 BF16（16 位）。内存是 FP32 的一半。在 tensor core 上矩阵乘法快 2 倍。
- **主权重（master weights）**：FP32（32 位）。由优化器维护，用于权重更新时的数值精度。
- **损失缩放（loss scaling）**：反向传播前把损失乘以一个大常数，防止 FP16 梯度下溢为零。优化器步之前再除以同一常数。

BF16（Brain Float 16）的指数范围与 FP32 相同（8 位指数），但精度降低（7 位尾数，FP32 是 23 位）。它很少需要损失缩放，因为它能表示相同的数值范围。FP16 有 5 位指数和 10 位尾数——它能表示更精细的值，但在极端量级下会溢出/下溢。

Google 的 TPU 原生使用 BF16。NVIDIA 的 A100 和 H100 同时支持 FP16 和 BF16。业界已经基本转向 BF16，因为它消除了损失缩放的麻烦。

**70 亿参数模型的内存对比：**

| 精度 | 权重 | 优化器 | 梯度 | 总计 |
|-----------|---------|-----------|-----------|-------|
| 全程 FP32 | 28 GB | 56 GB | 28 GB | 112 GB |
| 混合（BF16 + FP32 主权重） | 14 GB | 56 GB | 14 GB | 84 GB |

混合精度在这个模型上节省 28GB。优化器状态无论如何都保持 FP32——内存大头在这里。

### Megatron-LM 与 3D 并行

真正的大规模训练会组合全部三种并行：

- **数据并行**跨节点组（扩大 batch size）
- **张量并行**在节点内（把层拆到 8 块 GPU）
- **Pipeline 并行**跨节点（把层组分到多台机器）

Llama 3 405B 在 16,384 块 H100 上：
- 每个节点内 8 路张量并行（每节点 8 块 GPU）
- 跨节点 16 路 pipeline 并行（16 个流水级）
- 剩余维度上 128 路数据并行（16,384 / 8 / 16 = 128）

这种 3D 分解（8 × 16 × 128 = 16,384）就是扩展到数千块 GPU 的方式。每块 GPU 看到不同的数据分片（数据并行），持有每层的一个切片（张量并行），计算不同的一组层（pipeline 并行）。

DeepSeek V3 走了一条不同的路。他们的混合专家（MoE）架构每个 token 只激活 671B 参数中的 37B。这意味着每块 GPU 只需计算激活参数，并存储它们的激活值。他们在 2,048 块 H800 GPU 上训练——不到 Meta GPU 数量的 1/8——花费 560 万美元，而 Meta 估计花费 1 亿美元。

```mermaid
graph TD
    subgraph ThreeD["3D Parallelism (Llama 3 405B)"]
        direction TB
        subgraph DP["Data Parallel (128-way)\nSplit batch across 128 groups"]
            subgraph PP["Pipeline Parallel (16-way)\nSplit layers across 16 stages"]
                subgraph TP["Tensor Parallel (8-way)\nSplit each layer across 8 GPUs"]
                    G1["GPU 1\nSlice of layers 1-N"]
                    G2["GPU 2\nSlice of layers 1-N"]
                    G8["GPU 8\nSlice of layers 1-N"]
                end
            end
        end
    end

    N1["Total: 8 x 16 x 128 = 16,384 GPUs"]

    style G1 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G2 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G8 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style N1 fill:#1a1a2e,stroke:#e94560,color:#fff
```

```figure
paged-kv-cache
```

## 构建它

### 第 1 步：模拟数据并行

把一个 batch 分到模拟的 GPU 上。每块 GPU 在自己的分片上跑一次前向传播。对「梯度」求平均（这里用损失值来模拟梯度）。

```python
import numpy as np

def simulate_data_parallelism(data, num_gpus, model_fn):
    batch_size = len(data)
    shard_size = batch_size // num_gpus
    remainder = batch_size % num_gpus

    gpu_losses = []
    gpu_gradients = []

    offset = 0
    for gpu_id in range(num_gpus):
        extra = 1 if gpu_id < remainder else 0
        shard = data[offset:offset + shard_size + extra]
        offset += shard_size + extra

        loss, grad = model_fn(shard)
        gpu_losses.append(loss)
        gpu_gradients.append(grad)

    avg_loss = np.mean(gpu_losses)
    avg_gradient = np.mean(gpu_gradients, axis=0)

    return avg_loss, avg_gradient
```

all-reduce 操作（对梯度求平均）是数据并行中唯一的通信。实践中，这在 NVIDIA GPU 上使用 NCCL 库，它实现 ring all-reduce：每块 GPU 把自己的 1/N 梯度发给邻居，从另一侧邻居接收 1/N，经过 N-1 步后每块 GPU 都拥有完整的平均值。总通信量：2 × gradient_size × (N-1)/N，N 很大时接近梯度大小的 2 倍。

### 第 2 步：模拟张量并行

把权重矩阵拆到多块 GPU 上。每块 GPU 计算部分矩阵乘法。合并结果。

```python
def simulate_tensor_parallelism(input_data, weight_matrix, num_gpus):
    d_in, d_out = weight_matrix.shape
    assert d_out % num_gpus == 0, f"d_out {d_out} not divisible by num_gpus {num_gpus}"
    shard_size = d_out // num_gpus

    partial_results = []
    for gpu_id in range(num_gpus):
        start = gpu_id * shard_size
        end = start + shard_size
        weight_shard = weight_matrix[:, start:end]

        partial = input_data @ weight_shard
        partial_results.append(partial)

    full_output = np.concatenate(partial_results, axis=-1)

    direct_output = input_data @ weight_matrix
    error = np.abs(full_output - direct_output).max()

    return full_output, error
```

误差应当恰好为零（或机器精度）。张量并行在数学上是精确的——结果与在单块 GPU 上做完整矩阵乘法完全相同。拆分是沿输出维度进行的，每块 GPU 产生不同的列块，拼接即可重建完整结果。

对于列并行（column-parallel）线性层（拆分输出维度），你做拼接。对于行并行（row-parallel，拆分输入维度），你做求和。在 Transformer 的 FFN 中，第一个线性层（扩张）用列并行，第二个线性层（收缩）用行并行。这样避免了两个层之间的一次 all-reduce。

### 第 3 步：模拟 pipeline 并行

把模型的层拆到虚拟 GPU 上。展示气泡问题：早期阶段空闲，而后期阶段在计算。

```python
def simulate_pipeline_parallelism(num_layers, num_stages, num_microbatches):
    layers_per_stage = num_layers // num_stages

    timeline = {}
    clock = 0

    for mb in range(num_microbatches):
        for stage in range(num_stages):
            start_time = max(
                timeline.get((stage, mb - 1, "fwd"), (0, 0))[1] if mb > 0 else 0,
                timeline.get((stage - 1, mb, "fwd"), (0, 0))[1] if stage > 0 else 0,
            )
            end_time = start_time + layers_per_stage
            timeline[(stage, mb, "fwd")] = (start_time, end_time)

    last_fwd_end = max(v[1] for v in timeline.values())

    for mb in range(num_microbatches - 1, -1, -1):
        for stage in range(num_stages - 1, -1, -1):
            deps = [last_fwd_end]
            if mb < num_microbatches - 1 and (stage, mb + 1, "bwd") in timeline:
                deps.append(timeline[(stage, mb + 1, "bwd")][1])
            if stage < num_stages - 1 and (stage + 1, mb, "bwd") in timeline:
                deps.append(timeline[(stage + 1, mb, "bwd")][1])
            start_time = max(deps)
            end_time = start_time + layers_per_stage
            timeline[(stage, mb, "bwd")] = (start_time, end_time)

    total_time = max(v[1] for v in timeline.values())
    compute_time = num_microbatches * num_stages * layers_per_stage * 2
    bubble_fraction = 1.0 - compute_time / (total_time * num_stages)

    return timeline, total_time, bubble_fraction
```

4 个级、1 个 micro-batch 时，气泡比例为 75%——四块 GPU 中有三块在任意时刻空闲。用 16 个 micro-batch 时，降到约 19%。消除气泡的代价是内存：你必须同时为所有在途 micro-batch 存储激活值。

### 第 4 步：内存计算器

计算训练任意规模模型所需的确切内存。

```python
def memory_calculator(
    params_billions,
    precision_bytes=2,
    optimizer="adam",
    num_gpus=1,
    sharding="none",
    sequence_length=2048,
    batch_size_per_gpu=1,
    hidden_dim=None,
    num_layers=None,
):
    params = params_billions * 1e9

    weight_memory = params * precision_bytes

    if optimizer == "adam":
        optimizer_memory = params * 4 * 2
    elif optimizer == "sgd":
        optimizer_memory = params * 4
    else:
        optimizer_memory = 0

    gradient_memory = params * precision_bytes

    total_no_activation = weight_memory + optimizer_memory + gradient_memory

    if hidden_dim and num_layers:
        activation_per_layer = (
            sequence_length * batch_size_per_gpu * hidden_dim * precision_bytes * 4
        )
        activation_memory = activation_per_layer * num_layers
    else:
        activation_memory = params * precision_bytes * 0.5

    if sharding == "fsdp" or sharding == "zero3":
        weight_memory /= num_gpus
        optimizer_memory /= num_gpus
        gradient_memory /= num_gpus
    elif sharding == "zero2":
        optimizer_memory /= num_gpus
        gradient_memory /= num_gpus
    elif sharding == "zero1":
        optimizer_memory /= num_gpus

    per_gpu_total = weight_memory + optimizer_memory + gradient_memory + activation_memory

    return {
        "params_billions": params_billions,
        "weights_gb": weight_memory / 1e9,
        "optimizer_gb": optimizer_memory / 1e9,
        "gradients_gb": gradient_memory / 1e9,
        "activations_gb": activation_memory / 1e9,
        "per_gpu_total_gb": per_gpu_total / 1e9,
        "total_across_gpus_gb": per_gpu_total * num_gpus / 1e9,
        "fits_on_80gb": per_gpu_total / 1e9 <= 80,
        "num_gpus": num_gpus,
        "sharding": sharding,
    }
```

这个计算器回答每个 ML 工程师都会问的问题：「我需要多少块 GPU？」输入模型规模，看它是否装得下。调整分片策略，直到每 GPU 总量降到 80GB 以下。

### 第 5 步：混合精度模拟

比较 FP32、FP16 和混合精度训练的内存使用。

```python
def mixed_precision_comparison(params_billions):
    params = params_billions * 1e9

    fp32_weights = params * 4
    fp32_optimizer = params * 4 * 2
    fp32_gradients = params * 4
    fp32_total = fp32_weights + fp32_optimizer + fp32_gradients

    fp16_weights = params * 2
    fp16_master = params * 4
    fp16_optimizer = params * 4 * 2
    fp16_gradients = params * 2
    fp16_total = fp16_weights + fp16_master + fp16_optimizer + fp16_gradients

    mixed_weights = params * 2
    mixed_optimizer = params * 4 * 2
    mixed_gradients = params * 2
    mixed_total = mixed_weights + mixed_optimizer + mixed_gradients

    return {
        "fp32_total_gb": fp32_total / 1e9,
        "fp16_with_master_gb": fp16_total / 1e9,
        "mixed_bf16_gb": mixed_total / 1e9,
        "savings_vs_fp32": 1 - mixed_total / fp32_total,
    }
```

对大多数人来说最大的意外：混合精度并不会让内存减半。优化器状态（Adam 的 m 和 v）无论精度如何都保持 FP32。对 70 亿参数模型，FP32 训练用 112GB，混合精度用 84GB。这是 25% 的减少，不是 50%。优化器才是大头。

## 使用它

### 运行所有模拟

```python
def run_all_demos():
    print("=" * 70)
    print("DATA PARALLELISM SIMULATION")
    print("=" * 70)

    np.random.seed(42)
    data = np.random.randn(64, 32)
    weight = np.random.randn(32, 16)

    def model_fn(batch):
        output = batch @ weight
        loss = np.mean(output ** 2)
        grad = 2 * batch.T @ (batch @ weight) / len(batch)
        return loss, grad

    for n_gpus in [1, 2, 4, 8]:
        loss, grad = simulate_data_parallelism(data, n_gpus, model_fn)
        print(f"  {n_gpus} GPUs: loss={loss:.4f}, grad_norm={np.linalg.norm(grad):.4f}")

    print()
    print("=" * 70)
    print("TENSOR PARALLELISM SIMULATION")
    print("=" * 70)

    x = np.random.randn(4, 8192)
    W = np.random.randn(8192, 8192)

    for n_gpus in [1, 2, 4, 8]:
        output, error = simulate_tensor_parallelism(x, W, n_gpus)
        print(f"  {n_gpus} GPUs: output_shape={output.shape}, max_error={error:.2e}")

    print()
    print("=" * 70)
    print("PIPELINE PARALLELISM SIMULATION")
    print("=" * 70)

    for n_mb in [1, 4, 8, 16, 32]:
        _, total_t, bubble = simulate_pipeline_parallelism(32, 4, n_mb)
        print(f"  {n_mb:2d} micro-batches: total_time={total_t:4d}, bubble={bubble:.1%}")

    print()
    print("=" * 70)
    print("MEMORY CALCULATOR")
    print("=" * 70)

    configs = [
        (7, "none", 1),
        (7, "fsdp", 8),
        (70, "none", 1),
        (70, "fsdp", 8),
        (70, "fsdp", 16),
        (405, "fsdp", 64),
        (405, "fsdp", 128),
    ]

    print(f"  {'Model':>8} {'Sharding':>8} {'GPUs':>5} {'Per-GPU':>10} {'Fits 80GB':>10}")
    print("  " + "-" * 50)
    for params, shard, gpus in configs:
        result = memory_calculator(params, num_gpus=gpus, sharding=shard)
        fits = "Yes" if result["fits_on_80gb"] else "No"
        print(f"  {params:>6}B {shard:>8} {gpus:>5} {result['per_gpu_total_gb']:>8.1f}GB {fits:>10}")

    print()
    print("=" * 70)
    print("MIXED PRECISION COMPARISON")
    print("=" * 70)

    for params_b in [7, 13, 70, 405]:
        result = mixed_precision_comparison(params_b)
        print(f"  {params_b}B: FP32={result['fp32_total_gb']:.0f}GB, "
              f"Mixed BF16={result['mixed_bf16_gb']:.0f}GB, "
              f"Savings={result['savings_vs_fp32']:.0%}")
```

## 交付它

本课产出 `outputs/prompt-distributed-training-planner.md`——一个 prompt，接收模型规模和可用硬件，然后生成完整的分布式训练方案：并行策略、内存预算、通信开销和预期吞吐量。

## 练习

1. 修改内存计算器，加入激活 checkpoint。使用 checkpoint 时，只在每第 K 层存储激活值（典型 K=1，即全部重算）。展示内存与计算的权衡：checkpoint 节省多少内存，又让训练慢多少（完全 checkpoint 大约多 33% 计算）？

2. 扩展 pipeline 并行模拟，实现 PipeDream 使用的 1F1B（一前一后）调度。对 4 个级、8 个 micro-batch，比较气泡比例与朴素调度。1F1B 调度应当有更小的峰值内存，因为它更早开始反向传播。

3. 实现梯度累积模拟器。不是每个 micro-batch 后都 all-reduce，而是在本地累积 K 步梯度，再 all-reduce。展示这如何把通信减少 K 倍，却产生相同的最终梯度（因而训练结果相同）。

4. 构建一个成本估算器。给定模型规模、目标 token 数、GPU 类型（A100 每小时 2 美元，H100 每小时 3.50 美元）和并行策略，估算总训练成本（美元）。用已知成本验证：据报道 Llama 3 405B 约 1 亿美元，DeepSeek V3 约 560 万美元。

5. 给内存计算器加入 ZeRO-Offload。假设每节点 CPU 内存 512GB、NVMe 2TB。展示把优化器状态卸载到 CPU 如何让 70B 模型在 4 块 GPU 上训练而不是 16 块，代价是优化器步慢 30-50%。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|----------------|----------------------|
| 数据并行（Data parallelism） | 「把模型复制到每块 GPU」 | 每块 GPU 处理不同的数据分片；每步之后通过 all-reduce 对梯度求平均 |
| 张量并行（Tensor parallelism） | 「把一层拆到多块 GPU」 | 切分权重矩阵，让每块 GPU 计算部分矩阵乘法；需要快速 NVLink 互联 |
| Pipeline 并行（Pipeline parallelism） | 「把层拆到多块 GPU」 | 每块 GPU 运行不同的一组层；数据以 micro-batch 流过 pipeline 以减少气泡 |
| FSDP | 「把所有东西分片」 | Fully Sharded Data Parallel——每块 GPU 持有 1/N 的权重、梯度和优化器状态；计算前 all-gather |
| ZeRO | 「DeepSpeed 版的 FSDP」 | Zero Redundancy Optimizer，三个阶段：分片优化器（阶段 1）、+ 梯度（阶段 2）、+ 参数（阶段 3） |
| All-reduce | 「跨 GPU 求平均」 | 一种集合通信操作，结束时每块 GPU 都拥有所有 GPU 输入的求和（或平均）——通常实现为 ring all-reduce |
| All-gather | 「从所有 GPU 收集」 | 一种集合通信操作，结束时每块 GPU 都拥有所有 GPU 数据的拼接——FSDP 用它重建完整参数 |
| Reduce-scatter | 「求和并分发」 | 一种集合通信操作，对数据做归约（求和）并把不同的块分发到不同 GPU——FSDP 用它做梯度分片 |
| 混合精度（Mixed precision） | 「用半精度训练」 | 前向/反向用 FP16/BF16，优化器状态用 FP32——节省约 25% 内存而非 50%，因为优化器占大头 |
| Pipeline 气泡（Pipeline bubble） | 「流水线中的空闲时间」 | GPU 空闲等待上一级数据的时长比例——通过使用更多 micro-batch 来降低 |

## 延伸阅读

- [Rajbhandari 等，2020——「ZeRO: Memory Optimizations Toward Training Trillion Parameter Models」](https://arxiv.org/abs/1910.02054) —— 定义了三个分片阶段的 DeepSpeed ZeRO 论文
- [Shoeybi 等，2020——「Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism」](https://arxiv.org/abs/1909.08053) —— NVIDIA 面向 Transformer 的张量并行
- [Narayanan 等，2021——「Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM」](https://arxiv.org/abs/2104.04473) —— 结合数据、张量和 pipeline 的 3D 并行
- [Zhao 等，2023——「PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel」](https://arxiv.org/abs/2304.11277) —— PyTorch 原生 FSDP 实现
- [Llama 3 技术报告](https://arxiv.org/abs/2407.21783) —— 16,384 块 GPU 训练与 3D 并行细节
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437) —— MoE 架构如何把训练成本降低一个数量级
