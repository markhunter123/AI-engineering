# 流水线并行与气泡分析

> 张量并行把矩阵乘法切分到各 rank。流水线并行把模型切分到各 rank，每个 rank 一个阶段。Microbatch 在流水线中流动。开始和结束时的空闲时间就是气泡；最小化它就是全部的工艺所在。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 19 Track C lessons 42-49
**Time:** ~90 min

## 学习目标

- 把顺序模型切成 N 个阶段，并在 N 个 rank 上模拟 forward 流水线。
- 用 GPipe 调度（先 forward 填充，再 backward）把 M 个 microbatch 排过流水线，并计算气泡比例。
- 比较气泡与 Megatron-LM 和 PipeDream 使用的交错 1F1B 调度。
- 论证阶段分配：每个阶段的计算量相等比参数数量相等更重要。

## 问题

一个 fp16 的 70B 参数模型仅参数就需要 140 GB。没有消费级 GPU 装得下。ZeRO-3 把参数分片到各 rank，但仍然需要每个 rank 在每次 forward 时 allgather 完整层，每层付 $\log(N)$ 跳。流水线并行走另一条路：把模型切成 N 个阶段，每个 rank 放一个阶段。第 1 层的 forward 在 rank 0 上完成，把激活张量交给 rank 1；rank 1 运行第 2 层再交给 rank 2；以此类推。Backward 反向流动。内存线性下降，因为每个 rank 只持有一个阶段；计算是串行的，这就是气泡问题。

气泡是流水线开始时的空闲时间（等待第一个 microbatch 到达最后一个阶段）和结束时的空闲时间（等待最后一个 microbatch 排空回流）。有 M 个 microbatch 和 N 个阶段时，每个阶段的气泡比例是 (N-1)/(M+N-1)。M=8、N=4 时是 27%。M=64、N=4 时是 4.5%。每步 microbatch 越多，气泡越小，这意味着每个 microbatch 的 batch size 要小，这个约束驱动了 microbatch 的设计。

## 概念

```mermaid
flowchart LR
  R0[rank 0: stage 0 / layer 0] --> R1[rank 1: stage 1 / layer 1]
  R1 --> R2[rank 2: stage 2 / layer 2]
  R2 --> R3[rank 3: stage 3 / loss]
  R3 -.backward.-> R2
  R2 -.backward.-> R1
  R1 -.backward.-> R0
```

### GPipe 调度

先让所有 M 个 microbatch 的 forward 填满流水线，再开始任何 backward；然后按逆序排空 backward。每个 microbatch 的激活必须保留到它的 backward，所以内存随 M 线性增长。Forward 需要 M+N-1 个周期，backward 再需要 M+N-1 个周期。每个阶段的有用工作是 2M 个周期；每个阶段的气泡是 2(N-1) 个周期。当每次 forward 和 backward 各占一个时间单位时，气泡比例是 (N-1)/(M+N-1)。取 M 远大于 N 可以隐藏气泡。

### 1F1B 调度

交错执行：一个 microbatch 的 forward 一到最后一个阶段，就开始它的 backward，让它回流。调度在每个阶段交替一次 forward 和一次 backward。气泡仍是 N-1，但激活内存受流水线深度约束，而不是 microbatch 数量。生产流水线使用 1F1B（Megatron、PipeDream）。本课先实现 GPipe，因为它更简单，并把 1F1B 留作练习。

### 为什么每个阶段计算量相等很重要

如果 stage 0 需要 50 ms，stage 1 需要 100 ms，每个周期都被 stage 1 卡住。其他阶段每个周期空闲 50 ms 等待 stage 1 释放。参数数量相等是错误的分轴：Transformer 的计算由每层的注意力和 MLP 主导，而 embedding 层参数多但计算少。阶段分配应该均衡每个阶段的 FLOPs，而不是每个阶段的权重数。

### Microbatch 与 batch

流水线运行 M 个大小为 B 的 microbatch。有效 batch size 是 M*B。流水线 step 结束时的梯度是合并后的 M*B 个样本上的梯度。气泡比例取决于 M；优化器看到的是 M*B。调 M 就是在气泡（M 大则更小）和单 microbatch 内存（GPipe 下 M 大则激活内存更高）之间权衡。

```figure
cd-pipeline-bubble
```

## 构建它

`code/main.py` 实现：

- `PipelineStage`：一个小的 `nn.Module`，持有一个阶段的参数并暴露 `forward(activation)`。
- `Pipeline(stages, num_microbatches)`：在模拟阶段上用模拟的每阶段墙钟时间编排 GPipe 调度。
- `bubble_fraction(num_stages, num_microbatches)`：闭式公式 (N-1)/(M+N-1)。
- 一个 4 阶段演示，打印逐 microbatch 轨迹和实测气泡比例。

运行：

```bash
python3 code/main.py
```

输出：阶段 × microbatch 的甘特图，以及气泡百分比与闭式预测的对比。

## 生产环境中的模式

有三种模式能让流水线并行达到可交付的强度。

**激活重计算与流水线配对。** GPipe 上有 M 个 microbatch 在飞时，激活内存是一个 microbatch 的 M 倍。激活 checkpointing 在 backward 时重算 forward，用计算换内存；这个组合让流水线对长序列变得可行。

**阶段均衡靠测量，不靠假设。** 生产团队会运行分析流程，在目标硬件上测量实际的逐层计算量（FLOPs 和墙钟时间），然后按测量结果分区。Megatron-LM 的 `--num-layers-per-stage` 标志接受一个列表，允许在阶段间逐层成本不同时使用不均衡的层数。

**收发调度必须避免死锁。** 如果每个阶段都先 send 再 recv，流水线会在线路上死锁。标准修复是交错：偶数 rank 的阶段先 send 再 recv，奇数 rank 的阶段先 recv 再 send。本课显式调度各 rank，让这种模式可见。

## 使用它

生产实现：

- **Megatron-LM。** 大规模流水线并行的参考。使用 1F1B，支持张量 + 流水线 + 数据并行组合。
- **DeepSpeed Pipeline。** 与 ZeRO 集成；ZeRO-1 + 流水线是最大的开源模型常用的组合。
- **PyTorch Pipe。** PyTorch 原生的流水线包装器，基于 `torch.distributed.pipeline.sync.Pipe`。

## 交付它

第 80 课把逐阶段参数分片存入分片 checkpoint。第 81 课在端到端演示中组合 DDP + ZeRO + 流水线（理念上如此；演示为了运行时长保持流水线模拟）。

## 练习

1. 实现 1F1B，验证气泡比例与 GPipe 相同但激活内存有界。
2. 在更深的模型上分析真实每阶段时间，并按实测墙钟时间重新平衡阶段。
3. 添加跨流水线 microbatch 的梯度累积，检查梯度等于等价全 batch forward 的梯度。
4. 将流水线与激活 checkpointing 配对，测量内存下降与计算成本。
5. 把流水线与 DDP 结合（每个流水线 rank 在数据并行组中复制），推理二维调度。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Pipeline | 「沿深度模型并行」 | 每个 rank 一个阶段，激活逐阶段流动 |
| Bubble | 「流水线空闲时间」 | 开始和结束各 (N-1) 步，部分阶段无事可做 |
| Microbatch | 「batch 的切片」 | 一个 forward/backward 单元；M 增大气泡缩小 |
| GPipe | 「先填后排」 | 所有 M 个 forward 后才 backward；激活内存高 |
| 1F1B | 「交错调度」 | 每阶段一 forward 一 backward；激活内存有界 |

## 延伸阅读

- [Huang et al, GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)
- [Narayanan et al, PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377)
- [Megatron-LM pipeline parallel docs](https://github.com/NVIDIA/Megatron-LM)
- 第 19 阶段第 76 课 - 调度使用的 send/recv 原语
- 第 19 阶段第 78 课 - ZeRO 与流水线正交，常被组合使用
