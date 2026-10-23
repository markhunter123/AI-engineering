# 内核融合、Triton 与性能剖析工作流

> 每一个未融合的算子都要向 HBM 支付一次往返。内核融合、像 Triton 这样的 tile 中心编译器，以及成体系的性能剖析工作流，是把内存受限循环变成计算受限循环的三根杠杆。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 17 · 30 (GPU Architecture and the Execution Model)
**Time:** ~75 minutes

## 学习目标

- 使用 bytes-in/bytes-out 模型，计算融合与非融合算子序列之间的 HBM 流量差异。
- 解释 Triton 的 tile 中心编程模型：program 网格、每 tile 掩码、`tl.load`/`tl.store` 与 shared memory。
- 预测 `torch.compile` 何时有效（静态形状、子图封闭）以及何时失效（动态形状、graph break）。
- 针对具体诊断场景，选择 Nsight Systems（timeline 层面的间隙与空泡）还是 Nsight Compute（单内核 occupancy、内存吞吐、SOL）。
- 把 FlashAttention 的 online softmax 技巧描述为算法级融合，将 attention 的 HBM 流量从 $O(N^2)$ 降到 $O(N)$。
- 校准 roofline 模型，判断目标 GPU 上融合后的内核是内存受限还是计算受限。

## 问题背景

GPU 算力持续增长的速度远快于内存带宽。现代 Tensor Core 每从 HBM 取回一个字节，就能发出数千次 FLOP，因此 HBM 成为大多数中小算子的瓶颈。

考虑最小表达式 $z = \operatorname{ReLU}(x + y)$，在峰值 FP16 吞吐 $100$ TFLOP/s、HBM 带宽 $2$ TB/s 的 GPU 上运行。每个元素只需要一次加法和一次 `max(0, a)`：

```math
\text{每个元素的 FLOPs} = 2, \qquad \text{每个元素读+写字节数} = 3 \times 2 = 6 \text{ bytes (FP16)}
```

如果你启动三个独立内核——`add`、`relu`、`store`——中间结果在每一步之间都要往返 HBM。算术强度极低，执行受内存限制。如果你把整个表达式融合成一个内核，中间结果就留在寄存器或 shared memory 里，只有最终输出写回 HBM。同样的 FLOP 执行了，但 HBM 流量锐减。

随着模型加深，这种差距被放大。一个 transformer block 是逐元素、归约和矩阵乘法算子的长链。没有融合时，每个 layer norm、激活、缩放和偏置都是一次独立的 HBM 往返。有了融合，整个子图可以在 SRAM 内执行。

问题不在于是否要融合，而在于如何融合：手写 CUDA、像 Triton 这样的 Python 编译器，还是像 `torch.compile` 这样的图编译器。每一层工具都在控制力、可移植性和开发速度之间做了不同取舍。本节课建立一个财务模型来权衡这些选择，并把它与能证明优化是否真正生效的性能剖析工作流联系起来。

## 核心概念

### 为什么融合重要：HBM 是瓶颈

把内核的算术强度定义为：

```math
I = \frac{\text{FLOPs}}{\text{往返 HBM 的字节数}}
```

roofline 模型预测时间为：

```math
T = \max\Big( \frac{\text{FLOPs}}{P_{\text{flops}}}, \frac{\text{字节数}}{P_{\text{mem}}} \Big)
```

其中 $P_{\text{flops}}$ 是峰值算力，$P_{\text{mem}}$ 是峰值 HBM 带宽。当 $I$ 高于 ridge point 时，内核是计算受限的：

```math
I_{\text{ridge}} = \frac{P_{\text{flops}}}{P_{\text{mem}}}
```

对于 H100，$P_{\text{flops}} \approx 989$ TFLOP/s（FP16 Tensor Core），$P_{\text{mem}} \approx 3.35$ TB/s，因此 $I_{\text{ridge}} \approx 295$ FLOPs/byte。大多数逐元素内核都远低于这个值。

以 $z = \operatorname{ReLU}(x + y)$ 为例，$N$ 个 FP16 元素：

| 方案 | 内核数 | HBM 读取 | HBM 写入 | 总字节 | 算术强度 |
|------|--------|----------|----------|--------|---------|
| 未融合 | 2（`add`、`relu`） | $4N$ | $4N$ | $8N$ | $2N / 8N = 0.25$ |
| 融合 | 1 | $2N$ | $2N$ | $4N$ | $2N / 4N = 0.5$ |

融合版本把 HBM 流量减半，算术强度翻倍。在 H100 上以 $3.35$ TB/s 计算，绝对时间从 $8N / 3.35 \times 10^{12}$ 秒降到 $4N / 3.35 \times 10^{12}$ 秒。计算项 $2N / 989 \times 10^{12}$ 可以忽略，因此内核仍是内存受限——但速度快了一倍。

对于 $K$ 个逐元素算子的链，未融合流量大约是张量的 $2K$ 次遍历，而融合后只有 $2$ 次。融合把 $K$ 次 HBM 往返变成 $1$ 次。

### Triton 心智模型：tile 中心编程

Triton 是一种 Python DSL，把 tile 级程序编译成 CUDA/SASS。程序员一次只考虑一个 tile；编译器负责把这些 tile 调度到 GPU 上，并处理内存布局、合并访问和 shared memory 分配。

执行模型是：

- 启动一个 program **网格**，通常每个输出 tile 一个 program。
- 每个 program 用 `tl.load` 读取输入 tile，对边界元素做 mask。
- 计算在 **shared memory** 或寄存器中完成。
- 每个 program 用 `tl.store` 写回输出 tile。

一个 row-softmax 内核可以说明这个模型。每个 program 处理注意力分数矩阵的一行，把一行加载到 SRAM，计算 max、减去 max、指数化、求和、归一化，然后写回。在 CUDA 里这需要手写全局内存偏移、线程到元素的映射、块归约原语，还要小心 shared memory 的 bank conflict。在 Triton 里它就像一个带向量操作的 tile 嵌套循环：

```python
# Conceptual Triton pseudocode for row-softmax (explanatory, not runnable).
@triton.jit
def row_softmax_kernel(out_ptr, in_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_start = in_ptr + row_idx * n_cols
    x = tl.load(row_start + col_offsets, mask=mask, other=-float('inf'))

    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(out_ptr + row_idx * n_cols + col_offsets, out, mask=mask)
```

核心思想：

- `tl.program_id(0)` 是行索引；编译器把行映射到 SM。
- `tl.arange` 创建 tile 的 lane 索引向量。
- `mask` 处理长度不是 `BLOCK_SIZE` 整数倍的行。
- `tl.load` 和 `tl.store` 操作的是 tile，而不是标量。
- shared memory 是隐式的：编译器在 program 工作期间把 tile 保留在快速内存中。

与手写 CUDA 相比：CUDA 里你必须计算全局内存偏移、决定线程到元素的映射、显式使用 `__shared__` 数组，并用 `__syncthreads` 同步线程。Triton 把这些决策推进编译器。代价是控制力变少；收益是用 20 行 Python 函数写出能媲美 200 行 CUDA 内核的性能。

### torch.compile：PyTorch 图的编译器

`torch.compile` 是 eager PyTorch 代码的图模式编译器。它把 Python 代码跟踪成 FX 图，应用优化，再下放到后端，例如 Inductor——后者会为大量算子生成 Triton 内核。

它能融合的：

- 逐元素算子序列（`add`、`mul`、`relu`、`gelu`、softmax temperature 等）。
- 图安全时的原地更新和 view 操作。
- 广播输入的逐元素归约。
- 内存规划：中间 buffer 可以被复用或消除。

什么会破坏它：

- **动态形状**：如果张量维度在调用之间变化，图必须重新编译。反复重编译抵消收益。
- **Graph break**：Python 控制流、依赖数据的 `item()` 调用、`print()`，以及许多自定义 C 扩展会强制编译器在该边界回退到 eager 模式。
- **数据依赖形状**：输出大小依赖张量值，例如 `unique` 或 `nonzero`。

工程师的工作流是：

1. 用 `@torch.compile` 或 `torch.compile(model)` 标记区域。
2. 用有代表性的形状做 warmup，触发编译。
3. 跑一遍性能剖析（Nsight Systems），对比内核数量和 HBM 流量。
4. 如果编译器反复重编译，加日志或 shape guard 并修复动态形状。

常见错误是打开 `torch.compile` 就默认它生效。编译器日志（`torch._logging.set_logs`）和 profiler 共同证明融合是否真的发生。

### 性能剖析工作流：先 Systems，后 Compute

NVIDIA 提供了两个不同层次的剖析工具。

**Nsight Systems** 捕获整个应用 timeline：CPU 线程、CUDA API 调用、内核启动、内存拷贝和同步。先用它。它能回答：

- 内核之间是否存在大间隙？通常意味着 CPU 启动开销、Python GIL 竞争或 graph break。
- GPU 是否在空转（bubbles）？空泡说明 CPU 喂 GPU 的速度不够。
- 是否有意外的 `cudaMemcpy`？这些常来自 CPU 与 GPU 之间的张量传输。
- 内核序列是否符合预期的融合图？一长串小内核而不是一个融合内核，说明融合失败。

**Nsight Compute** 详细检查单个内核：occupancy、寄存器压力、shared memory 用量、内存吞吐，以及峰值 FLOP 百分比（SOL）。第二步再用它，分析 Systems 标出的昂贵内核。它能回答：

- 这个内核是否内存受限？如果内存吞吐接近 GPU 峰值带宽而计算 SOL 很低，答案是肯定的。
- occupancy 是否受寄存器或 shared memory 限制？增大 tile 可能有帮助，或减少寄存器压力。
- 内核是否浪费了算力？occupancy 高但 SOL 低，说明指令混合低效或同步卡住了。

诊断顺序很重要：

1. **Nsight Systems** 找到瓶颈类别：CPU 启动、GPU 空泡，还是少数慢内核。
2. **Nsight Compute** 分析慢内核，决定是优化内存流量、occupancy 还是指令混合。
3. 修改后**重新跑 Systems**，确认瓶颈转移。

常见读数：

| 现象 | 含义 |
|------|------|
| 内核前后大间隙 | CPU 开销或 graph break |
| 内存吞吐接近峰值 | 内存受限；融合或分块会有帮助 |
| 内存吞吐低、计算 SOL 低 | 延迟受限或 occupancy 受限 |
| 内存吞吐低、计算 SOL 高 | 计算受限；内核状态健康 |

### FlashAttention：算法级融合

FlashAttention 是算法级融合的经典例子。标准 attention 计算：

```math
S = QK^T, \qquad P = \operatorname{softmax}(S), \qquad O = PV
```

朴素实现会把 $S$ 和 $P$ 都物化到 HBM，每个都是 $O(N^2)$。长序列时 HBM 流量为 $O(N^2)$，内存带宽占主导。

FlashAttention 把 softmax 重写成 online 归约。它沿序列维度分块，一次只在 SRAM 中保留一个 $Q$、$K$、$V$ 的 tile，并增量更新正在运行的 softmax 归一化。完整的 $N \times N$ 分数矩阵永远不会写回 HBM。HBM 流量变成 $O(N)$：把 $Q$、$K$、$V$ 各读一次，把 $O$ 写一次。

关键在于 softmax 的分解。设 $m_i = \max_{j \le i} x_j$，$\ell_i = \sum_{j \le i} e^{x_j - m_i}$。当新 tile 到达时，运行统计更新为：

```math
m_{\text{new}} = \max(m_{\text{old}}, m_{\text{tile}})
```

```math
\ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \ell_{\text{old}} + e^{m_{\text{tile}} - m_{\text{new}}} \ell_{\text{tile}}
```

这让内核能在一次融合遍历中完成 attention，且 SRAM 工作集很小。同样的原则——把中间结果保留在快速内存中，只把大张量流式过一遍——也驱动了现代卷积、归一化和 MoE 路由的融合内核。

## 动手实现

本节课提供一个基于 NumPy 的"内核融合财务模型"。它把每个算子表示为 `(FLOPs, bytes_in, bytes_out)` 元组，计算融合与非融合的 HBM 流量，估算 roofline 时间，模拟带 shared memory 命中统计的 tile 调度器，并打印文本 timeline。

不需要真正的 GPU 代码。这个模拟器教授的正是 profiler 揭示的决策：哪里浪费了 HBM 流量、某个内核是否内存受限、以及分块如何改变画面。

运行方式：

```console
python3 main.py
```

你会看到一个小 transformer 子图的未融合与融合方案的 roofline 表、tile 调度器报告，以及一段模仿 Nsight Systems trace 的文本 timeline。

## 验证

`code/tests/test_main.py` 锁住了财务模型：

- 融合后的流量从不大于未融合流量，`relu(add)` 示例呈现预期的 2 倍流量缩减；
- roofline 模型在 ridge point 处正确区分内存受限与计算受限内核；
- tile 调度器产生确定性的 program 网格和 shared memory 命中数；
- SRAM 溢出阈值会把中间流量重新加回融合成本；
- 融合比例与 ridge point 公式与手算结果一致。

## 对工程师为什么重要

- **融合是内存受限循环的首优优化。** 在买更多带宽之前，先消除不必要的 HBM 往返。
- **Triton 是自定义内核的甜点。** 它用 Python 的易用性换来接近 CUDA 的速度；只有当编译器的 tile 选择失败时，才需要手写 CUDA。
- **`torch.compile` 不是魔法。** 验证编译后的图，警惕动态形状导致的重编译，并剖析结果。
- **先 Systems 后 Compute。** Compute 里看到的慢内核，可能是 Systems 里看到的 CPU 启动开销或空泡的症状。
- **FlashAttention 改变了设计空间。** online softmax 不只是 attention 技巧；它是一种模板，只要流式归约能替代物化中间结果，就适用。

## 提示

- 如果融合后的流量没有比未融合低多少，检查融合区域是否因超出 SRAM 而把中间张量 spill 回 HBM。
- 同一个内核在 A 卡上可能是计算受限，在 B 卡上可能是内存受限。始终为目标硬件计算 ridge point。
- 动态形状会导致 `torch.compile` 重编译；用 `TORCH_LOGS="+graph_breaks"` 记录 graph break。
- Nsight Systems 里内核之间的大间隙通常指向 Python 开销，而不是内核低效。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|---------|---------|
| Kernel fusion | "合并内核" | 把多个算子合并成一个内核，使中间结果留在 SRAM 而不是 HBM。 |
| Arithmetic intensity | "每字节 FLOP" | FLOPs 除以 HBM 移动字节数；与 roofline ridge point 比较。 |
| Roofline model | "瓶颈天花板" | $T = \max(\text{FLOPs}/P_{\text{flops}}, \text{bytes}/P_{\text{mem}})$。 |
| Triton | "Python 版 CUDA" | tile 中心 DSL，每个 program 处理一个 tile，编译器管理布局与 shared memory。 |
| `torch.compile` | "让 PyTorch 变快" | 融合子图并把逐元素内核下放到 Triton 的图模式编译器。 |
| Graph break | "编译器退出" | `torch.compile` 无法跟踪 Python 代码，回退到 eager 模式的点。 |
| Nsight Systems | "timeline 剖析器" | 用于 CPU/GPU timeline、间隙与空泡的全应用剖析器。 |
| Nsight Compute | "内核剖析器" | 用于 occupancy、内存吞吐和 SOL 的单内核剖析器。 |
| FlashAttention | "快速 attention" | 用 online softmax 避免物化 $N \times N$ 分数矩阵的融合 attention。 |
| Online softmax | "流式 softmax" | 增量更新 max 和指数和，使 softmax 能一次性遍历 tile 完成。 |
