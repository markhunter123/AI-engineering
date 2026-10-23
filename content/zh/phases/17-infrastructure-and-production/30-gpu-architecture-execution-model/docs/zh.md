# GPU Architecture and the Execution Model

> 单张 H100 能以约 3.35 TB/s 的带宽吞吐显存数据，但它只有让每字节数据被重复利用数百次时，才能逼近约 990 TFLOPS 的 FP16 峰值。线程、warp、占用率与 roofline 模型，是读懂 vLLM 与 Triton 内部机制的前置词汇，也是解释「为什么增大 batch 能让 LLM 推理更快」的透镜。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0 · 03 (GPU Setup & Cloud), Phase 3 · 02 (Multi-Layer Networks)
**Time:** ~60 minutes

## 学习目标
- 对比 CPU 面向延迟的设计与 GPU 面向吞吐的设计，解释 GPU 通过并行而非大缓存来隐藏延迟。
- 梳理 CUDA 执行层级：thread → warp → block → grid；描述 SIMT 执行方式以及 warp 分支发散的代价。
- 计算理论占用率，并识别寄存器、共享内存与 block 大小对占用率的限制。
- 用 NumPy 搭建一个模拟器：给定 kernel 配置，输出占用率、算术强度、roofline 位置与估算运行时间。
- 运用 roofline 模型解释：为什么朴素矩阵乘是 memory-bound、分块矩阵乘如何降低流量、以及 decoder 生成阶段为何仍是 memory-bound。
- 预测为什么增大 batch size 会把 LLM 推理从 memory-bound 推向 compute-bound。

## 问题所在

生产级 LLM 推理里充斥着听起来像硬件八卦、直到突然决定你延迟账单的论断：「prefill 是 compute-bound，decode 是 memory-bound」「KV cache 吃显存」「小 batch 占用率低」「tensor-core 吞吐」。如果没有对 GPU 工作机制的第一性原理模型，这些话只是口号。

这个缺口体现在三个地方。

**你无法调试一个无法建模的 kernel。** 某个 Triton flash-attention kernel 跑不过 benchmark。问题究竟是每个 SM 的 warp 太少、每个 block 共享内存太大、还是算术强度低于 ridge point？不同诊断指向不同修复方向，瞎猜会浪费好几天。

**没有 roofline 估算，你无法合理规划推理部署。** 一个 70B 模型在 FP16 下约需 140 GB 权重。在 decode 阶段，每个新 token 都要把所有权重读一遍。因此决定单 GPU 每秒 token 数的往往是内存墙，而非 FLOP 数量——直到 batch size 大到足以摊平权重的读取。

**不掌握这套词汇，你读不懂 vLLM 或 SGLang 源码。** PagedAttention、RadixAttention、chunked prefill、speculative decoding，本质上都是对同一组底层事实的工程回应：warp 以 SIMT 方式执行、内存层级很关键、占用率与算术强度共同决定你是 compute-bound 还是 memory-bound。

本课从硬件向上构建这套模型，并提供一个模拟器：输入 kernel 配置，输出占用率、算术强度、roofline 位置与估算运行时间。

## 核心概念

### CPU 吞吐设计 vs GPU 吞吐设计

现代 CPU 为**延迟**优化。它拥有大缓存、分支预测、乱序执行，以及数量相对较少但功能强大的核心。当单个线程必须快速完成且工作集能放进缓存时，CPU 胜出。

GPU 为**吞吐**优化。它的运算单元数量是 CPU 的数百倍，但这些单元更简单、按序执行，并共享控制逻辑。GPU 通过同时保持成千上万个线程在飞来隐藏延迟：当一个 warp 等待显存时，调度器会换上一个操作数已就绪的 warp。大缓存会浪费本可用来放更多运算单元的芯片面积。设计赌注是：用并行而非缓存来覆盖延迟。

同一条指令以 SIMD/向量风格在许多线程上运行，但 CUDA 把这套模型称为**单指令多线程（SIMT）**：每个线程拥有自己的寄存器，可以走独立的控制路径，但硬件一次为整个 warp 执行一条指令。

### 流式多处理器与 H100 实例

GPU 由**流式多处理器（Streaming Multiprocessors, SMs）**组成。每个 SM 包含运算单元、调度器、寄存器和共享内存。NVIDIA H100 SXM5 GPU 大约有 **132 个 SM**（具体数值因 SKU 略有差异）。为让数字具体可感，本课使用如下近似 H100 规格：

| Resource | Approximate value |
|----------|-------------------|
| SMs | 132 |
| Warp schedulers per SM | 4 |
| Threads per warp | 32 |
| HBM3 bandwidth | ~3.35 TB/s |
| L2 cache | ~50 MB |
| Shared memory per SM | ~228 KB |
| FP16 dense peak (no sparsity) | ~989 TFLOP/s |

这些是取整后的 SKU 相关数值，仅作教学锚点，不要当数据手册用。

每个 SM 上的 warp scheduler 每个周期都能从另一个 warp 发射指令。4 个 scheduler、32 线程 per warp，单个 SM 可同时驻留超过 2,000 个线程。132 个 SM 合计可驻留数万个 warp。调度器挑选操作数就绪的 warp，因此驻留的独立 warp 越多，硬件能隐藏的延迟就越长。

### CUDA 执行层级：thread → warp → block → grid

CUDA kernel 以 **grid** 形式启动，grid 由 **block** 组成，每个 block 又包含 **thread**。

```
thread ⊂ warp ⊂ block ⊂ grid
```

- **Thread**：最小执行单元，拥有私有寄存器和线程 ID。
- **Warp**：32 个线程。SM 调度的是整个 warp，而非单个线程。
- **Block**：运行在同一个 SM 上的一组线程，可通过**共享内存**和 block 级同步协作。Block 内的线程被划分为 warp。
- **Grid**：一次 kernel 启动产生的所有 block 的集合。硬件将这些 block 分布到各个 SM 上。

线程 ID 用于数据索引：每个线程计算输出的一部分。当线程访问连续的内存地址时，硬件会把这些 load 合并成更少的 cache line，这是全局内存带宽最重要的优化。

### SIMT 执行与 warp 分支发散

在 SIMT 中，warp 内 32 个线程执行同一条指令，只是数据不同。如果线程在 `if` 语句中走向不同分支，硬件无法同时执行两个分支。它会先执行分支 A，屏蔽没走这条路的线程；再执行分支 B，屏蔽没走这条路的线程。这就是 **warp divergence（分支发散）**，它把 warp 串行化：

```math
T_{\text{warp, divergent}} = T_{\text{branch A}} + T_{\text{branch B}}
```

最坏情况下，一条发散的 `if/else` 会让该 warp 耗时翻倍。应尽量避免在内层循环里出现基于线程 ID 的条件逻辑。这也是 flash attention 这类 kernel 努力让 warp 内所有线程保持同一条控制路径的原因。

### 占用率

**占用率（Occupancy）**是 SM 上活跃 warp 数与 SM 能容纳的最大 warp 数之比：

```math
\text{occupancy} = \frac{\text{active warps per SM}}{\text{max warps per SM}}
```

更高的占用率通常能改善延迟隐藏，因为调度器有更多 warp 可选。但占用率不是唯一指标——高占用率但访存模式糟糕的 kernel 仍然会很慢。硬件通过以下几种方式限制占用率：

- **Block 内的线程数。** 一个 1,024 线程的 block 占 32 个 warp。若 SM 最多支持 2,048 线程，则最多只能放两个这样的 block。
- **每线程寄存器数。** 每个 SM 的寄存器堆大小固定。每线程使用大量寄存器会减少可同时驻留的 warp 数。编译器自动分配寄存器，模拟器里用每线程寄存器预算来建模。
- **每 block 共享内存。** 共享内存快但稀缺。一个 block 若占满 SM 的大部分共享内存，就会阻止其他 block 在该 SM 上启动。
- **每 SM 最大 block 数。** 即使线程和寄存器允许更多，硬件也对每 SM 的并发 block 数设有上限。

限制因素就是导致驻留 block 数最少的那一项资源。

### 内存层级与延迟

GPU 内存是按容量、带宽和延迟分层的：

| Level | Capacity (H100 scale) | Latency | Access |
|-------|----------------------|---------|--------|
| Registers | ~64K per SM | ~1 cycle | Private to thread |
| Shared memory (L1) | ~228 KB per SM | ~20–30 cycles | Block-shared, user-managed |
| L2 cache | ~50 MB | ~100–200 cycles | GPU-wide |
| HBM3 | Tens of GB | Hundreds of cycles | Global |

一次 load 走得越远，所需周期越长。如果一个 kernel 的 load 无法被寄存器或共享内存满足，只能反复访问 HBM，那它就是 memory-bound。**分块（tiling）**是标准对策：把一小块数据一次性加载进共享内存，在完成多次运算后再写回。

以矩阵乘法为例，朴素实现会为每个输出元素把对应的输入元素各读一次；分块实现则把一块输入加载进共享内存，然后复用它计算整块输出的一整行或一整列。流量减少量与 tile 大小成正比。

### Roofline 模型

**Roofline 模型**把可达到的性能画成峰值算力和内存带宽上限的较小值：

```math
\text{achievable FLOP/s} = \min\left(P, B \cdot I\right)
```

其中 $P$ 是峰值算力，$B$ 是内存带宽，$I$ 是**算术强度（arithmetic intensity）**，单位是每字节 FLOP：

```math
I = \frac{\text{FLOPs}}{\text{bytes moved from HBM}}
```

**Ridge point**是两条屋顶线的交点：

```math
I_{\text{ridge}} = \frac{P}{B}
```

用上面的 H100 数据：

```math
I_{\text{ridge}} = \frac{989.4 \times 10^{12}}{3.35 \times 10^{12}} \approx 295 \text{ FLOPs/Byte}
```

任何 $I < 295$ 的 kernel 都是 **memory-bound**（受 HBM 带宽限制）；任何 $I > 295$ 的 kernel 都是 **compute-bound**（受峰值算力限制）。这解释了：

- 朴素 FP32 矩阵乘算术强度为 $1/4 = 0.25$ FLOPs/Byte——深陷 memory-bound。
- 分块大小为 $T$ 的 FP32 矩阵乘强度约为 $T/4$——更好，但对普通 tile 仍是 memory-bound。
- 70B 模型 FP16 decode 每个参数约 2 FLOP，且每个参数只读一次，因此 $I = 1$ FLOP/Byte——在 batch 变大、复用提高之前强烈 memory-bound。

增大 batch 有帮助，因为同一份权重被读取一次后，可服务于 batch 中的每个 token。batch size 为 $B$ 时，decode 的算术强度大致随 $B$ 增长，最终会跨过 ridge point，把工作负载从 memory-bound 推向 compute-bound。

## 动手实现

本课提供一个纯 NumPy 的「GPU 执行财务模型」。你给定 kernel 配置——grid 与 block 形状、每线程 FLOPs、每线程全局字节、每 block 共享内存、寄存器预算——模拟器就会输出占用率、算术强度、roofline 限制与估算运行时间。

它还会对比朴素矩阵乘与共享内存分块矩阵乘，并打印 H100 的文本版 roofline 表。

**运行方式：**

```console
python3 main.py
```

你会看到 H100 的 ridge point（约 295 FLOPs/Byte）、分块带来的巨大流量下降、120 KB 共享内存对占用率的打击，以及 70B 模型 decode 阶段的算术强度。输出中的每个常数都对应 `main.py` 里的命名常量与本文档中的概念。

## 验证它

`code/tests/test_main.py` 锁定了硬件与模拟器的不变量：

- H100 ridge point 约为 295 FLOPs/Byte；
- CUDA warp 含 32 个线程；
- 朴素矩阵乘是 memory-bound，算术强度低于 1 FLOP/Byte；
- 分块按 tile 大小比例降低全局内存流量；
- 分块相对朴素版本提高算术强度；
- 过量共享内存会降低占用率；
- 高寄存器使用会降低占用率；
- decoder 生成阶段的算术强度远低于 ridge point。

## 对工程师为何重要

- **Roofline 模型把争论变成算术。** 在责怪框架之前，先算一算算术强度。如果你低于 ridge point，量化、分块或 batching 会有效；再加 FLOP 没用。
- **Decoder 从设计上是 memory-bound。** 自回归生成每个 token 都要读完整权重矩阵。因此 batching、speculative decoding、量化都针对内存带宽，而非原始算力。
- **占用率是延迟隐藏预算，不是速度旋钮。** 100% 占用率的 kernel 如果访存模式散乱，仍然很慢；50% 占用率的 kernel 如果每个 warp 都计算密集且合并访问，也可以很好。
- **分块是 memory-bound kernel 的第一优化。** 模拟器显示 32×32 tile 可降低 32 倍 HBM 流量。真实的 cuBLAS 与 CUTLASS 还有大量其它技巧，但分块是基础。
- **Batch size 是一个 roofline 旋钮。** 更大的 batch 摊平权重读取，把 decode 从 memory-bound 推向 compute-bound。这就是推理引擎即使要付出尾部延迟代价也要做 batching 的经济学原因。
- **vLLM、SGLang 与 Triton 都说同一套语言。** PagedAttention 减少 KV cache 的 HBM 流量；RadixAttention 复用 prompt 前缀；chunked prefill 平衡计算与内存；Triton kernel 调节 block 与 tile 大小以优化占用率。同一个模型能解释所有这些问题。

## 提示

- 如果占用率意外偏低，按这个顺序检查限制因素：block 大小、每 block 共享内存、每线程寄存器、每 SM 最大 block 数。
- Memory-bound kernel 的收益来自合并访问、分块进共享内存、通过 batching 提高算术强度——而不是更多 SM。
- 热循环里的发散分支可能比多算几个 FLOP 更贵；可能时优先用谓词化或调整数据布局。
- 小 batch 的 decoder 生成强烈 memory-bound；在跨过 ridge point 之前，吞吐会随 batch size 近似线性增长。

## 关键术语

| Term | People say | Actual meaning |
|------|-----------|----------------|
| SM | "streaming multiprocessor" | GPU 的核心构建块：调度器、运算单元、寄存器与共享内存。 |
| Warp | "32 threads" | CUDA 的调度单元；32 个线程在同一条指令上对不同数据执行。 |
| SIMT | "single instruction, multiple threads" | CUDA 执行模型：每个 warp 一个程序计数器，每个线程私有寄存器。 |
| Warp divergence | "branch divergence" | 当 warp 内线程走向不同分支时，硬件串行化两条路径。 |
| Occupancy | "SM utilization" | 每 SM 活跃 warp 数与该 SM 能容纳的最大 warp 数之比。 |
| Shared memory | "L1 / SMEM" | 快速、片上、block 作用域的暂存空间；分块的目的地。 |
| Arithmetic intensity | "FLOPs per byte" | 计算量与 HBM 流量之比，决定处于 roofline 的哪个区域。 |
| Ridge point | "roofline knee" | 峰值算力等于内存带宽时的算术强度：$P/B$。 |
| Memory-bound | "bandwidth-limited" | 可达到性能受 HBM 带宽限制，而非峰值 FLOP。 |
| Compute-bound | "math-limited" | 可达到性能受峰值 FLOP 限制，而非内存带宽。 |
