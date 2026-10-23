# GPU 设置与云环境

> 用 CPU 训练足以学习。真正的训练需要 GPU。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## 学习目标

- 使用 `nvidia-smi` 和 PyTorch 的 CUDA API 验证本地 GPU 可用性
- 在 Google Colab 中配置 T4 GPU 进行免费云实验
- 在 CPU 与 GPU 上对比矩阵乘法并测量加速比
- 使用 fp16 经验法则估算能装入 VRAM 的最大模型

## 问题

阶段 1-3 的大部分课程在 CPU 上运行良好。但一旦开始训练 CNN、transformer 或 LLM（阶段 4+），就需要 GPU 加速。CPU 上需要 8 小时的训练在 GPU 上可能只需要 10 分钟。

你有三种选择：本地 GPU、云 GPU 或 Google Colab（免费）。

## 概念

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

```figure
s0-gpu-dispatch
```

## 构建它

### 选项 1：本地 NVIDIA GPU

检查是否拥有：

```bash
nvidia-smi
```

安装带 CUDA 的 PyTorch：

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 选项 2：Google Colab

1. 访问 [colab.research.google.com](https://colab.research.google.com)
2. 运行时 > 更改运行时类型 > T4 GPU
3. 运行 `!nvidia-smi` 进行验证

直接上传本课程的 notebook 到 Colab。

### 选项 3：云 GPU

对于 Lambda Labs、RunPod 或 Vast.ai：

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 没有 GPU？没问题

大多数课程都能在 CPU 上运行。需要 GPU 的课程会说明并提供 Colab 链接。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 构建它：GPU 与 CPU 基准测试

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 练习

1. 运行上面的基准测试，比较 CPU 与 GPU 耗时
2. 如果你没有 GPU，在 Google Colab 上运行并比较
3. 查看 GPU 显存大小，并估算能装入的最大模型（经验法则：fp16 每个参数占 2 字节）

### 提示

**练习 1。** 原样运行「构建它」部分的代码，然后记下 CPU 时间、GPU 时间和加速比。再运行一次；第一次 GPU 运行包含启动开销，所以第二次的数字才具有可比性。

**练习 2。** 在 Colab 中，选择 运行时 > 更改运行时类型 > GPU，然后把基准测试粘贴到一个 cell 中。先在另一个 cell 中运行 `!nvidia-smi`；它必须显示 GPU 名称，基准测试结果才有意义。

**练习 3。** 像 `code/gpu_check.py` 中那样，用 `torch.cuda.get_device_properties(0).total_memory / 1e9` 读取总显存。用字节数除以 2 得到 fp16 可承载的参数数量，但要留出余量，因为激活值和 CUDA 上下文也会占用显存。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| CUDA | "GPU programming" | NVIDIA 的并行计算平台，让你在 GPU 上运行代码 |
| VRAM | "GPU memory" | GPU 上的显存，与系统内存独立。它限制了模型大小 |
| fp16 | "Half precision" | 16 位浮点数，占用 fp32 一半的内存，精度损失很小 |
| Tensor Core | "Fast matrix hardware" | 专门用于矩阵乘法的 GPU 核心，比普通核心快 4-8 倍 |
