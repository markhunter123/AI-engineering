# 数值稳定性

> 浮点数是一种有漏洞的抽象。训练时它一定会让你吃苦头，而且你还无从察觉。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~120 minutes

## 学习目标

- 使用「最大值减法」技巧实现数值稳定的 softmax 与 log-sum-exp
- 识别浮点计算中的溢出（overflow）、下溢（underflow）和灾难性抵消（catastrophic cancellation）
- 使用中心有限差分验证解析梯度与数值梯度
- 解释为什么训练时优先使用 bfloat16 而非 float16，以及 loss scaling 如何防止梯度下溢

## 问题

你的模型训练了三个小时，然后损失变成 NaN。你加了一条打印语句。第 9,000 步时 logits 还正常，第 9,001 步就变成了 `inf`。到第 9,002 步，所有梯度都是 `nan`，训练彻底死掉。

或者：你的模型训练完了，但准确率比论文低 2%。你检查了所有地方。架构一致，超参数一致，数据一致。问题是论文用了 float32，而你用了 float16 却没有做正确的缩放。累计的舍入误差悄悄吃掉了你的准确率。

或者：你从头实现了交叉熵损失。小 logits 时工作正常，logits 超过 100 就返回 `inf`。softmax 溢出了，因为 `exp(100)` 已经超出了 float32 能表示的范围。每个 ML 框架都用一个两行代码的技巧来处理这个问题，而你不知道这个技巧存在。

数值稳定性不是理论上的担忧。它决定了一次训练是成功还是静默失败。你最终要调试的每一个严重 ML bug，归根结底都是浮点数的问题。

## 概念

### IEEE 754：计算机如何存储实数

计算机按照 IEEE 754 标准把实数存成浮点数。一个浮点数有三部分：符号位、指数和尾数（significand）。

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

尾数决定精度（有多少位有效数字），指数决定范围（能表示多大或多小的数）。

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 大约有 7 位十进制精度。它能区分 1.0000001 和 1.0000002，但区分不了 1.00000001 和 1.00000002。7 位之后全是舍入噪声。

float16 大约只有 3 位精度。它能表示的最大值是 65,504。对 ML 来说这小得可怕，因为 logits、梯度和激活值 routinely 会超过这个数。

bfloat16 是 Google 对 float16 范围问题的回答。它有和 float32 一样的 8 位指数（范围同样达到 3.4e38），但只有 7 位尾数（精度比 float16 还低）。对神经网络训练来说，范围比精度更重要，因此 bfloat16 通常是赢家。

### 为什么 0.1 + 0.2 ≠ 0.3

数字 0.1 在二进制浮点中无法精确表示。以 2 为底时，它是无限循环小数：

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

float32 把它截断到 23 位尾数，存储值约等于 0.100000001490116。同样，0.2 存储为约 0.200000002980232。它们的和是 0.300000004470348，而不是 0.3。

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对 ML 的影响在于：

1. 像 `if loss < threshold` 这样的损失比较可能给出错误结果
2. 累加大量小值（数千步的梯度更新）会偏离真实和
3. 校验和与可复现性测试如果用 `==` 比较浮点数会失败

修复方法：永远不要用 `==` 比较浮点数。使用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性抵消

当你用两个几乎相等的浮点数相减时，有效数字会相互抵消，剩下的只有被提升到首位的舍入噪声。

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

一次减法就带来 19% 的相对误差。在 ML 中，这种情况发生在：

- 计算大均值数据的方差：当 $\mathbb{E}[x]$ 很大时，$\mathbb{E}[x^2] - \mathbb{E}[x]^2$
- 减去几乎相等的对数概率
- 用太小的 epsilon 计算有限差分梯度

修复方法：重新排列公式，避免用两个大且几乎相等的数相减。对方差使用 Welford 算法，或先对数据中心化。对对数概率，全程在对数空间内运算。

### 溢出与下溢

溢出指结果太大而无法表示。下溢指结果太小（比最小可表示正数更接近 0）。

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

`exp()` 函数是 ML 中溢出的主要来源：

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()` 函数则朝另一个方向出问题：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

在 ML 中，`exp()` 出现在 softmax、sigmoid 和概率计算中。`log()` 出现在交叉熵、对数似然和 KL 散度中。`log(exp(x))` 这个组合如果没有正确技巧，就是一片雷区。

### Log-Sum-Exp 技巧

直接计算 `log(sum(exp(x_i)))` 在数值上很危险。只要某个 `x_i` 很大，`exp(x_i)` 就会溢出。如果所有 `x_i` 都非常负，每个 `exp(x_i)` 都会下溢成 0，而 `log(0)` 是 `-inf`。

技巧：在取指数前先减去最大值。

```math
\log\left(\sum_i \exp(x_i)\right) = \max(x) + \log\left(\sum_i \exp(x_i - \max(x))\right)
```

为什么有效：减去 `max(x)` 后，最大的指数是 `exp(0) = 1`，不可能溢出。求和里至少有一项是 1，所以总和至少为 1，而 `log(1) = 0`，不可能下溢成 `-inf`。

证明：

```math
\begin{aligned}
& \log\left(\sum_i \exp(x_i)\right) \\
&= \log\left(\sum_i \exp(x_i - c + c)\right) && \text{(add and subtract } c\text{)} \\
&= \log\left(\sum_i \exp(x_i - c) \cdot \exp(c)\right) && (\exp(a+b) = \exp(a) \cdot \exp(b)) \\
&= \log\left(\exp(c) \cdot \sum_i \exp(x_i - c)\right) && \text{(factor out } \exp(c)\text{)} \\
&= c + \log\left(\sum_i \exp(x_i - c)\right) && (\log(a \cdot b) = \log(a) + \log(b))
\end{aligned}
```

令 `c = max(x)`，溢出就被消除了。

这个技巧在 ML 中无处不在：
- Softmax 归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合模型
- 变分推断

### 为什么 Softmax 需要最大值减法技巧

Softmax 把 logits 转换成概率：

```math
\mathop{\mathrm{softmax}}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
```

不用这个技巧，logits 为 [100, 101, 102] 时就会溢出：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

用上技巧，减去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率完全相同，但计算是安全的。这不是优化，而是正确性的必要条件。

### NaN 与 Inf：检测与预防

`nan`（Not a Number）和 `inf`（infinity）会在计算中病毒式传播。一次梯度更新中出现一个 `nan`，权重就变成 `nan`，随后的所有输出都变成 `nan`。一步之内训练就死了。

`inf` 如何出现：
- 对很大的正数取 `exp()`
- 除以零：`1.0 / 0.0`
- float32 累加溢出

`nan` 如何出现：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 对负数取 `sqrt()`
- 对负数取 `log()`
- 任何涉及已有 `nan` 的运算

检测：

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

预防策略：

1. 把 `exp()` 的输入限制在范围内：`exp(clamp(x, -80, 80))`
2. 分母加 epsilon：`x / (y + 1e-8)`
3. `log()` 内部加 epsilon：`log(x + 1e-8)`
4. 使用稳定实现（log-sum-exp、稳定 softmax）
5. 梯度裁剪防止权重爆炸
6. 调试时每次前向传播后检查 `nan`/`inf`

### 数值梯度检查

解析梯度（来自反向传播）可能有 bug。数值梯度检查用有限差分计算梯度来验证它们。

中心差分公式：

```math
\frac{df}{dx} \approx \frac{f(x + h) - f(x - h)}{2h}
```

这是 $O(h^2)$ 精度，比前向差分 $(f(x+h) - f(x)) / h$ 的 $O(h)$ 好得多。

选择 $h$：太大近似会错，太小则灾难性抵消会毁掉结果。典型值是 `1e-5` 到 `1e-7`。

检查方法：计算解析梯度与数值梯度的相对差异。

```math
\text{relative\_error} = \frac{\lvert \text{grad}_{\text{analytical}} - \text{grad}_{\text{numerical}} \rvert}{\max(\lvert \text{grad}_{\text{analytical}} \rvert, \lvert \text{grad}_{\text{numerical}} \rvert, 10^{-8})}
```

经验法则：
- relative_error < 1e-7：完美，梯度正确
- relative_error < 1e-5：可接受，很可能正确
- relative_error > 1e-3：有问题
- relative_error > 1：梯度完全错误

实现新层或新损失函数时总要检查梯度。PyTorch 提供了 `torch.autograd.gradcheck()` 来做这件事。

### 混合精度训练

现代 GPU 有专用硬件（Tensor Core），计算 float16 矩阵乘法比 float32 快 2-8 倍。混合精度训练利用这一点：

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

纯 float16 训练的问题：梯度通常非常小（1e-8 或更小）。float16 把低于约 6e-8 的任何值下溢成 0。你的模型停止学习，因为所有梯度更新都是 0。

修复方法是 loss scaling：

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

动态 loss scaling 会自动调整缩放因子。从较大值（65536）开始，如果梯度溢出成 `inf` 就减半，如果连续 $N$ 步没有溢出就翻倍。

### bfloat16 vs float16：为什么 bfloat16 更适合训练

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 精度更高（10 位尾数 vs 7 位），但范围有限（最大约 65,504）。bfloat16 精度较低，但范围和 float32 一样（最大约 3.4e38）。

对神经网络训练：

- 激活值和 logit 在训练尖峰时经常超过 65,504。float16 会溢出，bfloat16 能应付。
- float16 需要 loss scaling，而 bfloat16 通常不需要，因为它的范围覆盖了梯度量级谱。
- bfloat16 是对 float32 的简单截断：丢掉尾数的低 16 位。转换很简单，指数部分无损失。

float16 更适合推断，因为此时数值有界且精度更重要。bfloat16 更适合训练，因为此时范围更重要。这就是 TPU 和现代 NVIDIA GPU（A100、H100）原生支持 bfloat16 的原因。

### 梯度裁剪

梯度爆炸发生在梯度在许多层中指数级增长时（在 RNN、深层网络和 Transformer 中常见）。一个巨大的梯度会在一步内破坏所有权重。

两种裁剪方式：

**按值裁剪：** 独立地夹住每个梯度元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单，但会改变梯度向量的方向。

**按范数裁剪：** 缩放整个梯度向量，使其范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保持梯度方向。这就是 `torch.nn.utils.clip_grad_norm_()` 做的事，也是标准选择。

典型值：Transformer 用 `max_norm=1.0`，RL 用 `max_norm=0.5`，更简单的网络用 `max_norm=5.0`。

梯度裁剪不是 hack，而是一种安全机制。没有它，一个异常 batch 产生的梯度就可能毁掉数周的训练。

### 归一化层作为数值稳定器

Batch Normalization、Layer Normalization 和 RMS Normalization 通常被介绍为帮助训练收敛的正则化方法，但它们也是数值稳定器。

没有归一化时，激活值会在各层之间指数级增长或缩小：

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

归一化在每一层重新居中并缩放激活值：

```math
\mathop{\mathrm{LayerNorm}}(x) = \frac{x - \mathop{\mathrm{mean}}(x)}{\mathop{\mathrm{std}}(x) + \epsilon} \cdot \gamma + \beta
```

`epsilon`（通常 1e-5）防止所有激活值相同时除以零。可学习参数 `gamma` 和 `beta` 让网络恢复它需要的任何尺度。

这让整网的值都保持在数值安全范围内，既防止前向传播溢出，也防止反向传播梯度爆炸。

### 常见 ML 数值 Bug

**Bug：训练几个 epoch 后损失变成 NaN。**
原因：logits 太大，softmax 溢出；或者学习率太高，权重发散。
修复：使用稳定 softmax（最大值减法）、降低学习率、加梯度裁剪。

**Bug：损失卡在 log(num_classes) 不动。**
原因：模型输出接近均匀概率。通常意味着梯度消失或模型根本没在学。
修复：检查数据标签是否正确、验证损失函数、检查 dead ReLU。

**Bug：验证准确率比预期低 1-3%。**
原因：混合精度没有正确的 loss scaling。梯度下溢悄悄把小更新清零。
修复：开启动态 loss scaling，或切换到 bfloat16。

**Bug：某些层的梯度范数为 0.0。**
原因：dead ReLU 神经元（所有输入为负），或 float16 下溢。
修复：使用 LeakyReLU 或 GELU、使用梯度缩放、检查权重初始化。

**Bug：模型在一块 GPU 上正常，在另一块上结果不同。**
原因：浮点累加顺序不确定。GPU 并行归约求和在不同硬件上顺序不同，而浮点加法不满足结合律。
修复：接受微小差异（1e-6），或设置 `torch.use_deterministic_algorithms(True)` 并承担速度损失。

**Bug：损失计算中 `exp()` 返回 `inf`。**
原因：原始 logits 直接传入 `exp()`，没有最大值减法技巧。
修复：使用内部实现了 log-sum-exp 的 `torch.nn.functional.log_softmax()`。

**Bug：从 float32 切换到 float16 后训练发散。**
原因：float16 无法表示低于 6e-8 的梯度量级，也无法表示高于 65,504 的激活值。
修复：使用带 loss scaling 的混合精度（AMP），或改用 bfloat16。

```figure
logsumexp-stability
```

## 构建它

### 第 1 步：演示浮点精度限制

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 第 2 步：实现朴素版与稳定版 softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### 第 3 步：实现稳定 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### 第 4 步：实现稳定交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### 第 5 步：梯度检查

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 使用它

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

完整实现（包含所有边界情况演示）请见 `code/numerical.py`。

## 交付它

本节课产出：
- `code/numerical.py`：稳定的 softmax、log-sum-exp、交叉熵、梯度检查与混合精度模拟
- `outputs/prompt-numerical-debugger.md`：用于诊断训练中的 NaN/Inf 和数值问题

这些稳定实现会在第三阶段构建训练循环和第四阶段实现注意力机制时再次用到。

## 练习

1. **灾难性抵消。** 在 float32 中用朴素公式 $\mathbb{E}[x^2] - \mathbb{E}[x]^2$ 计算 [1000000.0, 1000001.0, 1000002.0] 的方差。然后用 Welford 在线算法计算。将两者与真实方差 0.6667 比较误差。

2. **精度探寻。** 找出最小的正 float32 值 `x`，使得在 Python 中 `1.0 + x == 1.0`。这就是机器 epsilon。验证它等于 `numpy.finfo(numpy.float32).eps`。

3. **Log-sum-exp 边界情况。** 用以下输入测试你的 `logsumexp_stable` 函数：(a) 所有值相等，(b) 一个值远大于其他值，(c) 所有值都非常负（-1000）。验证它在朴素版本失败的地方仍能给出正确结果。

4. **神经网络层的梯度检查。** 实现一个线性层 `y = Wx + b` 及其解析反向传播。对一个 3x2 的权重矩阵使用 `numerical_gradient` 验证正确性。

5. **Loss scaling 实验。** 模拟 float16 训练：生成范围 [1e-9, 1e-3] 的随机梯度，转成 float16，测量有多少比例变成 0。然后应用 loss scaling（乘以 1024），转成 float16，再缩放回原来尺度，再次测量变成 0 的比例。

### 提示

**练习 1。** 平方前先用 `np.float32` 转换三个值，然后调用 `variance_naive` 和 `welford_variance`。float32 中接近 $10^{12}$ 的值的间距远大于 1，因此可以预期朴素结果会远离 0.6667，而 Welford 结果会很接近。

**练习 2。** 从 `x = np.float32(1.0)` 开始，不断减半，直到 `np.float32(1.0) + x != np.float32(1.0)` 不再成立。全程保持 float32，然后把最后一个还能改变和的值与 `numpy.finfo(numpy.float32).eps`（即 $2^{-23}$）比较。

**练习 3。** 先预测每个答案。对于 $n$ 个相等的值 $c$，结果是 $c + \ln n$。对于一个远大于其他值的项，结果接近该值。对于接近 -1000 的值，结果接近 $-1000 + \ln n$，而 `logsumexp_naive` 会因为每个指数都下溢成 0 而返回 `-inf`。

**练习 4。** 对于损失 $L$ 下的 $y = Wx + b$，$W$ 的梯度是上游梯度与 $x$ 的外积。把损失写成展平 $W$ 的函数，调用 `numerical_gradient`，再把两个结果传给 `check_gradient`。

**练习 5。** 用 `10 ** np.random.uniform(-9, -3, n)` 在对数尺度上采样量级，然后用 `np.float16` 转换。float16 无法保留约 $6 \times 10^{-8}$ 以下的值，因此缩放前大约有四分之一会变成 0，乘以 1024 后几乎没有。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| IEEE 754 | 「浮点标准」 | 定义二进制浮点格式、舍入规则和特殊值（inf、nan）的国际标准。每台现代 CPU 和 GPU 都实现了它。 |
| Machine epsilon | 「精度极限」 | 在给定浮点格式下，使 $1.0 + e \ne 1.0$ 的最小值 $e$。对 float32 约为 1.19e-7。 |
| Catastrophic cancellation | 「减法导致的精度损失」 | 用两个几乎相等的浮点数相减时，有效数字抵消，舍入噪声主导结果。 |
| Overflow | 「数字太大」 | 结果超出可表示的最大值，变成 inf。exp(89) 会溢出 float32。 |
| Underflow | 「数字太小」 | 结果比最小可表示正数更接近 0，变成 0.0。exp(-104) 会下溢成 float32 的 0。 |
| Log-sum-exp trick | 「先减最大值」 | 通过提出 $\exp(\max(x))$ 来计算 $\log(\sum \exp(x))$，防止溢出和下溢。用于 softmax、交叉熵和对数概率运算。 |
| Stable softmax | 「不会爆炸的 softmax」 | 指数化前先减去 $\max(\text{logits})$。数值结果完全相同，但不可能溢出。 |
| Gradient checking | 「验证你的反向传播」 | 把反向传播的解析梯度与有限差分的数值梯度比较，以发现实现错误。 |
| Mixed precision | 「float16 前向，float32 反向」 | 对速度关键操作使用低精度浮点，对数值敏感操作使用高精度浮点。典型加速 2-3 倍。 |
| Loss scaling | 「防止梯度下溢」 | 反向传播前把损失乘以一个大常数，使梯度留在 float16 可表示范围内，然后在更新权重前除以同一个常数。 |
| bfloat16 | 「Brain floating point」 | Google 的 16 位格式，8 位指数（与 float32 同范围），7 位尾数（精度低于 float16）。训练时更常用。 |
| Gradient clipping | 「限制梯度范数」 | 缩放梯度向量，使其范数不超过阈值。防止梯度爆炸破坏权重。 |
| NaN | 「Not a Number」 | 未定义操作（0/0、inf-inf、sqrt(-1)）产生的特殊浮点值。会在后续所有运算中传播。 |
| Inf | 「Infinity」 | 溢出或除以零产生的特殊浮点值。相互组合可产生 NaN（inf - inf、inf * 0）。 |
| Numerical gradient | 「暴力求导」 | 通过计算 $f(x+h)$ 和 $f(x-h)$ 再除以 $2h$ 来近似导数。慢，但用于验证时很可靠。 |

## 延伸阅读

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) —— 权威参考，密集但完整
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) —— 为 float16 训练引入 loss scaling 的 NVIDIA 论文
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) —— PyTorch 混合精度实践指南
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) —— Google 为 TPU 选择该格式的原因
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) —— 减少浮点数求和舍入误差的算法
