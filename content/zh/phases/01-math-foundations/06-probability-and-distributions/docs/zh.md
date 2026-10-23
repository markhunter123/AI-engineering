# 概率与分布

> 概率是 AI 表达不确定性的语言。

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 minutes

## 学习目标

- 从零实现 Bernoulli、categorical、Poisson、uniform 和 normal 分布的 PMF 与 PDF
- 计算期望值、方差，并用中心极限定理解释为何高斯分布占据主导
- 用数值稳定技巧（减去最大 logit）构建 softmax 和 log-softmax 函数
- 从 logits 计算交叉熵损失，并将其与负对数似然联系起来

## 问题

分类器输出 `[0.03, 0.91, 0.06]`。语言模型从 50,000 个候选中挑选下一个词。扩散模型通过对学习到的分布采样来生成图像。这些都是概率在起作用。

模型的每个预测都是一个概率分布。每个损失函数都在度量预测分布与真实分布的距离。每个训练步骤都在调整参数，使一个分布更像另一个。没有概率，你无法读懂任何 ML 论文，无法调试任何模型，也无法理解训练损失为何变成 NaN。

## 概念

### 事件、样本空间与概率

样本空间 $S$ 是所有可能结果的集合。事件是样本空间的子集。概率将事件映射到 0 到 1 之间的数。

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

三条公理定义了全部概率：
1. 对任意事件 $A$，$P(A) \ge 0$
2. $P(S) = 1$（总有某件事发生）
3. 当 $A$ 与 $B$ 不能同时发生时，$P(A \text{ or } B) = P(A) + P(B)$

其他一切（贝叶斯定理、期望、分布）都源自这三条规则。

### 条件概率与独立性

$P(A \mid B)$ 是在 $B$ 发生条件下 $A$ 的概率。

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

两个事件独立，是指知道其中一个对另一个毫无提示：

```math
\begin{aligned}
\text{Independent:} \quad & P(A \mid B) = P(A) \\
\text{Equivalent to:} \quad & P(A \text{ and } B) = P(A) \cdot P(B)
\end{aligned}
```

抛硬币是独立的。不放回抽牌不是。

### 概率质量函数 vs 概率密度函数

离散随机变量具有概率质量函数（PMF）。每个结果都有可直接读取的特定概率。

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

连续随机变量具有概率密度函数（PDF）。单点处的密度不是概率。概率来自对区间上的密度积分。

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

这种区分在 ML 中很重要。分类输出是 PMF（离散选择）。VAE 隐空间使用 PDF（连续）。

### 常见分布

**Bernoulli：** 一次试验，两种结果。用于建模二分类。

```math
\begin{aligned}
P(X = 1) &= p \\
P(X = 0) &= 1 - p \\
\text{Mean} = p, \quad & \text{Variance} = p(1-p)
\end{aligned}
```

**Categorical：** 一次试验，$k$ 种结果。用于建模多分类（softmax 输出）。

```math
\begin{aligned}
& P(X = i) = p_i, \quad \text{where } \textstyle\sum_i p_i = 1 \\
& \text{Example: } P(\text{cat}) = 0.7, \quad P(\text{dog}) = 0.2, \quad P(\text{bird}) = 0.1
\end{aligned}
```

**Uniform：** 所有结果等可能。用于随机初始化。

```math
\begin{aligned}
\text{Discrete:} \quad & P(X = k) = 1/n \text{ for } k \in \lbrace 1, \ldots, n \rbrace \\
\text{Continuous:} \quad & f(x) = 1/(b-a) \text{ for } x \in [a, b]
\end{aligned}
```

**Normal（Gaussian）：** 钟形曲线。由均值（$\mu$）和方差（$\sigma^2$）参数化。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson：** 固定区间内稀有事件的计数。用于建模事件率。

```math
\begin{aligned}
P(X = k) &= \frac{\lambda^k e^{-\lambda}}{k!} \\
\text{Mean} = \lambda, \quad & \text{Variance} = \lambda
\end{aligned}
```

### 期望值与方差

期望值是加权平均结果。

```math
\begin{aligned}
\text{Discrete:} \quad & \mathbb{E}[X] = \sum_i x_i \, P(X = x_i) \\
\text{Continuous:} \quad & \mathbb{E}[X] = \int x \, f(x) \, dx
\end{aligned}
```

方差度量围绕均值的离散程度。

```math
\begin{aligned}
\mathrm{Var}(X) &= \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 \\
\text{Standard deviation} &= \sqrt{\mathrm{Var}(X)}
\end{aligned}
```

在 ML 中，期望值体现为损失函数（数据分布上的平均损失）。方差告诉你模型的稳定性。梯度方差高意味着训练噪声大。

### 联合分布与边际分布

联合分布 $P(X, Y)$ 同时描述两个随机变量。

联合 PMF 示例（$X$ = 天气，$Y$ = 是否带伞）：

| | $Y=0$（no umbrella） | $Y=1$（umbrella） | Marginal $P(X)$ |
|---|---|---|---|
| $X=0$（sun） | 0.40 | 0.10 | $P(X=0) = 0.50$ |
| $X=1$（rain） | 0.05 | 0.45 | $P(X=1) = 0.50$ |
| **Marginal $P(Y)$** | $P(Y=0) = 0.45$ | $P(Y=1) = 0.55$ | 1.00 |

边际分布通过对另一个变量求和得到：

```math
P(X = x) = \sum_{y} P(X = x, Y = y)
```

上表的行合计与列合计就是边际分布。

### 为什么正态分布无处不在

中心极限定理：许多独立随机变量的和（或平均）会收敛于正态分布，无论原始分布是什么。

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

这就是为什么：
- 测量误差近似正态（许多小的独立来源）
- 神经网络权重初始化使用正态分布
- SGD 中的梯度噪声近似正态（许多样本梯度的和）
- 给定均值和方差时，正态分布是最大熵分布

### 对数概率

原始概率会带来数值问题。许多小概率相乘很快会下溢为 0。

```math
\begin{aligned}
P(\text{sentence}) &= P(\text{word}_1) \cdot P(\text{word}_2) \cdot \ldots \cdot P(\text{word}_n) \\
&= 0.01 \cdot 0.003 \cdot 0.02 \cdot \ldots \\
&\to 0.0 \quad \text{(underflow after } \sim 30 \text{ terms)}
\end{aligned}
```

对数概率解决这个问题。乘法变成加法。

```math
\begin{aligned}
\log P(\text{sentence}) &= \log P(\text{word}_1) + \log P(\text{word}_2) + \ldots + \log P(\text{word}_n) \\
&= -4.6 + -5.8 + -3.9 + \ldots \\
&\to \text{finite number (no underflow)}
\end{aligned}
```

规则：
- $\log(a \cdot b) = \log(a) + \log(b)$
- 对数概率总是 $\le 0$（因为 $0 < P \le 1$）
- 越负 = 越不可能
- 交叉熵损失是正确类别的负对数概率

### Softmax 作为概率分布

神经网络输出原始分数（logits）。softmax 将其转换为合法的概率分布。

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

softmax 技巧：在指数化前减去最大 logit，防止溢出。

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax 将 softmax 与 log 结合以获得数值稳定性。PyTorch 在交叉熵损失内部使用它。

### 采样

采样指从分布中抽取随机值。在 ML 中：
- Dropout 随机采样将哪些神经元置零
- 数据增强采样随机变换
- 语言模型从预测分布中采样下一个 token
- 扩散模型采样噪声并逐步去噪

从任意分布采样需要逆变换采样、拒绝采样或重参数化技巧（VAE 中使用）等技术。

```figure
gaussian-pdf
```

## 构建它

### 第 1 步：概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### 第 2 步：从零实现 PMF 与 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 第 3 步：期望值与方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 第 4 步：从分布采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 第 5 步：softmax 与对数概率

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 第 6 步：中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 第 7 步：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

完整实现与所有可视化见 `code/probability.py`。

## 使用它

用 NumPy 和 SciPy，以上所有内容都是一行代码：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你从零构建了这些。现在你知道库函数在做什么。

## 练习

1. 为指数分布实现逆变换采样。验证：采样 10,000 个值并比较直方图与真实 PDF。

2. 为两个有偏骰子构建联合分布表。计算边际分布并检查骰子是否独立。

3. 一个 5 分类分类器输出 logits `[2.0, 0.5, -1.0, 3.0, 0.1]`，正确类别为索引 3，计算交叉熵损失。然后用 PyTorch 的 `nn.CrossEntropyLoss` 验证。

4. 编写函数，接收对数概率列表，返回最可能序列、总对数概率以及等价的原始概率。用一句 50 个词的句子测试，每个词概率为 0.01。

### 提示

**练习 1.** 从 uniform 分布抽取 `u`，返回 $-\ln(1 - u) / \lambda$。检查样本均值接近 $1/\lambda$，且直方图柱形遵循 $\lambda e^{-\lambda x}$。

**练习 2.** 填充一个 6×6 表，每个单元格是一个概率，所有单元格之和为 1，然后调用 `joint_to_marginals`。使用 `check_independence`：仅当每个单元格等于其两个边际的乘积时，骰子才独立。

**练习 3.** 对 logits 应用 softmax，取索引 3 处概率的 $-\ln$。结果应接近 0.421，即 `nn.CrossEntropyLoss` 对相同 logits 和 target 给出的值。

**练习 4.** 把对数概率相加，不要乘原始概率。对于 50 个概率为 0.01 的词，总对数概率应为 $50 \ln(0.01) \approx -230.26$，原始概率为 $10^{-100}$。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Sample space | "所有可能性" | 实验所有可能结果的集合 $S$ |
| PMF | "概率函数" | 给出每个离散结果精确概率的函数，所有概率之和为 1 |
| PDF | "概率曲线" | 连续变量的密度函数。对区间积分才能得到概率 |
| Conditional probability | "给定某事的概率" | $P(A \mid B) = P(A \text{ and } B) / P(B)$。贝叶斯思维与贝叶斯定理的基础 |
| Independence | "互不影响" | $P(A \text{ and } B) = P(A) \cdot P(B)$。知道一个事件对另一个毫无提示 |
| Expected value | "平均值" | 所有结果按概率加权的和。损失函数就是一种期望值 |
| Variance | "多分散" | 与均值之间期望的平方偏差。方差大 = 噪声大、估计不稳定 |
| Normal distribution | "钟形曲线" | $f(x) = (1/\sqrt{2\pi\sigma^2}) \cdot \exp(-(x-\mu)^2/(2\sigma^2))$。因 CLT 而无处不在 |
| Central Limit Theorem | "平均变正态" | 大量独立样本的均值会收敛于正态分布，无论来源分布是什么 |
| Joint distribution | "两个变量一起" | $P(X, Y)$ 描述 $X$ 与 $Y$ 每种结果组合的概率 |
| Marginal distribution | "把另一个变量加掉" | $P(X) = \sum_y P(X, Y)$。从联合分布恢复单个变量的分布 |
| Log probability | "概率的对数" | $\log P(x)$。把乘积变求和，防止长序列中的数值下溢 |
| Softmax | "把分数变概率" | $\text{softmax}(z_i) = \exp(z_i) / \sum_j \exp(z_j)$。将实值 logits 映射为合法概率分布 |
| Cross-entropy | "损失函数" | $-\sum p_{\text{true}} \cdot \log(p_{\text{predicted}})$。度量两个分布的差异。越低越好 |
| Logits | "原始模型输出" | softmax 之前的未归一化分数。名称来自 logistic 函数 |
| Sampling | "抽取随机值" | 按照概率分布生成值。模型就是这样生成输出的 |

## 延伸阅读

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) —— 为什么平均会趋于正态的可视化证明
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) —— 涵盖以上内容及更多的简明参考
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) —— 为什么数值稳定性重要以及如何实现
