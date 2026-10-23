# 用你自己的训练跑数拟合 Scaling Law

> 论文里的 scaling law 是漂亮的曲线；生产里的 scaling law 是决策工具：给你手头的 loss sweep，下一个模型该训 8B 还是 14B、用多少 token？这节课教你用自己的数据拟合。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 13 (Scaling Laws), Phase 3 · 14 (Experiment Tracking and Hyperparameter Sweeps)
**Time:** ~65 minutes

## 学习目标
- 在 log 空间拟合幂律，并把指数理解为收益递减的速率。
- 对比 Kaplan 式的参数外推与 Chinchilla 的 compute-optimal 拟合，包括学习率调度污染是如何让早期 law 产生偏差的。
- 实现 Chinchilla 的三种估计法，并识别每种方法引入的偏差。
- 用 bootstrap 给 scaling 指数估计置信区间，并说明这些区间何时可信。
- 从拟合好的 law 求解 compute-optimal $(N^*, D^*)$，并与 $D \approx 20N$ 经验规则做对比。
- 识别实际拟合陷阱：数据布置污染、不可约损失 E 的估计、小模型学习率调参，以及过度自信的外推。

## 问题

你刚跑完一轮 sweep：二十个模型，不同的宽度和深度，不同的 token 预算，都在同一份语料上训练。loss 随 FLOPs 下降的图有噪声，但趋势明显。这时有人问了唯一重要的问题：

**"我们有 $10^{21}$ FLOPs，该训 8B 还是 14B 的模型，用多少 token？"**

照着论文里的图回答很危险。已发表的 scaling law 是用别人的数据、别人的优化器、别人的学习率调度、甚至别人的 token 分布拟出来的，它们的常数不是你的常数。更糟的是，选错函数形式会悄悄把推荐翻过来：在 Kaplan 拟合下看起来更好的模型，在 Chinchilla 拟合下可能更差，因为两条曲线对不可约损失 floor 趋近速度的判断不同。

唯一安全的答案是用你自己的跑数亲手拟合 law，并清楚知道拟合在哪里是脆弱的。

## 概念

### 幂律与 log 图为何误导

最简单的 scaling law 是单变量幂律：

```math
L(N) = \left(\frac{N_c}{N}\right)^\alpha
```

两边取 log 就变成直线：

```math
\log L = \log N_c - \alpha \log N
```

这就是所有人都会先画 log-log 图的原因。但这条直线只在模型没有不可约损失 floor 时才成立。真实的语言模型有。Chinchilla 的形式把 floor 和可学习部分分开：

```math
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
```

其中 $E$ 是不可约损失，$A$、$B$ 是尺度常数，$\alpha \approx 0.34$、$\beta \approx 0.28$ 是 Hoffmann 的著名指数。如果你直接对完整的 $L$ 取 log，会得到一条在大尺度处弯向 $\log E$ 的曲线；只有对 $(L - E)$ 取 log，才能恢复 log 幂项的线性组合。

这意味着估计 $E$ 不是可选步骤。把它当 nuisance 参数忽略，拟合出的指数会吸收 floor，通常会压低斜率并低估 $\alpha$。

### Kaplan vs. Chinchilla：同一批数据，不同结论

Kaplan et al. (2020) 拟合的是按固定步数训练的模型。随着 $N$ 增大，$D$ 也随之增大，且学习率调度没有按模型大小单独调优。结果得到的参数 scaling 指数 $\alpha_N \approx 0.076$ 比 Chinchilla 后来的值小得多。

这个差异不是学术问题。更小的指数在说"建更大的模型"；更大的指数在说"模型尺寸收益很快饱和，所以也需要更多数据"。Kaplan 的数据被三件事污染：

1. **共变。** $N$ 和 $D$ 一起动，两个幂项混淆。
2. **学习率调度。** 小模型和大模型用的是对各自并非最优的调度，带来与尺寸相关的噪声。
3. **函数形式。** Kaplan law 没有显式的不可约损失项，floor 渗进了指数。

Hoffmann et al. (2021) 把这三点都修了：他们训练时要么固定 $N$、要么固定 $D$，按尺寸调学习率，并显式建模 $E$。得到的指数（$\alpha \approx 0.34$、$\beta \approx 0.28$）导出了 $D \approx 20N$ 的 compute-optimal 规则。

### Chinchilla 的三种估计法

Chinchilla 没有依赖单一拟合，而是用三种实验设计互相交叉验证，每种都有各自的盲区：

**方法一：固定 $N$，扫 $D$。** 用一个模型尺寸训练多组 token 量。拟合：

```math
L(D) = \left(E + \frac{A}{N^\alpha}\right) + \frac{B}{D^\beta}
```

括号里的项是常数偏移。只要 $N$ 足够大、$A/N^\alpha$ 项已不再快速变化，就能干净估计 $\beta$ 和 $B$。如果 $N$ 太小，偏移里仍含一个收缩中的 $N$ 项，$\beta$ 就会有偏。

**方法二：固定 $D$，扫 $N$。** 用相同的 token 数训练多个模型尺寸。拟合：

```math
L(N) = \left(E + \frac{B}{D^\beta}\right) + \frac{A}{N^\alpha}
```

这能干净估计 $\alpha$ 和 $A$，前提同样是 $D$ 足够大，数据项几乎平坦。

**方法三：联合拟合 $L(N, D)$。** 在 $(N, D)$ 平面上采点，同时拟合五个参数：

```math
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
```

当 sweep 同时覆盖两个轴时，这是最高效的，但参数彼此耦合。$E$ 估得不好会拉扯 $\alpha$、$\beta$；网格稀疏时，多个局部极小值可能有相近的损失。

正确的工作流是：以方法三为 headline，再用方法二校验 $\alpha$、方法一校验 $\beta$。如果三者不一致，说明某条实验轴被污染了。

### 实际拟合陷阱

**数据布置污染。** 如果每次跑用同样的 batch 顺序、同样的验证划分、同样的种子初始化模式，残差就是相关的。Bootstrap 置信区间会比真实更窄，因为你重采的是相关噪声。

**估计 $E$。** 不可约损失是曲线渐近的值，不是你恰好观测到的最低损失。用小 sweep 的最小值估计 $E$ 通常会高估它，并让拟合曲线变陡。更安全的方法是联合拟合 $E$，并检查拟合出的 $E$ 是否与你最大、最长训练的跑数 loss 接近。

**小模型学习率。** 如果小模型的学习率没有单独调优，它们的 loss 会高于真实 scaling 曲线。由于小模型对 $N$ 很大时的渐近行为最有信息量，这会向下偏 $\alpha$。

**Bootstrap 置信区间。** Bootstrap 百分位区间只在设计矩阵良态、噪声独立时才有效。如果你的 sweep 很小，或者点在 $(N, D)$ 空间里沿对角线聚集，CI 会过于乐观。可以报告它，但别把它当保证。

## 动手实现

`code/main.py` 生成一份合成 sweep，拟合完整 Chinchilla law，拟合两条单变量 law，做 bootstrap 置信区间，并求解 compute-optimal $(N^*, D^*)$。

**合成 sweep。** 从已知 law $L = E + A/N^\alpha + B/D^\beta$ 加上高斯噪声，生成 30 个 $(N, D, L)$ 点。这代替了你从 tracker 导出的真实超参 sweep。

**拟合。** 对固定的指数 $(\alpha, \beta)$，系数 $(E, A, B)$ 可用普通最小二乘解析求解。非线性指数则通过梯度下降优化约化后的平方误差目标。得益于正态方程 $X^T r = 0$，在最佳线性系数处只有指数需要迭代优化，梯度是解析的。多重启避免局部极小。

**单变量校验。** 再生成两份合成数据：一份固定 $D$ 变 $N$，一份固定 $N$ 变 $D$，分别用 $L(X) = E + A/X^\alpha$ 拟合。当实验设计干净时，它们能恢复真实指数。

**Bootstrap CI。** 将 30 个点有放回地重采样 100 次，每次重拟合完整模型，指数的 2.5 分位和 97.5 分位构成 95% 置信区间。

**Compute-optimal 前沿。** 给定预算 $C \approx 6ND$，最优分配有闭式解：

```math
N^* = \left( \frac{\alpha A}{\beta B (6/C)^\beta} \right)^{1/(\alpha+\beta)}, \qquad D^* = \frac{C}{6N^*}
```

脚本会打印从 $10^{19}$ 到 $10^{23}$ FLOPs 的前沿，并回答 $10^{21}$ FLOPs 下 8B vs 14B 的问题。

**运行：**

```console
python3 main.py
```

## 验证

`code/tests/test_main.py` 锁定让拟合可信的行为：

- 生成的合成数据形状正确，且 loss 始终高于不可约 floor；
- 完整 Chinchilla 拟合在 5% 以内恢复真实 $\alpha$ 和 $\beta$；
- 固定 $D$ 的单变量拟合恢复 $\alpha$；
- 固定 $N$ 的单变量拟合恢复 $\beta$；
- 闭式 compute-optimal $(N^*, D^*)$ 与暴力网格搜索一致；
- Kaplan 式无 $E$ 拟合系统性偏离真实指数；
- Bootstrap 置信区间包含真实指数；
- 最优模型尺寸随 compute budget 单调增长。

## 为什么工程师要关心

- **这条曲线就是预算决策。** "8B 还是 14B" 正是拟合 scaling law 能回答的问题。用自己的数据拟合，避免把别人的常数套到你的语料上。
- **函数形式很关键。** 忽略 $E$ 或混淆 $N$ 与 $D$ 都可能把推荐翻过来。方法三加上方法一/二的交叉校验，是最低限度的尽职调查。
- **学习率调度是实验的一部分。** 如果小模型相对其最优调度欠训，拟合出的指数就是错的，预算预测也会错。
- **置信区间是诚实检查。** 干净 sweep 的紧 CI 允许你插值；宽 CI 告诉你，在押注硅片之前还要跑更多实验。
- **外推很危险。** Scaling law 是可靠的内插器，却是危险的外推器。在你 sweep 的范围内拟合；不要用 1M 参数点的 law 去预测 100B 模型。

## 提示

- 在解释 log-log 斜率前，务必先减去不可约损失估计。直线属于 $(L - E)$，不属于 $L$。
- 如果方法一、方法二、方法三不一致，怀疑学习率污染，或某条轴覆盖不足。
- Bootstrap CI 假设点独立。共享种子、数据顺序或验证划分的跑数会违背这一假设。
- $D \approx 20N$ 规则是针对特定常数组和 compute range 的标题数字。你自己的拟合 law 可能给出 15 或 35；重要的是这个数字来自你的数据。
- 保留最大、最长训练的跑数参与拟合。它们锚定 $E$，而 $E$ 又锚定所有其他参数。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|---------|---------|
| Scaling law | "loss 随尺寸变化" | 用模型大小、数据量或 compute 预测 loss 的参数化曲线。 |
| 幂律指数 | "斜率" | 可约损失随 $N$ 或 $D$ 增长而下降的速率；参数方向为 $\alpha$，token 方向为 $\beta$。 |
| 不可约损失 | "$E$" | 即使模型和数据无限，loss 也会逼近的下限；任务的熵。 |
| Kaplan law | "2020 scaling" | $L(N) \approx (N_c/N)^{\alpha_N}$，在 $N$、$D$ 共变下拟合；指数较小，偏向大模型。 |
| Chinchilla law | "$E + A/N^\alpha + B/D^\beta$" | 用受控实验拟合的联合 law，导出 $D \approx 20N$ 的 compute-optimal 规则。 |
| Compute-optimal | "$D/N \approx 20$" | 在固定训练 FLOP 预算下使 loss 最小的 $(N, D)$ 分配。 |
| Bootstrap CI | "误差条" | 对指数做重采样的百分位区间；只有在数据点独立时才诚实。 |
| 数据污染 | "sweep 噪声" | 共享种子、调度或验证划分导致残差相关，使 CI 过窄。 |
