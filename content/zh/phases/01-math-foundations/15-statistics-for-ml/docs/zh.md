# 机器学习中的统计学

> 统计学让你知道模型是真的有效，还是只是运气好。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06 (Probability and Distributions), 07 (Bayes' Theorem)
**Time:** ~120 minutes

## 学习目标

- 从零计算描述性统计、Pearson/Spearman 相关性和协方差矩阵
- 执行假设检验（t 检验、卡方检验），正确解释 p 值和置信区间
- 使用 bootstrap 重采样，在没有分布假设的情况下为任意指标构建置信区间
- 使用效应量指标区分统计显著性与实际显著性

## 问题

你训练了两个模型。模型 A 在测试集上得分 0.87，模型 B 得分 0.89。你部署了模型 B。三周后，生产指标反而比之前更差。发生了什么？

模型 B 实际上并没有超越模型 A。0.02 的差距只是噪声。你的测试集太小，或者方差太大，或者两者兼有。你把随机性包装成了改进。

这种情况 constantly 发生。Kaggle 排行榜波动、无法复现的论文、仅凭几百个样本就宣布胜者的 A/B 测试。根本原因总是一样的：有人跳过了统计学。

统计学给你区分信号与噪声的工具。它告诉你一个差异是否真实、你应有多自信、以及需要多少数据才能相信结果。每个 ML pipeline、每次模型比较、每个实验都需要统计学。没有它，你只是在猜。

## 概念

### 描述性统计：概括数据

在建模之前，你需要知道数据长什么样。描述性统计把数据集压缩成几个能刻画其形态的数字。

**集中趋势度量**回答「中间在哪里？」

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

均值是平衡点，中位数是中间线。当它们分开，分布就是偏斜的。收入分布通常 $\text{mean} \gg \text{median}$（亿万富翁导致右偏）。训练期间的损失分布通常 $\text{mean} \ll \text{median}$（简单样本导致左偏）。

**离散程度度量**回答「数据有多分散？」

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**百分位数**把排序后的数据分成 100 等份。第 25 百分位数（Q1）表示 25% 的值低于该点，第 50 百分位数是中位数，第 75 百分位数是 Q3。

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

在 ML 中，你关心百分位数的地方包括推断延迟、预测置信度分布、理解误差分布。平均误差低但 P99 误差糟糕的模型，对安全关键应用可能毫无用处。

**样本统计量 vs 总体统计量。** 从样本计算方差时，要用 $(n-1)$ 而不是 $n$ 作分母。这就是 Bessel 校正。它补偿了你的样本均值并非真实总体均值这一事实。用 $n$ 作分母会系统性地低估真实方差，用 $(n-1)$ 则得到无偏估计。

```math
\begin{aligned}
\text{Population variance:} \quad \sigma^2 &= \frac{1}{N} \sum (x_i - \mu)^2 \\
\text{Sample variance:} \quad s^2 &= \frac{1}{n-1} \sum (x_i - \bar{x})^2
\end{aligned}
```

实践中：如果 $n$ 很大（数千样本），差别可忽略。如果 $n$ 很小（数十样本），就有影响。

### 相关性：变量如何共同变动

相关性度量两个变量之间线性关系的强度和方向。

**Pearson 相关系数**度量线性关联：

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Pearson 假设关系是线性的，且两个变量大致正态分布。它对异常值敏感。一个极端点就能把 $r$ 从 0.1 拖到 0.9。

**Spearman 秩相关**度量单调关联：

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**何时使用哪一种：**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**黄金法则：** 相关不等于因果。冰淇淋销量与溺亡人数相关，因为两者都在夏天上升。模型准确率与参数数量相关，但增加参数并不会自动提高准确率（参见：过拟合）。

### 协方差矩阵

两个变量之间的协方差度量它们如何共同变动：

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

对 $d$ 个特征，协方差矩阵 $C$ 是一个 $d \times d$ 矩阵，其中 $C[i][j] = \mathop{\mathrm{Cov}}(\text{feature}_i, \text{feature}_j)$。对角线元素 $C[i][i]$ 是各特征的方差。

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**与 PCA 的联系。** PCA 对协方差矩阵做特征分解。特征向量是主成分（方差最大的方向），特征值告诉你每个成分捕获了多少方差。这正是第 10 课的内容，而现在你明白为什么协方差矩阵是正确分解对象：它编码了数据中所有两两线性关系。

**与相关性的联系。** 相关矩阵是标准化变量（每个除以其标准差）后的协方差矩阵。相关性把协方差归一化，使所有值落在 $[-1, 1]$。

### 假设检验

假设检验是在不确定性下做决策的框架。你从一个主张出发，收集数据，再判断数据是否与该主张一致。

**设定：**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p 值**是在假设 H0 为真的前提下，观察到像当前数据一样极端的数据的概率。它不是「H0 为真的概率」。这是统计学中最常见的误解。

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**置信区间**给出参数的一个合理取值范围：

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

置信区间的宽度反映精确度。宽区间意味着不确定性高，窄区间意味着估计精确（但如果数据有偏，不一定准确）。

### t 检验

t 检验比较均值。有几种变体。

**单样本 t 检验：** 总体均值是否与某个假设值不同？

```math
\begin{aligned}
t &= \frac{\bar{x} - \mu_0}{s / \sqrt{n}} \\[4pt]
\text{degrees of freedom} &= n - 1
\end{aligned}
```

**双样本 t 检验（独立）：** 两组均值是否不同？

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**配对 t 检验：** 测量成对出现时使用（同一模型在相同数据划分上评估）：

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

在 ML 中，配对 t 检验很常见：你把两个模型跑在相同的 10 个交叉验证折上，然后逐对比较它们的分数。

### 卡方检验

卡方检验检查观测频数是否与期望频数一致。适用于分类数据。

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### 面向 ML 模型的 A/B 测试

ML 中的 A/B 测试与网页 A/B 测试不同。模型比较有特定的挑战：

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**流程：**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### 统计显著性 vs 实际显著性

一个结果可能在统计上显著，但在实际上毫无意义。数据足够多时，再微小的差异也会变得统计显著。

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**效应量**量化差异有多大，与样本量无关：

```math
\begin{aligned}
\text{Cohen's } d &= \frac{\text{mean}_1 - \text{mean}_2}{\text{pooled\_std}} \\[4pt]
d &= 0.2: \quad \text{small effect} \\
d &= 0.5: \quad \text{medium effect} \\
d &= 0.8: \quad \text{large effect}
\end{aligned}
```

总是要同时报告 p 值和效应量。p 值告诉你差异是否真实，效应量告诉你差异是否重要。

### 多重比较问题

当你检验很多假设时，有些会碰巧「显著」。如果在 $\alpha = 0.05$ 下检验 20 件事，即使什么都不真实，你也期望出现 1 个假阳性。

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Bonferroni 校正：** 把 $\alpha$ 除以检验次数。

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

在 ML 中，当你跨多个指标比较模型、测试很多超参数配置、或在多个数据集上评估时，这一点很重要。

### Bootstrap 方法

Bootstrap 通过有放回地重采样你的数据来估计统计量的抽样分布。不需要对底层分布做任何假设。

**算法：**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Bootstrap 置信区间（百分位法）：**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**为什么 Bootstrap 对 ML 重要：**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**用于模型比较的 Bootstrap：**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

这比对 t 检验更稳健，因为它不做分布假设。

### 参数检验 vs 非参数检验

**参数检验**假设特定分布（通常是正态）：

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**非参数检验**不做分布假设：

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**何时使用非参数检验：**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**何时使用参数检验：**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

在 ML 实验中，你通常只有很小的 $n$（5 或 10 个交叉验证折），因此 Wilcoxon 符号秩检验等非参数检验通常比 t 检验更合适。

### 中心极限定理：实际含义

CLT 指出，无论总体分布如何，样本均值的分布都会随着 $n$ 增大而趋于正态分布。

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**为什么对 ML 重要：**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**CLT 不能做什么：**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### ML 论文中常见的统计错误

1. **在训练集上测试。** 必然导致过拟合。始终保留模型在训练期间从未见过的数据。

2. **没有置信区间。** 只报告一个准确率数字而不带不确定性，会让结果无法复现、无法验证。

3. **忽略多重比较。** 测试 50 种配置却只报告最好的，不做校正，会夸大假阳性率。

4. **混淆统计显著性与实际显著性。** 对 0.01% 的准确率提升得到 p 值 0.001 并没有意义。

5. **在类别不平衡数据上使用准确率。** 99% 的准确率在 99% 为负类的数据集上意味着模型什么都没学到。应使用精确率、召回率、F1 或 AUC。

6. **挑选指标。** 只报告模型获胜的指标。诚实的评估会报告所有相关指标。

7. **训练/测试划分间泄漏信息。** 在划分前做归一化，或用未来数据预测过去。

8. **小测试集且无方差估计。** 在 100 个样本上评估并声称 2% 提升，这是噪声而非信号。

9. **在数据不独立时假设独立。** 同一患者的多张医学影像、同一文档中的多个句子。组内观测是相关的。

10. **P-hacking。** 尝试不同检验、子集或排除标准，直到得到 $p < 0.05$。结果是搜索过程的产物。

## 构建它

你将实现：

1. **从零开始的描述性统计**（均值、中位数、众数、标准差、百分位数、IQR）
2. **相关函数**（Pearson 和 Spearman，含协方差矩阵）
3. **假设检验**（单样本 t 检验、双样本 t 检验、卡方检验）
4. **Bootstrap 置信区间**（对任意统计量，无需分布假设）
5. **A/B 测试模拟器**（生成数据、检验、检查一类错误和二类错误）
6. **统计显著性 vs 实际显著性演示**（展示大 $n$ 如何让一切都「显著」）

全部只使用 `math` 和 `random`。不用 numpy，不用 scipy。

```figure
f3-bootstrap-resample
```

## 关键术语

| 术语 | 定义 |
|------|------|
| Mean | 数值之和除以数量。对异常值敏感。 |
| Median | 排序后数据的中间值。对异常值稳健。 |
| Standard deviation | 方差的开方。用原始单位度量离散程度。 |
| Percentile | 给定百分比的数据低于该值。 |
| IQR | 四分位距。Q3 减 Q1。中间 50% 数据的离散程度。 |
| Pearson correlation | 度量两个变量之间的线性关联。范围 $[-1, 1]$。 |
| Spearman correlation | 使用秩度量单调关联。 |
| Covariance matrix | 所有特征两两之间协方差构成的矩阵。 |
| Null hypothesis | 默认假设，通常表示无效应或无差异。 |
| p-value | 在零假设为真的前提下，观察到如此极端数据的概率。 |
| Confidence interval | 在给定置信水平下，参数的合理取值范围。 |
| t-test | 检验均值是否有显著差异。使用 t 分布。 |
| Chi-squared test | 检验观测频数是否与期望频数不同。 |
| Effect size | 差异的大小，与样本量无关。Cohen's d 是常用指标。 |
| Bonferroni correction | 把显著性阈值除以检验次数以控制假阳性。 |
| Bootstrap | 有放回重采样以估计抽样分布。 |
| Type I error | 假阳性。零假设为真时拒绝零假设。 |
| Type II error | 假阴性。零假设为假时未能拒绝。 |
| Statistical power | 正确拒绝错误零假设的概率。功效 = 1 减二类错误率。 |
| Central limit theorem | 样本均值随样本量增大而趋于正态分布。 |
| Parametric test | 假设数据服从特定分布（通常为正态）。 |
| Non-parametric test | 不做分布假设。基于秩或符号。 |
