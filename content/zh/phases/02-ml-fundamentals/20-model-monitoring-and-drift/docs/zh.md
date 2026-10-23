# 生产环境模型监控 —— 数据漂移与概念漂移

> 已部署的模型在没有报错的情况下失效了。它持续返回分数，而现实世界却在远离它的训练数据，只有监控才能揭示这个差距。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2, Lessons 09 (Model Evaluation), 16 (Anomaly Detection), and 19 (ML System Design)
**Time:** ~110 minutes

## 学习目标

- 在四个层面选择监控信号：系统健康、输入、预测和结果
- 根据联合分布中变化的部分区分协变量漂移、先验漂移和概念漂移
- 从零实现 Population Stability Index（PSI）、双样本 Kolmogorov-Smirnov 统计量和卡方检验
- 比较固定参考窗口与滑动当前窗口，并解释窗口大小如何影响噪声和延迟
- 设计告警策略，通过效应量、多重检验校正和持久性规则来限制误报
- 判断哪种漂移需要重训练、哪种需要新标签、哪种无需处理

## 问题

一个支付风险模型以 95% 的准确率上线。两个月后，一场营销活动带来了购物车金额更高的 App 用户。四个月后，欺诈者改变了策略，原本安全的支付开始变成拒付。

在此期间，模型服务对每个请求都返回 HTTP 200。延迟正常，日志也没有错误。第一个麻烦迹象是几周后才出现的一份关于拒付的财务报告。

软件监控检查服务是否有响应。模型监控检查答案是否仍然正确，以及数据是否仍然像训练模型的数据。本节课为第二种工作构建统计量、窗口和告警策略。你将在一个注入了两次变化的合成数据流上运行它们，并观察哪个监控器对哪种变化作出反应。

## 概念

### 监控什么

监控四个层面。前三个提供早期预警。只有第四个衡量正确性，但它到来得很晚。

| 层面 | 信号 | 何时可用 | 它回答的问题 |
|---|---|---|---|
| 系统健康 | p50 和 p99 延迟、错误率、吞吐量、内存 | 立即 | 服务是否有响应？ |
| 输入 | 每个特征的分布、缺失值比例、模式、新类别 | 立即 | 在线数据是否像训练数据？ |
| 预测 | 分数分布、正例率、默认答案占比 | 立即 | 模型表现是否和之前一致？ |
| 结果 | 准确率、精确率、校准度、业务指标 | 标签延迟之后 | 模型仍然正确吗？ |

输入监控是本节课的主题，因为它在标签到达之前就能工作。它有一个限制，演示会展示：有些变化不会改变输入。

### 三种漂移

用两种方式写出输入 `x` 和标签 `y` 的联合分布：

```math
P(x, y) = P(x) \cdot P(y \mid x) = P(y) \cdot P(x \mid y)
```

每种漂移改变不同的因子：

| 漂移类型 | 什么变化 | 什么不变 | 支付示例 | 哪个监控器能看到 |
|---|---|---|---|---|
| 协变量漂移 | `P(x)` | `P(y \| x)` | 更多使用 App 且购物车金额更高的用户 | 输入监控器 |
| 先验漂移 | `P(y)` | `P(x \| y)` | 攻击期间欺诈率翻倍，但欺诈模式未变 | 预测和结果监控器 |
| 概念漂移 | `P(y \| x)` | 可能不变 | 同样的购物车现在有不同的欺诈风险 | 只有结果监控器 |

Gama 等人将 `P(y | x)` 的变化称为真实概念漂移（real concept drift）。他们将改变 `P(x)` 但保持 `P(y | x)` 不变的变化称为虚拟漂移（virtual drift）。他们的综述还按时间发生方式对漂移进行分类：突发的、渐进的、逐渐的，或周期复发的概念。

协变量漂移并不总是有害。如果模型很好地学习了 `P(y | x)`，它对每个输入仍然正确。但当流量移动到训练数据很少或靠近决策边界的区域时，准确率仍可能下降。

### 延迟标签

结果在预测之后到达。延迟取决于具体问题：

| 问题 | 标签何时到达 |
|---|---|
| 广告点击 | 数秒 |
| 未来 30 天内的流失 | 30 天 |
| 信用卡欺诈 | 拒付到达时，通常是数周后 |
| 贷款违约 | 数月或数年 |

在延迟期间，你拥有输入和预测，但没有结果。监控输入和预测，在标签到达时进行关联，并仅在已标注部分计算结果指标。演示使用 500 个事件的标签延迟，因此结果监控器总是回顾过去 500 个事件。

### 参考窗口与当前窗口

每个漂移检验都比较两个样本：

- **参考窗口。** 训练数据，或你手工检查过的一段生产期。它在重训练之前保持固定。
- **当前窗口。** 最近的事件，可以是最近的 `N` 个事件或最近的 `T` 小时。

滑动窗口以小于其大小的步长前进，因此相邻窗口会重叠。翻转窗口则按完整大小移动。窗口大小需要权衡。小窗口反应快，但会把噪声误判为漂移。大窗口稳定，但报告变化较晚。

季节性会破坏简单的窗口。在许多产品中，周一流量与周六流量不同。让参考覆盖完整周期，或将每个窗口与更早周期中的同一时段比较。

### Population Stability Index

PSI 源自信用评分。将参考值切分为若干区间，通常按参考十分位数分成十个区间，使每个区间包含 10% 的参考样本。统计当前值在每个区间中的占比，然后对每个区间累加一项：

```math
\text{PSI} = \sum_{\text{bins}} (\text{current\_share} - \text{reference\_share}) \cdot \ln\left(\frac{\text{current\_share}}{\text{reference\_share}}\right)
```

占比为 0 的区间会让对数变为无穷大，因此将每个占比裁剪到 0.0001 这样的小值。PSI 是对称的，等于从当前到参考的 KL 散度加上从参考到当前的 KL 散度。

实践者按固定区间读取 PSI：低于 0.1 为稳定，0.1 到 0.25 为中等偏移，高于 0.25 为显著偏移。把这些区间当作惯例而非严格检验。PSI 随区间数量增加而增大，随样本量增大而减小。在演示中，无漂移时 500 个事件的窗口得分在 0.01 到 0.04 之间。

### Kolmogorov-Smirnov 统计量

双样本 KS 统计量 `D` 是两个经验累积分布函数之间最大的垂直差距：

```math
D = \max_x \left\lvert F_{\text{reference}}(x) - F_{\text{current}}(x) \right\rvert
```

相同样本的 `D` 为 0，不重叠样本的 `D` 为 1。它不需要分箱。p 值来自 Kolmogorov 分布，有效样本量为 `n * m / (n + m)`。

在大样本下，即使一个没人关心的差距，p 值也会趋近于 0。每个窗口百万事件时，0.02 个标准差的偏移也会「显著」。把 `D` 本身作为效应量，仅当 `D` 和 p 值都超过阈值时才告警。

### 类别特征的卡方检验

对于类别特征，例如支付渠道，构建一个两行（参考、当前）每类别一列的表格。每个单元格的期望计数来自行总计和列总计：

```math
\begin{aligned}
\text{expected}[r][c] &= \text{row\_total}[r] \cdot \text{column\_total}[c] / \text{grand\_total} \\
\text{chi2} &= \sum_{\text{cells}} (\text{observed} - \text{expected})^2 / \text{expected}
\end{aligned}
```

对于 `k` 个类别，检验的自由度为 `k - 1`。只在当前窗口出现的类别会带来很大的项，因此检验会对新值作出反应。Cramér's V 是效应量。对于两行表格，它等于 `chi2 / grand_total` 的平方根。

### 告警阈值与误报

当没有变化时，每个检验的误报率为 `alpha`。监控会运行大量检验。如果每小时检查 50 个特征，`alpha = 0.05`，在无变化时每天大约会有 60 次误报。人们在此之前就已经不再看告警了。

四条规则让告警率保持可用：

1. **校正多重检验。** Bonferroni 校正将 `alpha` 除以检验数量。演示中两个检验的族 `alpha` 为 0.01，因此每个检验使用 0.005。
2. **要求效应量。** 仅当 PSI、`D` 或 Cramér's V 也超过阈值时才告警。
3. **要求持久性。** 仅在连续 `k` 个漂移窗口后才告警。演示使用 `k = 2`。
4. **区分警告和告警。** 警告进入仪表盘，告警通知到人。

演示展示了规则 2 的实际效果。一个干净的窗口卡方 p 值为 0.0009，低于每个检验的 `alpha`。但其 Cramér's V 低于 0.1，因此策略记录为警告而非告警。

### 重训练触发条件

有三种常见的重训练触发条件：

- **按计划。** 每周或每月重训练。这很简单，但会在无变化时重训练，在变化发生时等待。
- **漂移。** 当重要特征的输入漂移持续高于阈值时重训练。
- **性能。** 当结果指标下降到参考值减去容差以下时重训练。

漂移类型决定了重训练能做什么：

| 漂移类型 | 有效操作 |
|---|---|
| 协变量漂移，准确率稳定 | 通常无需操作。继续观察结果指标。 |
| 协变量漂移到数据稀少的区域 | 用包含这些区域的近期数据重训练 |
| 先验漂移 | 重新校准分数或调整决策阈值 |
| 概念漂移 | 先收集新标签。用旧标签重训练只会再次学到旧概念。 |

重训练后，在最近的已标注数据上测试新模型。按照系统设计课中的 shadow 和 canary 阶段逐步上线。然后将参考窗口重置为新的训练数据。

```figure
mlprod-drift-psi
```

## 构建它

`code/main.py` 中的代码仅使用标准库。

### 步骤 1：分位数区间与 PSI

区间边界是参考十分位数。`bisect_right` 找到每个值所在的区间。

```python
def quantile_edges(reference: list[float], bins: int = 10) -> list[float]:
    ordered = sorted(reference)
    n = len(ordered)
    edges = [ordered[min(n - 1, (n * i) // bins)] for i in range(1, bins)]
    return sorted(set(edges))


def psi_from_fractions(expected: list[float], actual: list[float], eps: float = 1e-4) -> float:
    total = 0.0
    for e, a in zip(expected, actual):
        e, a = max(e, eps), max(a, eps)
        total += (a - e) * math.log(a / e)
    return total
```

### 步骤 2：KS 统计量及其 p 值

同时遍历两个已排序样本。在每个不同取值处，跳过两个样本中该值的所有副本，然后比较两个累积占比。这样能正确处理相同值。

```python
def ks_statistic(sample_a: list[float], sample_b: list[float]) -> float:
    a, b = sorted(sample_a), sorted(sample_b)
    n, m = len(a), len(b)
    i = j = 0
    d = 0.0
    while i < n and j < m:
        x = min(a[i], b[j])
        while i < n and a[i] <= x:
            i += 1
        while j < m and b[j] <= x:
            j += 1
        d = max(d, abs(i / n - j / m))
    return d
```

p 值使用 Kolmogorov 分布的级数 `2 * sum((-1)^(k-1) * exp(-2 k^2 lambda^2))`，其中 `lambda = sqrt(n * m / (n + m)) * D`。

### 步骤 3：卡方检验

`chi2_sf` 通过正则化上不完全伽马函数计算卡方生存函数。小参数使用级数，大参数使用连分数。`chi_square_test` 构建两行表格，计算统计量，并返回 p 值和 Cramér's V。

```python
def chi_square_test(reference_counts: dict[str, int], current_counts: dict[str, int]) -> ChiSquareResult:
    categories = [c for c in sorted(set(reference_counts) | set(current_counts))
                  if reference_counts.get(c, 0) + current_counts.get(c, 0) > 0]
    rows = [[reference_counts.get(c, 0) for c in categories], [current_counts.get(c, 0) for c in categories]]
    row_totals = [sum(r) for r in rows]
    grand = sum(row_totals)
    if len(categories) < 2 or 0 in row_totals:
        return ChiSquareResult(0.0, 0, 1.0, 0.0)
    statistic = 0.0
    for col, _ in enumerate(categories):
        col_total = rows[0][col] + rows[1][col]
        for r in range(2):
            expected = row_totals[r] * col_total / grand
            statistic += (rows[r][col] - expected) ** 2 / expected
    dof = len(categories) - 1
    return ChiSquareResult(statistic, dof, chi2_sf(statistic, dof), math.sqrt(statistic / grand))
```

### 步骤 4：窗口与漂移监控器

`SlidingWindow` 在 `deque` 中保存最近的 `size` 个事件。`DriftMonitor` 只计算一次参考区间和计数。它的 `check` 方法返回每个数值特征的 PSI、`D` 和 KS p 值，以及每个类别特征的卡方结果。

### 步骤 5：告警策略

策略将每组指标转换为一个级别，并按信号统计连续漂移窗口的数量：

```python
def level(self, metrics: dict[str, float]) -> str:
    if "psi" in metrics:
        if metrics["psi"] >= self.psi_drift or (metrics["ks"] >= self.ks_drift and metrics["ks_p"] < self.alpha):
            return "drift"
        return "warn" if metrics["psi"] >= self.psi_warn else "ok"
    if metrics["p"] < self.alpha and metrics["v"] >= self.min_cramers_v:
        return "drift"
    return "warn" if metrics["p"] < self.alpha else "ok"
```

`update` 在完成连击的窗口返回 `ALERT`，在告警后的第一个干净窗口返回 `resolved`。

### 步骤 6：注入漂移的数据流

数据流包含 9000 个支付事件，每个事件有 `amount`（金额）、`channel`（渠道）和欺诈 `label`（标签）。四个分段定义如下：

| 事件 | 金额 | 渠道占比（App、门店、网页） | 标签边界 | 变化 |
|---|---|---|---|---|
| 0 到 2999 | 均值 100，标准差 20 | 35%、15%、50% | 120 | 参考期 |
| 3000 到 4999 | 均值 118，标准差 24 | 60%、10%、30% | 120 | 协变量漂移 |
| 5000 到 6499 | 均值 100，标准差 20 | 35%、15%、50% | 120 | 恢复正常 |
| 6500 到 8999 | 均值 100，标准差 20 | 35%、15%、50% | 105 | 概念漂移 |

模型是基于 `amount` 的阈值，拟合于前 2000 个事件。窗口包含 500 个事件，移动步长为 250。标签延迟 500 个事件到达。

```bash
python3 code/main.py
```

```console
  end    PSI   KS D   chi2 p    acc  levels (amount/channel/accuracy)  events
 3000  0.010  0.041  3.0e-01  0.932  ok/ok/ok
 3250  0.152  0.163  7.6e-06  0.924  drift/warn/ok
 3500  0.463  0.298  2.8e-26  0.936  drift/drift/ok                     ALERT amount
 3750  0.609  0.343  3.5e-35  0.942  drift/drift/ok                     ALERT channel
 ...
 5500  0.026  0.044  2.6e-01  0.898  ok/ok/ok                           resolved amount, resolved channel
 ...
 7250  0.040  0.048  7.1e-01  0.848  ok/ok/drift
 7500  0.022  0.039  6.1e-01  0.778  ok/ok/drift                        ALERT accuracy
```

分三部分读取结果：

- **协变量漂移。** 漂移从事件 3000 开始，`amount` 告警在 3500 触发。准确率保持在 0.89 到 0.95 之间，因为标签规则没有改变。
- **恢复。** 输入在 5000 恢复正常，两个告警在 5500 解除。
- **概念漂移。** 漂移从 6500 开始，输入监控器保持安静，因为输入没有改变。准确率从 0.946 下降到 0.76 到 0.79 之间。准确率在 7500 触发，比变化晚了 1000 个事件，其中 500 个事件由标签延迟造成。

## 使用它

**scipy。** `scipy.stats` 实现了这两种检验。统计量与从零实现的代码一致。运行本节课不需要 scipy。

```python
from scipy import stats

result = stats.ks_2samp(reference_amounts, current_amounts)
print(result.statistic, result.pvalue)

table = [[1000, 700, 300], [140, 290, 70]]
chi2 = stats.chi2_contingency(table)
print(chi2.statistic, chi2.pvalue, chi2.dof)
```

我们在随机样本上将 `ks_statistic` 和 `chi_square_test` 与 scipy 1.18.1 进行了核对。KS 统计量、卡方统计量和卡方 p 值都达到了浮点精度一致。在 200 对随机样本中，KS p 值与 scipy 的差异最高达到 scipy 值的 18%。最大差距出现在极小 p 值处。

差距来自方法。我们的代码使用极限 Kolmogorov 分布。默认情况下，当两个样本都最多 10,000 个值时 `ks_2samp` 使用精确方法。它的渐近模式使用带舍入有效样本量的有限样本分布。对于两个类别，`chi2_contingency` 默认应用 Yates 连续性校正，因此统计量会有所不同。

**监控库。** 几个开源库将这些检验打包成报告和仪表盘：

- **Evidently** 为每列选择一种漂移检验。参考数据不超过 1000 行时，数值列使用 KS 检验，类别列使用卡方检验。更大的数据使用 Wasserstein 距离和 Jensen-Shannon 散度，阈值为 0.1。PSI 也是可选的其他方法之一。
- **NannyML** 在标签到达前估计模型性能。它的基于置信度的性能估计（CBPE）使用模型预测的概率，这些概率必须校准良好。NannyML 文档指出，CBPE 能处理协变量偏移，但无法应对概念漂移。演示数据流说明了原因：只有标签能揭示概念漂移。
- **TensorFlow Data Validation** 比较连续的数据片段。类别特征使用 L-infinity 距离，数值特征使用近似 Jensen-Shannon 散度。它还会比较训练数据和服务数据，这部分内容由特征存储课涵盖。

从零实现的代码展示了这些工具各自计算的内容。工具增加了存储、调度、仪表盘和更多检验。

对于 LLM 应用，基础设施阶段的 [LLM 可观测性课程](../../../17-infrastructure-and-production/13-llm-observability/docs/en.md) 涵盖了追踪和评估工具。本节课的思想仍然适用于这些系统的输入和输出。

## 交付它

本节课生成 `outputs/skill-drift-monitoring.md`。该 skill 为新模型规划监控器并对漂移告警进行分流。它会询问哪个层面触发了告警、哪种漂移符合证据，以及正确做法是等待、重新校准、收集标签还是重训练。

## 练习

1. 用 200 和 2000 个事件的窗口运行数据流。记录事件 3000 之前的误报警告数量和告警延迟。
2. 设置 `n_tests=1`、`patience=1` 和 `min_cramers_v=0`。统计 5000 到 6500 的干净期内的告警数量。
3. 添加一个均值每 100 个事件增长 0.01 的特征。找出 PSI 和 KS 哪个信号先反应。
4. 在相同的十分位区间上实现 Jensen-Shannon 距离。在每个窗口上与 PSI 比较。
5. 添加一个模型正预测率的监控器。解释为什么它在概念漂移期间保持安静。

## 关键术语

| 术语 | 常见说法 | 含义 |
|------|----------------|----------------------|
| 数据漂移 | "数据变了" | 输入、预测或标签的分布相对于参考窗口发生变化 |
| 协变量漂移 | "新类型的用户" | `P(x)` 变化，而 `P(y \| x)` 保持不变 |
| 先验漂移 | "正例比以前多" | `P(y)` 变化，而 `P(x \| y)` 保持不变，也称为标签偏移 |
| 概念漂移 | "游戏规则变了" | `P(y \| x)` 变化，因此相同输入现在有不同的正确答案 |
| 参考窗口 | "基线数据" | 固定样本，通常是训练数据，每个当前窗口都与它比较 |
| PSI | "稳定性分数" | 对每个区间，用（当前占比减参考占比）乘以两者比值的对数，再对称求和 |
| KS 统计量 | "KS 检验" | 两个经验累积分布函数之间的最大差距 |
| Cramér's V | "类别效应量" | 将卡方统计量缩放到 0 到 1 的范围 |
| 标签延迟 | "真实标签后来才到" | 预测与其真实结果到达之间的时间 |
| 告警疲劳 | "太多告警" | 由于大多数告警是误报，人们开始忽略告警 |

## 进一步阅读

- [Gama et al., A Survey on Concept Drift Adaptation (ACM Computing Surveys, 2014)](https://research.tue.nl/en/publications/a-survey-on-concept-drift-adaptation): real and virtual drift, drift patterns, and adaptation methods
- [Rabanser, Günnemann, and Lipton, Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift (NeurIPS 2019)](https://arxiv.org/abs/1810.11953): two-sample tests for shift detection
- [Lu et al., Learning under Concept Drift: A Review (2020)](https://arxiv.org/abs/2004.05785): drift detection, understanding, and adaptation methods
- [Yurdakul, Statistical Properties of Population Stability Index (Western Michigan University, 2018)](https://scholarworks.wmich.edu/dissertations/3208/): how PSI behaves with sample size and bin count
- [scipy.stats.ks_2samp documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html): exact and asymptotic p-value methods for the two-sample KS test
- [Evidently, How data drift detection works](https://docs.evidentlyai.com/metrics/explainer_drift): default test selection by column type and data size
- [NannyML, Performance estimation](https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html): CBPE and its assumptions
