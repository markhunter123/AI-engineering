# 特征选择

> 特征并非越多越好，选对特征才是关键。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 2, Lessons 01-09, 08 (feature engineering)
**Time:** ~75 minutes

## 学习目标

- 从零实现过滤法（方差阈值、互信息、卡方检验）和包装法（RFE、前向选择）
- 解释为什么互信息能捕捉相关性所遗漏的非线性特征-目标关系
- 比较 L1 正则化（嵌入法选择）与 RFE（包装法选择），并评估它们的计算权衡
- 构建一个结合多种方法的特征选择 pipeline，并证明其在留出数据上的泛化能力提升

## 问题

你有 500 个特征。模型训练很慢，不断过拟合，没人能解释它学到了什么。你增加更多特征希望提升性能，结果反而更差。

这就是维度灾难的体现。随着特征数量增加，特征空间的体积呈指数级膨胀。数据点变得稀疏，点之间的距离趋同。模型需要指数级更多的数据才能找到真正的模式。噪声特征淹没了信号特征。过拟合成了默认状态。

特征选择就是解药。剥除噪声，剔除冗余，保留真正携带目标信息的特征。结果是：训练更快、泛化更好、模型也更容易解释。

目标不是使用所有可用信息，而是使用正确的信息。

## 概念

### 特征选择的三个类别

每种特征选择方法都属于以下三类之一：

```mermaid
flowchart TD
    A[Feature Selection Methods] --> B[Filter Methods]
    A --> C[Wrapper Methods]
    A --> D[Embedded Methods]

    B --> B1["Variance Threshold"]
    B --> B2["Mutual Information"]
    B --> B3["Chi-squared Test"]
    B --> B4["Correlation Filtering"]

    C --> C1["Recursive Feature Elimination"]
    C --> C2["Forward Selection"]
    C --> C3["Backward Elimination"]

    D --> D1["L1 / Lasso Regularization"]
    D --> D2["Tree-based Importance"]
    D --> D3["Elastic Net"]
```

**过滤法（Filter methods）** 使用统计指标独立地为每个特征打分。它们不训练模型。速度快，但会遗漏特征之间的交互。

**包装法（Wrapper methods）** 训练模型来评估特征子集。它们以模型性能作为评分标准。效果更好，但代价高昂，因为需要多次重训模型。

**嵌入法（Embedded methods）** 将特征选择作为模型训练的一部分。L1 正则化将权重压到零；决策树会按最有用的特征进行分裂。选择发生在拟合过程中，而不是作为一个独立步骤。

### 方差阈值

最简单的过滤法。如果一个特征在样本间几乎不变化，那它携带的信息就极少。

考虑一个特征，在 1000 个样本中有 999 个都是 0.0。它的方差接近零。没有任何模型能用它区分类别。删掉它。

```math
\text{variance}(x) = \text{mean}\bigl((x - \text{mean}(x))^2\bigr)
```

设置一个阈值（例如 0.01），删掉方差低于该阈值的所有特征。这能在完全不依赖目标变量的情况下去除常量或近常量特征。

何时使用：作为其他方法之前的预处理步骤。它以接近零的成本筛掉明显无用的特征。

局限：一个特征可以有很高的方差，却仍然是纯噪声。方差阈值是必要的，但不充分。

### 互信息

互信息衡量知道特征 $X$ 的值后，目标 $Y$ 的不确定性减少了多少。

```math
I(X; Y) = \sum_x \sum_y p(x, y) \cdot \log \frac{p(x, y)}{p(x) \cdot p(y)}
```

如果 $X$ 和 $Y$ 独立，则 $p(x, y) = p(x) \cdot p(y)$，对数项为零，因此 $I(X; Y) = 0$。$X$ 能告诉你关于 $Y$ 的信息越多，互信息就越高。

相比相关性的关键优势：互信息能捕捉非线性关系。一个特征可能与目标的相关性为零，但因为关系是二次或周期性的，互信息却很高。

对于连续特征，先离散到若干分箱（基于直方图估计）。分箱数量会影响估计——分箱太少会丢失信息，太多会引入噪声。常见选择：$\sqrt{n}$ 个分箱，或 Sturges 规则（$1 + \log_2 n$）。

```mermaid
flowchart LR
    A[Feature X] --> B[Discretize into Bins]
    B --> C["Compute Joint Distribution p(x,y)"]
    C --> D["Compute MI = sum p(x,y) * log(p(x,y) / p(x)p(y))"]
    D --> E["Rank Features by MI Score"]
    E --> F[Select Top K]
```

### 递归特征消除（RFE）

RFE 是一种包装法。它利用模型自身的特征重要性进行迭代剪枝：

1. 用全部特征训练模型
2. 按重要性为特征排序（线性模型用系数绝对值，树模型用不纯度减少量）
3. 移除最不重要的特征
4. 重复，直到剩下目标数量的特征

```mermaid
flowchart TD
    A["Start: All N Features"] --> B["Train Model"]
    B --> C["Rank Feature Importances"]
    C --> D["Remove Least Important"]
    D --> E{"Features == Target Count?"}
    E -->|No| B
    E -->|Yes| F["Return Selected Features"]
```

RFE 会考虑特征交互，因为模型同时看到所有剩余特征。移除一个特征会改变其他特征的重要性。这使它比过滤法更彻底。

代价：你需要训练模型 $N - \text{target}$ 次。如果有 500 个特征，目标是 10 个，那就是 490 次训练。对于昂贵的模型来说这很慢。可以通过每轮移除多个特征来加速（例如每轮移除最差的 10%）。

### L1（Lasso）正则化

L1 正则化将权重绝对值之和加到损失函数上：

```math
\text{loss} = \text{prediction\_error} + \alpha \cdot \sum \lvert w_i \rvert
```

alpha 参数控制剪枝的激进程度。alpha 越大，越多的权重会被精确地压到零。

为什么恰好是零？L1 惩罚在权重空间中形成了一个菱形约束区域。最优解往往落在这个菱形的角上，也就是一个或多个权重为零的位置。L2 正则化（岭回归）形成圆形约束，权重会缩小，但很少精确为零。

这就是嵌入法特征选择：模型在训练过程中学习忽略哪些特征。权重为零的特征实际上被移除了。

优点：只需一次训练，能处理相关特征（选一个，把其他的压到零），内置于大多数线性模型实现中。

局限：只适用于线性模型。无法捕捉非线性特征重要性。

### 基于树的特征重要性

决策树及其集成（随机森林、梯度提升）天然就会为特征排序。每次分裂都会减少不纯度（分类用 Gini 或熵，回归用方差）。带来更大不纯度减少的特征更重要。

对于有 $T$ 棵树的随机森林：

```math
\text{importance}(\text{feature}_j) = \frac{1}{T} \sum_{\text{all trees}} \; \sum_{\text{all nodes splitting on } \text{feature}_j} (\text{n\_samples} \cdot \text{impurity\_decrease})
```

这为每个特征给出了归一化重要性分数。它能自动处理非线性关系和特征交互。

注意：基于树的重要性偏向唯一值多的特征（高基数）。随机 ID 列会看起来很重要，因为它能完美分裂每个样本。应使用置换重要性作为校验。

### 置换重要性

一种模型无关的方法：

1. 训练模型，记录验证数据上的基线性能
2. 对每个特征：随机打乱其取值，测量性能下降幅度
3. 下降越大，特征越重要

如果打乱某个特征后性能没有下降，说明模型不依赖它。如果性能崩塌，那该特征至关重要。

置换重要性避免了基于树重要性的基数偏差。但它很慢：每个特征都要做一次完整评估，而且要重复多次以保证稳定性。

### 对比表

| 方法 | 类型 | 速度 | 非线性 | 特征交互 |
|--------|------|-------|-----------|---------------------|
| Variance threshold | 过滤法 | 非常快 | 否 | 否 |
| Mutual information | 过滤法 | 快 | 是 | 否 |
| Correlation filter | 过滤法 | 快 | 否 | 否 |
| RFE | 包装法 | 慢 | 取决于模型 | 是 |
| L1 / Lasso | 嵌入法 | 快 | 否（线性） | 否 |
| Tree importance | 嵌入法 | 中等 | 是 | 是 |
| Permutation importance | 模型无关 | 慢 | 是 | 是 |

### 决策流程图

```mermaid
flowchart TD
    A[Start: Feature Selection] --> B{How many features?}
    B -->|"< 50"| C["Start with variance threshold + mutual information"]
    B -->|"50-500"| D["Variance threshold, then L1 or tree importance"]
    B -->|"> 500"| E["Variance threshold, then mutual info filter, then RFE on survivors"]

    C --> F{Using linear model?}
    D --> F
    E --> F

    F -->|Yes| G["L1 regularization for final selection"]
    F -->|No - trees| H["Tree importance + permutation importance"]
    F -->|No - other| I["RFE with your model"]

    G --> J[Validate: compare selected vs all features]
    H --> J
    I --> J

    J --> K{Performance improved?}
    K -->|Yes| L["Ship with selected features"]
    K -->|No| M["Try different method or keep all features"]
```

```figure
f3-feature-prune
```

## 构建它

### 步骤 1：生成已知特征结构的合成数据

```python
import numpy as np


def make_feature_selection_data(n_samples=500, seed=42):
    rng = np.random.RandomState(seed)

    x1 = rng.randn(n_samples)
    x2 = rng.randn(n_samples)
    x3 = rng.randn(n_samples)
    x4 = x1 + 0.1 * rng.randn(n_samples)
    x5 = x2 + 0.1 * rng.randn(n_samples)

    informative = np.column_stack([x1, x2, x3, x4, x5])

    correlated = np.column_stack([
        x1 * 0.9 + 0.1 * rng.randn(n_samples),
        x2 * 0.8 + 0.2 * rng.randn(n_samples),
        x3 * 0.7 + 0.3 * rng.randn(n_samples),
        x1 * 0.5 + x2 * 0.5 + 0.1 * rng.randn(n_samples),
        x2 * 0.6 + x3 * 0.4 + 0.1 * rng.randn(n_samples),
    ])

    noise = rng.randn(n_samples, 10) * 0.5

    X = np.hstack([informative, correlated, noise])
    y = (2 * x1 - 1.5 * x2 + x3 + 0.5 * rng.randn(n_samples) > 0).astype(int)

    feature_names = (
        [f"info_{i}" for i in range(5)]
        + [f"corr_{i}" for i in range(5)]
        + [f"noise_{i}" for i in range(10)]
    )

    return X, y, feature_names
```

我们知道真实结构：特征 0-4 是有信息量的（其中 3 和 4 是 0 和 1 的相关副本），特征 5-9 与有信息量的特征相关，特征 10-19 是纯噪声。一个好的选择方法应该把 0-4 排得最高，把 10-19 排得最低。

### 步骤 2：方差阈值

```python
def variance_threshold(X, threshold=0.01):
    variances = np.var(X, axis=0)
    mask = variances > threshold
    return mask, variances
```

### 步骤 3：互信息（离散化实现）

```python
def discretize(x, n_bins=10):
    min_val, max_val = x.min(), x.max()
    if max_val == min_val:
        return np.zeros_like(x, dtype=int)
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    binned = np.digitize(x, bin_edges[1:-1])
    return binned


def mutual_information(X, y, n_bins=10):
    n_samples, n_features = X.shape
    mi_scores = np.zeros(n_features)

    y_vals, y_counts = np.unique(y, return_counts=True)
    p_y = y_counts / n_samples

    for f in range(n_features):
        x_binned = discretize(X[:, f], n_bins)
        x_vals, x_counts = np.unique(x_binned, return_counts=True)
        p_x = dict(zip(x_vals, x_counts / n_samples))

        mi = 0.0
        for xv in x_vals:
            for yi, yv in enumerate(y_vals):
                joint_mask = (x_binned == xv) & (y == yv)
                p_xy = np.sum(joint_mask) / n_samples
                if p_xy > 0:
                    mi += p_xy * np.log(p_xy / (p_x[xv] * p_y[yi]))
        mi_scores[f] = mi

    return mi_scores
```

### 步骤 4：递归特征消除

```python
def simple_logistic_importance(X, y, lr=0.1, epochs=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = pred - y
        w -= lr * (X.T @ error) / n_samples
        b -= lr * np.mean(error)

    return w, b


def rfe(X, y, n_features_to_select=5, lr=0.1, epochs=100):
    n_total = X.shape[1]
    remaining = list(range(n_total))
    rankings = np.ones(n_total, dtype=int)
    rank = n_total

    while len(remaining) > n_features_to_select:
        X_subset = X[:, remaining]
        w, _ = simple_logistic_importance(X_subset, y, lr, epochs)
        importances = np.abs(w)

        least_idx = np.argmin(importances)
        original_idx = remaining[least_idx]
        rankings[original_idx] = rank
        rank -= 1
        remaining.pop(least_idx)

    for idx in remaining:
        rankings[idx] = 1

    selected_mask = rankings == 1
    return selected_mask, rankings
```

### 步骤 5：L1 特征选择

```python
def soft_threshold(w, alpha):
    return np.sign(w) * np.maximum(np.abs(w) - alpha, 0)


def l1_feature_selection(X, y, alpha=0.1, lr=0.01, epochs=500):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = pred - y

        gradient_w = (X.T @ error) / n_samples
        gradient_b = np.mean(error)

        w -= lr * gradient_w
        w = soft_threshold(w, lr * alpha)
        b -= lr * gradient_b

    selected_mask = np.abs(w) > 1e-6
    return selected_mask, w
```

### 步骤 6：基于树的特征重要性（简单决策树）

```python
def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def best_split(X, y, feature_idx):
    values = np.unique(X[:, feature_idx])
    if len(values) <= 1:
        return None, -1.0

    best_threshold = None
    best_gain = -1.0
    parent_gini = gini_impurity(y)
    n = len(y)

    for i in range(len(values) - 1):
        threshold = (values[i] + values[i + 1]) / 2.0
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        if n_left == 0 or n_right == 0:
            continue

        gain = parent_gini - (n_left / n) * gini_impurity(y[left_mask]) - (n_right / n) * gini_impurity(y[right_mask])

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_threshold, best_gain


def tree_importance(X, y, n_trees=50, max_depth=5, seed=42):
    rng = np.random.RandomState(seed)
    n_samples, n_features = X.shape
    importances = np.zeros(n_features)

    for _ in range(n_trees):
        sample_idx = rng.choice(n_samples, size=n_samples, replace=True)
        feature_subset = rng.choice(n_features, size=max(1, int(np.sqrt(n_features))), replace=False)

        X_boot = X[sample_idx]
        y_boot = y[sample_idx]

        tree_imp = _build_tree_importance(X_boot, y_boot, feature_subset, max_depth)
        importances += tree_imp

    total = importances.sum()
    if total > 0:
        importances /= total

    return importances


def _build_tree_importance(X, y, feature_subset, max_depth, depth=0):
    n_features = X.shape[1]
    importances = np.zeros(n_features)

    if depth >= max_depth or len(np.unique(y)) <= 1 or len(y) < 4:
        return importances

    best_feature = None
    best_threshold = None
    best_gain = -1.0

    for f in feature_subset:
        threshold, gain = best_split(X, y, f)
        if gain > best_gain:
            best_gain = gain
            best_feature = f
            best_threshold = threshold

    if best_feature is None or best_gain <= 0:
        return importances

    importances[best_feature] += best_gain * len(y)

    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    importances += _build_tree_importance(X[left_mask], y[left_mask], feature_subset, max_depth, depth + 1)
    importances += _build_tree_importance(X[right_mask], y[right_mask], feature_subset, max_depth, depth + 1)

    return importances
```

### 步骤 7：运行所有方法并比较

代码文件在相同的合成数据集上运行上述五种方法，并打印对比表，展示每种方法选出了哪些特征。

## 使用它

使用 scikit-learn 时，特征选择可以直接集成到 pipeline 中：

```python
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    RFE,
    SelectFromModel,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

vt = VarianceThreshold(threshold=0.01)
X_filtered = vt.fit_transform(X)

mi_scores = mutual_info_classif(X, y)
top_k = np.argsort(mi_scores)[-10:]

rfe_selector = RFE(LogisticRegression(), n_features_to_select=10)
rfe_selector.fit(X, y)
X_rfe = rfe_selector.transform(X)

lasso_selector = SelectFromModel(Lasso(alpha=0.01))
lasso_selector.fit(X, y)
X_lasso = lasso_selector.transform(X)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
importances = rf.feature_importances_
```

从零实现展示了每种方法内部到底发生了什么。方差阈值只是计算 `var(X, axis=0)` 并应用掩码。互信息是在列联表中统计联合频率和边缘频率。RFE 是一个训练、排序、剪枝的循环。L1 是带软阈值步骤的梯度下降。树重要性是跨分裂累积不纯度减少量。没有魔法，只有统计和循环。

scikit-learn 版本增加了稳健性（例如 `mutual_info_classif` 使用 k-NN 密度估计而非分箱）、速度（C 实现）和 pipeline 集成。

## 交付它

本节课产出：
- `outputs/skill-feature-selector.md` —— 选择合适特征选择方法的快速参考决策树

## 练习

1. **前向选择**：实现与 RFE 相反的过程。从零个特征开始，每步加入最能提升模型性能的特征，直到加入特征不再有帮助。将选出的特征与 RFE 结果比较。哪个更快？哪个效果更好？

2. **稳定性选择**：运行 L1 特征选择 50 次，每次在随机 80% 数据子样本上用略有不同的 alpha 值。统计每个特征被选中的频率。在超过 80% 运行中被选中的特征是「稳定的」。将稳定特征与单次 L1 选择结果比较。哪个更可靠？

3. **多重共线性检测**：计算所有特征的相关矩阵。实现一个函数，给定相关性阈值（例如 0.9），从每对高度相关的特征中移除一个（保留与目标互信息更高的那个）。在合成数据集上测试，验证它能去除冗余的相关特征。

4. **特征选择 pipeline**：将方差阈值、互信息过滤和 RFE 串成一个 pipeline。先移除近零方差特征，再保留互信息最高的前 50%，最后对幸存者运行 RFE。将该 pipeline 与直接在全部特征上运行 RFE 比较。pipeline 更快吗？准确率是否相当？

5. **从零实现置换重要性**：实现置换重要性。对每个特征，打乱其取值 10 次，测量 F1 分数的平均下降。将排序与基于树的重要性比较。找出两者不一致的情况并解释原因（提示：相关特征）。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|----------------|----------------------|
| Filter method | "Score features independently" | 使用统计指标为特征独立打分、不训练模型、单独评估每个特征的特征选择方法 |
| Wrapper method | "Use the model to pick features" | 通过训练模型并以其性能作为选择标准来评估特征子集的方法 |
| Embedded method | "The model selects features during training" | 作为模型拟合一部分发生的特征选择，例如 L1 正则化将权重压到零 |
| Mutual information | "How much one variable tells you about another" | 已知 $X$ 后 $Y$ 不确定性减少程度的度量，能捕捉线性和非线性依赖 |
| Recursive Feature Elimination | "Train, rank, prune, repeat" | 迭代包装法：训练模型、移除最不重要特征，重复直到达到目标数量 |
| L1 / Lasso regularization | "Penalty that kills features" | 将权重绝对值之和加入损失函数，使不重要特征的权重精确为零 |
| Variance threshold | "Remove constant features" | 删除样本间方差低于指定阈值的特征，过滤掉不携带信息的特征 |
| Feature importance | "Which features matter most" | 指示每个特征对模型预测贡献程度的分数，来自分裂增益（树）或系数大小（线性） |
| Permutation importance | "Shuffle and measure the damage" | 随机打乱每个特征的取值，测量模型性能下降幅度，从而评估特征重要性 |
| Curse of dimensionality | "Too many features, not enough data" | 特征增加使特征空间体积指数级膨胀、数据稀疏、距离失去意义的现象 |

## 延伸阅读

- [An Introduction to Variable and Feature Selection (Guyon & Elisseeff, 2003)](https://jmlr.org/papers/v3/guyon03a.html) —— 特征选择方法的基础性综述，至今仍被广泛引用
- [scikit-learn Feature Selection Guide](https://scikit-learn.org/stable/modules/feature_selection.html) —— 过滤法、包装法和嵌入法的实用参考，附代码示例
- [Stability Selection (Meinshausen & Buhlmann, 2010)](https://arxiv.org/abs/0809.2932) —— 将子采样与特征选择结合，获得稳健、可复现的结果
- [Beware Default Random Forest Importances (Strobl et al., 2007)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25) —— 证明基于树重要性的基数偏差，并提出条件重要性作为替代方案
