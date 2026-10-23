# 朴素贝叶斯

> 「朴素」假设在数学上是错的，但它依然能工作。这正是它的美妙之处。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 2, Lessons 01-07 (classification, Bayes' theorem)
**Time:** ~75 minutes

## 学习目标

- 从零实现带 Laplace 平滑的多项式朴素贝叶斯（Multinomial Naive Bayes），用于文本分类
- 解释为什么朴素独立性假设在数学上不正确，却能在实践中产生正确的类别排序
- 比较多项式、伯努利和高斯三种朴素贝叶斯变体，并为给定特征类型选择合适的一种
- 在高维稀疏数据上评估朴素贝叶斯与逻辑回归，并解释其中的偏差-方差权衡

## 问题

你需要对文本进行分类。把邮件分成垃圾邮件或非垃圾邮件，把用户评论分成正面或负面，把支持工单分成不同类别。你有成千上万个特征（每个词一个），而训练数据却很有限。

大多数分类器在这里都会捉襟见肘。逻辑回归需要足够的样本来可靠地估计数千个权重。决策树每次只根据一个词进行拆分，极易过拟合。KNN 在 10,000 维空间里毫无意义，因为每个点之间的距离都差不多。

朴素贝叶斯能解决这个问题。它做了一个数学上错误的假设（即每个特征在给定类别下彼此独立），但在文本分类上仍然优于那些「更聪明」的模型，尤其是在训练集很小的时候。它只需对数据单次遍历即可完成训练。它能扩展到数百万个特征。它还能输出概率估计（尽管由于独立性假设，这些概率通常校准得不好）。

理解为什么错误的假设能带来好的预测，会让你学到机器学习中一个根本道理：最好的模型不是最正确的那个，而是在你的数据上偏差-方差权衡最优的那个。

## 概念

### 贝叶斯定理（快速回顾）

贝叶斯定理翻转了条件概率：

```math
P(\text{class} \mid \text{features}) = \frac{P(\text{features} \mid \text{class}) \cdot P(\text{class})}{P(\text{features})}
```

我们想要的是 `P(class | features)`——给定文档中的词，该文档属于某个类别的概率。它可以通过以下部分计算：
- `P(features | class)`——在该类别文档中看到这些词的可能性
- `P(class)`——类别的先验概率（垃圾邮件整体有多常见？）
- `P(features)`——证据，对所有类别都相同，因此在比较时可以忽略

`P(class | features)` 最高的类别获胜。

### 朴素独立性假设

精确计算 `P(features | class)` 需要估计所有特征联合分布。如果词汇量有 10,000 个词，你就需要估计 $2^{10{,}000}$ 种可能组合的分布。这不可能。

朴素假设：每个特征在给定类别下条件独立。

```math
P(w_1, w_2, \ldots, w_n \mid \text{class}) = P(w_1 \mid \text{class}) \cdot P(w_2 \mid \text{class}) \cdots P(w_n \mid \text{class})
```

不必估计一个不可能的联合分布，你只需估计 $n$ 个简单的单特征分布。每一个都只需要一个计数。

这个假设显然是错的。在任何文档中，「machine」和「learning」都不是独立的。但分类器不需要正确的概率估计，它只需要正确的排序——哪个类别的概率最高。独立性假设会引入系统性误差，但这些误差对所有类别影响相似，因此排序仍然正确。

### 为什么它仍然有效

三个原因：

1. **排序优于校准。** 分类只需要最高排名的类别正确即可。即使模型估计 $P(\text{spam}) = 0.99999$，而真实概率是 0.7，分类器仍然正确地选出 spam。我们不需要正确的概率，只需要正确的赢家。

2. **高偏差，低方差。** 独立性假设是一个很强的先验。它严重约束了模型，从而防止过拟合。在训练数据有限时，一个稍微错误但稳定的模型，会击败一个理论上正确但极不稳定的模型。这就是偏差-方差权衡的实际体现。

3. **特征冗余会相互抵消。** 相关特征提供了冗余证据。分类器会重复计算这些证据，但它对正确类别也会重复计算。如果「machine」和「learning」总是一起出现，两者都会为「tech」类别提供证据。朴素贝叶斯把它们算了两次，但也是对正确的类别算了两次。

第四个实用的原因：朴素贝叶斯极快。训练只是单次遍历数据统计频次，预测只是矩阵乘法。你可以在几秒钟内训练百万级文档。这种速度意味着你可以更快迭代、尝试更多特征集、运行更多实验。

### 逐步数学推导

让我们通过一个具体例子来跟踪计算。假设有两个类别：spam 和 not-spam。词汇表有三个词：「free」、「money」、「meeting」。

训练数据：
- Spam 邮件提到「free」80 次，「money」60 次，「meeting」10 次（共 150 个词）
- 非 Spam 邮件提到「free」5 次，「money」10 次，「meeting」100 次（共 115 个词）
- 40% 的邮件是 spam，60% 不是 spam

使用 Laplace 平滑（$\alpha=1$）：

```math
\begin{aligned}
P(\text{free} \mid \text{spam}) &= (80 + 1) / (150 + 3) = 81/153 = 0.529 \\
P(\text{money} \mid \text{spam}) &= (60 + 1) / (150 + 3) = 61/153 = 0.399 \\
P(\text{meeting} \mid \text{spam}) &= (10 + 1) / (150 + 3) = 11/153 = 0.072 \\[6pt]
P(\text{free} \mid \text{not-spam}) &= (5 + 1) / (115 + 3) = 6/118 = 0.051 \\
P(\text{money} \mid \text{not-spam}) &= (10 + 1) / (115 + 3) = 11/118 = 0.093 \\
P(\text{meeting} \mid \text{not-spam}) &= (100 + 1) / (115 + 3) = 101/118 = 0.856
\end{aligned}
```

新邮件包含：「free」2 次，「money」1 次，「meeting」0 次。

```math
\begin{aligned}
\log P(\text{spam} \mid \text{email}) &= \log(0.4) + 2 \log(0.529) + 1 \log(0.399) + 0 \log(0.072) \\
&= -0.916 + 2(-0.637) + (-0.919) + 0 \\
&= -3.109 \\[6pt]
\log P(\text{not-spam} \mid \text{email}) &= \log(0.6) + 2 \log(0.051) + 1 \log(0.093) + 0 \log(0.856) \\
&= -0.511 + 2(-2.976) + (-2.375) + 0 \\
&= -8.838
\end{aligned}
```

Spam 以很大优势获胜。「free」出现两次是 spam 的强烈证据。注意，「meeting」没有出现对两个对数和都没有贡献（$0 \cdot \log P$）——在多项式朴素贝叶斯中，缺失的词没有影响。而伯努利朴素贝叶斯会显式地建模词的缺失。

### 三种变体

朴素贝叶斯有三种形式。每种对 `P(feature | class)` 的建模方式不同。

#### 多项式朴素贝叶斯

把每个特征建模为计数。最适合特征是词频或 TF-IDF 值的文本数据。

```math
P(\text{word}_i \mid \text{class}) = \frac{\text{count of } \text{word}_i \text{ in class} + \alpha}{\text{total words in class} + \alpha \cdot \text{vocab\_size}}
```

`alpha` 就是 Laplace 平滑（见下文解释）。这种变体是文本分类的主力军。

#### 高斯朴素贝叶斯

把每个特征建模为正态分布。最适合连续特征。

```math
P(x_i \mid \text{class}) = \frac{1}{\sqrt{2 \pi \cdot \text{var}}} \cdot \exp\left(-\frac{(x_i - \text{mean})^2}{2 \cdot \text{var}}\right)
```

每个类别对每个特征都有自己的均值和方差。当特征在每个类别内部确实近似呈钟形分布时，效果很好。

#### 伯努利朴素贝叶斯

把每个特征建模为二值（出现或未出现）。最适合短文本或二值特征向量。

```math
P(\text{word}_i \mid \text{class}) = \frac{\text{docs in class containing } \text{word}_i + \alpha}{\text{total docs in class} + 2 \alpha}
```

与多项式不同，伯努利会显式地惩罚词的缺失。如果「free」通常出现在 spam 中，但这封邮件里没有「free」，伯努利会把它算作反对 spam 的证据。

### 何时使用哪种变体

| 变体 | 特征类型 | 最适合 | 示例 |
|---------|-------------|----------|---------|
| Multinomial | 计数或频次 | 文本分类、词袋 | 垃圾邮件、主题分类 |
| Gaussian | 连续值 | 特征近似正态分布的表格数据 | Iris 分类、传感器数据 |
| Bernoulli | 二值（0/1） | 短文本、二值特征向量 | 短信垃圾、出现/缺失特征 |

### Laplace 平滑

如果一个词在训练数据的某个类别中从未出现，但在测试数据中出现了，会发生什么？

没有平滑时：`P(word | class) = 0/N = 0`。一个零值乘以整个乘积，会让 `P(class | features) = 0`，无论其他证据有多强。一个未见过的词就能毁掉整个预测。

Laplace 平滑给每个特征计数都加上一个小的 `alpha`（通常为 1）：

```math
P(\text{word}_i \mid \text{class}) = \frac{\text{count}(\text{word}_i, \text{class}) + \alpha}{\text{total\_words\_in\_class} + \alpha \cdot \text{vocab\_size}}
```

当 $\alpha=1$ 时，每个词都至少有一个很小的概率。测试邮件中出现「discombobulate」不会再把 spam 概率清零。平滑有一种贝叶斯解释：它等价于在词分布上放置一个均匀的 Dirichlet 先验。

更高的 alpha 意味着更强的平滑（分布更均匀）。更低的 alpha 意味着模型更信任数据。Alpha 是一个需要调优的超参数。

Alpha 的影响：

| Alpha | 影响 | 何时使用 |
|-------|--------|-------------|
| 0.001 | 几乎不平滑，信任数据 | 训练集很大，预计没有未见特征 |
| 0.1 | 轻度平滑 | 训练集较大 |
| 1.0 | 标准 Laplace 平滑 | 默认起点 |
| 10.0 | 强平滑，压平分布 | 训练集很小，预计有很多未见特征 |

### 对数空间计算

把数百个概率（每个都小于 1）相乘会导致浮点数下溢。乘积在浮点数中会变为 0，即使真实值是一个很小的正数。

解决方法：在对数空间工作。不乘概率，而是把它们的对数相加：

```math
\log P(\text{class} \mid x_1, x_2, \ldots, x_n) = \log P(\text{class}) + \sum_i \log P(x_i \mid \text{class})
```

这样预测就变成了点积：

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

矩阵乘法。这就是朴素贝叶斯预测如此快的原因——它与单层线性模型的运算相同。

### 朴素贝叶斯 vs 逻辑回归

两者都是文本的线性分类器。区别在于它们建模的对象。

| 方面 | 朴素贝叶斯 | 逻辑回归 |
|--------|------------|-------------------|
| 类型 | 生成式（建模 $P(X \mid Y){}$） | 判别式（建模 $P(Y \mid X){}$） |
| 训练 | 统计频次 | 优化损失函数 |
| 小数据 | 更好（强先验有帮助） | 更差（不足以估计权重） |
| 大数据 | 更差（错误假设拖后腿） | 更好（灵活边界） |
| 特征 | 假设独立 | 能处理相关性 |
| 速度 | 单次遍历，极快 | 迭代优化 |
| 校准 | 概率较差 | 概率更好 |

经验法则：先用朴素贝叶斯。如果数据充足且 NB 遇到瓶颈，再切换到逻辑回归。

### 分类流程

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

在实践中，我们在对数空间工作以避免浮点数下溢。不乘许多小概率，而是把其对数相加：

```math
\log P(\text{class} \mid \text{features}) = \log P(\text{class}) + \sum_i \log P(\text{feature}_i \mid \text{class})
```

```figure
naive-bayes
```

## 构建它

`code/naive_bayes.py` 中的代码从零实现了 MultinomialNB 和 GaussianNB。

### MultinomialNB

从零实现的步骤：

1. **fit(X, y)**：对每个类别，统计每个特征的频次。加上 Laplace 平滑。计算对数概率。保存类别先验（类别频次的对数）。

2. **predict_log_proba(X)**：对每个样本，计算所有类别的 $\log P(\text{class})$ 加上所有 $\log P(\text{feature}_i \mid \text{class})$ 之和。这是一个矩阵乘法：X @ log_probs.T + log_priors。

3. **predict(X)**：返回对数概率最高的类别。

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

关键洞察：拟合之后，预测只是矩阵乘法加一个偏置。这就是朴素贝叶斯这么快的原因。

### GaussianNB

对于连续特征，我们估计每个类别每个特征的均值和方差：

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

预测使用每个特征的高斯 PDF，跨特征相乘（在对数空间中相加）。

### 演示：文本分类

代码生成模拟两个类别（科技文章 vs 体育文章）的合成词袋数据。每个类别有不同的词频分布。MultinomialNB 用词计数对它们分类。

合成数据的工作原理：创建 200 个「词」（特征列）。词 0-39 在科技文章中高频、在体育文章中低频。词 80-119 在体育文章中高频、在科技文章中低频。词 40-79 在两类中都是中等频次。这创造了一个真实场景：有些词是强类别指示词，有些只是噪声。

### 演示：连续特征

代码生成类似 Iris 的数据（3 个类别、4 个特征、高斯簇）。GaussianNB 使用每个类别的均值和方差进行分类。每个类别有不同的中心（均值向量）和不同的散布（方差），模拟了真实世界中不同类别测量值系统性地不同的场景。

代码还演示了：
- **平滑对比**：用不同的 alpha 值训练 MultinomialNB，展示平滑强度对准确率的影响。
- **训练规模实验**：当训练数据从 20 增加到 1600 个样本时，NB 的准确率如何提升。NB 即使在样本很少时也能达到不错的准确率——这是它的主要优势。
- **混淆矩阵**：每个类别的精确率、召回率和 F1 分数，展示 NB 在哪里出错。

### 预测速度

朴素贝叶斯预测是矩阵乘法。对于 $n$ 个样本、$d$ 个特征、$k$ 个类别：
- MultinomialNB：一次矩阵乘法 $(n \times d) \mathbin{@} (d \times k) = O(n \cdot d \cdot k)$
- GaussianNB：$n \cdot k$ 次高斯 PDF 求值，每次覆盖 $d$ 个特征 = $O(n \cdot d \cdot k)$

两者在每个维度上都是线性的。相比之下，KNN 需要计算到所有训练点的距离，带 RBF 核的 SVM 需要对所有支持向量求核。NB 在预测时间上快了几个数量级。

## 使用它

使用 sklearn，两种变体都是一行代码：

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

使用 sklearn 进行文本分类：

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

`naive_bayes.py` 中的代码将从头实现与 sklearn 在同一数据上的结果进行对比，以验证正确性。

### 将 TF-IDF 与朴素贝叶斯结合

原始词计数让每个词每次出现权重相同。但像「the」和「is」这样在每个类别中都很常见的词没有信息量。TF-IDF（词频-逆文档频率）会降低常见词的权重，提高罕见、有区分力的词的权重。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF 值非负，因此可以与 MultinomialNB 配合。TF-IDF + MultinomialNB 的组合是文本分类最强基线之一。在训练样本少于 10,000 的数据集上，它经常击败更复杂的模型。

### 短文本使用 BernoulliNB

对于短文本（推文、短信、聊天消息），BernoulliNB 可能比 MultinomialNB 更好。短文本词频很低，MultinomialNB 依赖的频次信息很嘈杂。BernoulliNB 只关心词是否出现，这在短文本中更可靠。

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

CountVectorizer 中的 `binary=True` 会把所有计数转为 0/1。如果不设置，BernoulliNB 仍然能运行，但它看到的是它未设计的计数。

### 校准 NB 概率

NB 的概率校准较差。当 NB 说 $P(\text{spam}) = 0.95$ 时，真实概率可能是 0.7。如果你需要可靠的概率估计（例如，设置阈值或与其他模型组合），使用 sklearn 的 CalibratedClassifierCV：

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

这会在 NB 原始分数之上用交叉验证拟合一个逻辑回归。得到的结果概率更接近真实的类别频率。

### 常见陷阱

1. **负特征值。** MultinomialNB 要求特征非负。如果你有负值（例如某些设置下的 TF-IDF 或标准化特征），改用 GaussianNB，或将特征平移为正。

2. **零方差特征。** GaussianNB 要除以方差。如果某个特征在某个类别内方差为 0（所有值相同），概率计算会崩溃。代码会给所有方差加一个很小的平滑项（1e-9）来防止这种情况。

3. **类别不平衡。** 如果 99% 的邮件不是 spam，先验 $P(\text{not-spam}) = 0.99$ 会强到压过似然证据。你可以手动设置类别先验，或使用 sklearn 的 class_prior 参数。

4. **特征缩放。** MultinomialNB 不需要缩放（它作用于计数）。GaussianNB 也不需要缩放（它估计每个特征自身的统计量）。这是它相对于逻辑回归和 SVM 的优势，后两者对特征尺度敏感。

## 交付它

本节课产出：
- `outputs/skill-naive-bayes-chooser.md`——选择合适 NB 变体的决策技能
- `code/naive_bayes.py`——从零实现的 MultinomialNB 和 GaussianNB，并与 sklearn 对比

### 朴素贝叶斯何时失效

当独立性假设导致错误排序（而不仅仅是错误概率）时，NB 会失效。这种情况发生在：

1. **强特征交互。** 如果类别依赖于两个特征的组合，而不是单独某个特征（类似 XOR 模式），NB 会完全错过。每个特征单独看都没有证据，而 NB 无法非线性地组合它们。

2. **高度相关但证据相反的 correlated 特征。** 如果特征 A 指向 spam，特征 B 指向 not-spam，但 A 和 B 实际上完全相关（现实中总是一致），NB 会看到不存在的冲突证据。

3. **非常大的训练集。** 数据足够多时，判别式模型如逻辑回归能学到真正的决策边界并超过 NB。在小数据时有帮助的独立性假设，现在会限制模型。

在实践中，这些失效模式在文本分类中很少见。文本特征数量多、单个弱，独立性假设的误差往往会抵消。对于特征少且高度相关的表格数据，优先考虑逻辑回归或树模型。

## 练习

1. **平滑实验。** 在文本数据上用 alpha 取 0.01、0.1、1.0、10.0、100.0 训练 MultinomialNB。绘制准确率随 alpha 变化的曲线。性能在哪里达到峰值？为什么很高的 alpha 会损害性能？

2. **特征独立性测试。** 取一个真实文本数据集。挑选两个明显相关的词（如「machine」和「learning」）。计算 $P(\text{word}_1 \mid \text{class}) \cdot P(\text{word}_2 \mid \text{class})$，并与 $P(\text{word}_1 \text{ AND } \text{word}_2 \mid \text{class})$ 比较。独立性假设错得有多离谱？它会影响分类准确率吗？

3. **Bernoulli 实现。** 扩展代码，增加一个 BernoulliNB 类。把词袋转为二值（出现/缺失），并在文本数据上与 MultinomialNB 比较准确率。BernoulliNB 何时胜出？

4. **NB vs 逻辑回归。** 在文本数据上训练两者。从 100 个训练样本开始，增加到 10,000。绘制两者准确率随训练集大小变化的曲线。逻辑回归在什么时候超过朴素贝叶斯？

5. **垃圾邮件过滤器。** 构建一个完整的垃圾邮件分类器：对原始邮件文本分词、构建词汇表、创建词袋特征、训练 MultinomialNB、用精确率和召回率评估（不只是准确率——为什么？）。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| Naive Bayes | 「简单的概率分类器」 | 应用贝叶斯定理并假设特征在给定类别下条件独立的分类器 |
| Conditional independence | 「特征互不影响」 | $P(A, B \mid C) = P(A \mid C) \cdot P(B \mid C)$——一旦知道 $C$，$B$ 对 $A$ 没有额外信息 |
| Laplace smoothing | 「加一平滑」 | 给每个特征计数加一个小值，防止零概率主导预测 |
| Prior | 「看到数据之前的信念」 | $P(\text{class})$——在观察任何特征之前每个类别的概率 |
| Likelihood | 「数据拟合程度」 | $P(\text{features} \mid \text{class})$——已知类别时观察到这些特征的概率 |
| Posterior | 「看到数据之后的信念」 | $P(\text{class} \mid \text{features})$——观察到特征后更新得到的类别概率 |
| Generative model | 「建模数据如何生成」 | 学习 $P(X \mid Y)$ 和 $P(Y)$，再用贝叶斯定理得到 $P(Y \mid X)$ 的模型 |
| Discriminative model | 「建模决策边界」 | 直接学习 $P(Y \mid X)$，而不建模 $X$ 如何生成的模型 |
| Log probability | 「避免下溢」 | 用 $\log P$ 而不是 $P$ 工作，防止许多小数的乘积在浮点数中变成零 |

## 延伸阅读

- [scikit-learn Naive Bayes 文档](https://scikit-learn.org/stable/modules/naive_bayes.html)——三种变体及其数学细节
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf)——Multinomial 与 Bernoulli 用于文本分类的经典比较
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)——针对文本的 NB 改进
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf)——证明在数据较少时 NB 比逻辑回归收敛更快
