# 范数与距离

> 你的距离函数定义了「相似」的含义。选错了，下游所有东西都会崩掉。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**Time:** ~90 minutes

## 学习目标

- 从零实现 L1、L2、余弦、马氏距离、Jaccard 和编辑距离函数
- 为给定 ML 任务选择合适的距离度量，并解释为什么替代方案会失败
- 把 L1 和 L2 范数与 LASSO、Ridge 正则化及其几何约束区域联系起来
- 演示同一数据集在不同度量下如何产生不同的最近邻

## 问题

你有两个向量。也许是词嵌入，也许是用户画像，也许是像素数组。你需要知道：它们有多接近？

答案完全取决于你选的距离函数。两个数据点可能在一个度量下是最近邻，在另一个度量下却相隔很远。你的 KNN 分类器、推荐引擎、向量数据库、聚类算法、损失函数——全都依赖这个选择。选错，模型就会朝错误的方向优化。

不存在 universally 最好的距离。L2 适合空间数据。余弦相似度主导 NLP。Jaccard 处理集合。编辑距离处理字符串。马氏距离考虑相关性。Wasserstein 搬动概率质量。每一种都编码了对「相似」含义的不同假设。

本节课从零构建每一个主要距离函数，告诉你什么时候该用哪一种，并演示同一数据在不同度量下如何产生完全不同的最近邻。

## 概念

### 范数：度量向量大小

范数度量向量的「大小」。两个向量之间的任意距离函数都可以写成它们差值的范数：$d(a, b) = \lVert a - b \rVert$。所以理解范数就是理解距离。

### L1 范数（曼哈顿距离）

L1 范数对所有分量的绝对值求和。

```math
\lVert x \rVert_1 = |x_1| + |x_2| + \cdots + |x_n|
```

它被称为曼哈顿距离，因为它度量你在城市网格上只能沿坐标轴行走的距离。不能走对角线。

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

何时使用 L1：
- 高维稀疏数据（文本特征、one-hot 编码）
- 想要对异常值更稳健（单个巨大差异不会主导结果）
- 特征选择问题（L1 正则化促进稀疏性）

与 L1 正则化（Lasso）的联系：在损失函数中加上 $\lVert w \rVert_1$ 会对权重绝对值之和进行惩罚。这会把小权重恰好推到 0，实现自动特征选择。L1 惩罚在权重空间中形成菱形约束区域，菱形的角落在坐标轴上，对应某些权重为 0。

与损失函数的联系：平均绝对误差（MAE）就是预测与目标之间 L1 距离的平均。它线性惩罚所有误差，因此比 MSE 更稳健。

### L2 范数（欧几里得距离）

L2 范数是直线距离，即各分量平方和的开方。

```math
\lVert x \rVert_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}
```

这就是你在几何课上学过的距离，$n$ 维空间中的勾股定理。

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

何时使用 L2：
- 低维到中等维度的连续数据
- 各特征尺度可比
- 物理距离（空间数据、传感器读数）
- 像素级图像相似度

与 L2 正则化（Ridge）的联系：在损失函数中加上 $\lVert w \rVert_2^2$ 会对大权重进行惩罚。与 L1 不同，它不会把权重推到 0，而是按比例把所有权重往 0 收缩。L2 惩罚形成圆形约束区域，所以坐标轴上没有角，权重会变小但很少恰好为 0。

与损失函数的联系：均方误差（MSE）是 L2 距离平方的平均。平方会放大对大误差的惩罚。

```math
\begin{aligned}
\text{MAE (L1 loss):} \quad & |y - \hat{y}| && \text{Linear penalty. Robust to outliers.} \\
\text{MSE (L2 loss):} \quad & (y - \hat{y})^2 && \text{Quadratic penalty. Sensitive to outliers.}
\end{aligned}
```

### Lp 范数：通用族

L1 和 L2 都是 Lp 范数的特例：

```math
\lVert x \rVert_p = (|x_1|^p + |x_2|^p + \cdots + |x_n|^p)^{1/p}
```

不同的 $p$ 会产生不同形状的「单位球」（到原点距离为 1 的所有点的集合）：

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-无穷范数（切比雪夫距离）

当 $p$ 趋于无穷时，Lp 范数收敛为最大绝对分量。

```math
\lVert x \rVert_\infty = \max(|x_1|, |x_2|, \ldots, |x_n|)
```

两点之间的距离由它们差异最大的那个维度决定，其他维度被忽略。

```math
\begin{aligned}
\text{Point } A &= (1, 1) \\
\text{Point } B &= (4, 5) \\[6pt]
L_\infty \text{ distance} &= \max(|4-1|, |5-1|) = \max(3, 4) = 4
\end{aligned}
```

何时使用 L-无穷：
- 任意单个维度的最坏偏差很重要时
- 棋盘（国际象棋中的王按 L-无穷 移动：朝任何方向走一步代价都是 1）
- 制造公差（每个维度都必须在规格内）

### 余弦相似度与余弦距离

余弦相似度度量两个向量之间的夹角，忽略它们的大小。

```math
\text{cos\_sim}(a, b) = \frac{a \cdot b}{\lVert a \rVert_2 \, \lVert b \rVert_2}
```

取值范围从 -1（方向相反）到 +1（方向相同）。垂直向量的余弦相似度为 0。

余弦距离把它转成距离：cosine_distance = 1 - cosine_similarity。范围从 0（方向相同）到 2（方向相反）。

```math
\begin{aligned}
& a = (1, 0) \qquad b = (1, 1) \\[6pt]
& \text{cos\_sim} = \frac{1 \cdot 1 + 0 \cdot 1}{1 \cdot \sqrt{2}} = \frac{1}{\sqrt{2}} = 0.707 \\
& \text{cos\_dist} = 1 - 0.707 = 0.293
\end{aligned}
```

为什么余弦相似度主导 NLP 和嵌入：在文本中，文档长度不应影响相似度。一篇关于猫的长文档和另一篇关于猫的短文档仍应被视为「相似」。余弦相似度忽略大小（长度），只关心方向。两篇词分布相同但长度不同的文档指向同一方向，余弦相似度为 1.0。

何时使用余弦相似度：
- 文本相似度（TF-IDF 向量、词嵌入、句子嵌入）
- 任何大小是噪声、方向是信号的领域
- 推荐系统（用户偏好向量）
- 嵌入搜索（向量数据库几乎总是用余弦或点积）

### 点积相似度 vs 余弦相似度

两个向量的点积为：

```math
\begin{aligned}
a \cdot b &= a_1 b_1 + a_2 b_2 + \cdots + a_n b_n \\
&= \lVert a \rVert \, \lVert b \rVert \cos(\text{angle})
\end{aligned}
```

余弦相似度是用两边大小归一化后的点积。当两个向量已经单位化（大小为 1）时，点积与余弦相似度完全相同。

```math
\text{If } \lVert a \rVert = 1 \text{ and } \lVert b \rVert = 1: \quad a \cdot b = \cos(\text{angle between } a \text{ and } b)
```

当它们不同时：点积包含大小信息。大小更大的向量会得到更高的点积分。在某些检索系统中这很重要，因为你希望「热门」项目排名更高。大小充当隐式的质量或重要性信号。

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

实践中：
- 想要纯方向相似度时用余弦相似度
- 大小携带有意义信息时用点积
- 许多向量数据库（Pinecone、Weaviate、Qdrant）允许你在两者间选择
- 如果嵌入已经 L2 归一化，选哪个都一样

### 马氏距离

欧几里得距离对所有维度一视同仁。但如果特征相关或尺度不同，L2 会给出误导性结果。

马氏距离考虑数据的协方差结构。

```math
d_M(x, y) = \sqrt{(x - y)^T S^{-1} (x - y)}
```

其中 $S$ 是数据的协方差矩阵。

直观理解：马氏距离先把数据去相关并归一化（白化），然后在变换后的空间里计算 L2 距离。如果 $S$ 是单位矩阵（特征不相关且方差为 1），马氏距离就退化为欧几里得距离。

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

何时使用马氏距离：
- 异常值检测（与均值马氏距离很大的点是异常值）
- 特征尺度不同且相关时的分类
- 有足够数据估计可靠协方差矩阵时
- 制造业质量控制（多变量过程监控）

### Jaccard 相似度（用于集合）

Jaccard 相似度度量两个集合的交叠程度。

```math
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
```

取值范围从 0（无交叠）到 1（完全相同）。Jaccard 距离 = 1 - Jaccard 相似度。

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

何时使用 Jaccard：
- 比较标签、类别或特征集合
- 基于词语存在（而非频率）的文档相似度
- 近似重复检测（MinHash 对 Jaccard 的近似）
- 比较二元特征向量（存在/缺失数据）
- 评估分割模型（IoU = Jaccard）

### 编辑距离（Levenshtein 距离）

编辑距离统计把一个字符串转成另一个字符串所需的最少单字符操作次数。操作包括插入、删除和替换。

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

用动态规划计算。填充一个矩阵，其中 $(i, j)$ 项是字符串 A 前 $i$ 个字符与字符串 B 前 $j$ 个字符之间的编辑距离。

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

何时使用编辑距离：
- 拼写检查与纠正
- DNA 序列比对（带加权操作）
- 模糊字符串匹配
- 杂乱文本数据去重

### KL 散度（不是距离，但被当作距离用）

KL 散度度量一个概率分布与另一个概率分布的差异。第 09 课已覆盖，但它属于这里的讨论，因为人们常把它当作「距离」，尽管它并不是。

```math
D_{KL}(P \,\|\, Q) = \sum_x p(x) \log\frac{p(x)}{q(x)}
```

关键性质：KL 散度不对称。

```math
D_{KL}(P \,\|\, Q) \ne D_{KL}(Q \,\|\, P)
```

这意味着它连距离度量的基本要求都不满足。它也不满足三角不等式。它是散度，不是距离。

前向 KL $(D_{KL}(P \Vert Q))$ 是「均值寻找」：$Q$ 试图覆盖 $P$ 的所有模态。
反向 KL $(D_{KL}(Q \Vert P))$ 是「模态寻找」：$Q$ 聚焦于 $P$ 的单个模态。

何时见到 KL 散度：
- VAE（ELBO 中的 KL 项把隐变量分布推向先验）
- 知识蒸馏（学生试图匹配教师分布）
- RLHF（KL 惩罚让微调模型靠近基础模型）
- 策略梯度方法（约束策略更新）

### Wasserstein 距离（推土机距离）

Wasserstein 度量把一个概率分布变成另一个所需的最小「工作量」。想象成：一个分布是一堆土，另一个是一个坑，你需要搬多少土、搬多远？

```math
W(P, Q) = \inf_{\gamma} \mathbb{E}[d(x, y)] \quad \text{(over all transport plans } \gamma\text{)}
```

对一维分布，它简化为两个累积分布函数绝对差的积分：

```math
W_1(P, Q) = \int |\text{CDF}_P(x) - \text{CDF}_Q(x)| \, dx
```

Wasserstein 为何重要：
- 它是真正的度量（对称，满足三角不等式）
- 即使分布不重叠也能提供梯度（KL 散度会趋于无穷）
- 这一性质让它成为 Wasserstein GAN（WGAN）的核心，解决了原始 GAN 的训练不稳定问题

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

何时使用 Wasserstein：
- GAN 训练（WGAN、WGAN-GP）
- 比较可能不重叠的分布
- 最优传输问题
- 图像检索（比较颜色直方图）

### 为什么不同任务需要不同距离

| 任务 | 最佳距离 | 原因 |
|------|----------|------|
| 文本相似度 | 余弦 | 大小是噪声，方向是语义 |
| 图像像素比较 | L2 | 空间关系重要，特征尺度可比 |
| 稀疏高维特征 | L1 | 稳健，不会放大罕见的大差异 |
| 集合交叠（标签、类别） | Jaccard | 数据天然是集合，不是向量 |
| 字符串匹配 | 编辑距离 | 操作对应人类编辑直觉 |
| 异常值检测 | 马氏距离 | 考虑特征相关性与尺度 |
| 比较分布 | KL 散度 | 度量用 $Q$ 代替 $P$ 所损失的信息 |
| GAN 训练 | Wasserstein | 即使分布不重叠也能提供梯度 |
| 嵌入（向量数据库） | 余弦或点积 | 嵌入被训练成用方向编码语义 |
| 推荐系统 | 点积 | 大小可编码流行度或置信度 |
| DNA 序列 | 加权编辑距离 | 不同核苷酸对的替换代价不同 |
| 制造 QC | L-无穷 | 任意维度的最坏偏差很重要 |

### 与损失函数的联系

损失函数就是施加在预测与目标上的距离函数。

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### 与正则化的联系

正则化在损失函数中对权重施加范数惩罚。

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

为什么 L1 产生稀疏性而 L2 不：在二维权重空间中画出约束区域。L1 是菱形，L2 是圆形。损失函数的等高线（椭圆）最可能在菱形的角落与之相切，而角落处某个权重为 0。它们与圆在光滑点相切，两个权重都非零。

### 最近邻搜索

每个距离函数都隐含一个最近邻搜索问题：给定查询点，在数据集中找到最近的点。

精确最近邻搜索在包含 $n$ 个 $d$ 维点的数据集中每次查询是 $O(n \cdot d)$。对大数据集来说太慢。

近似最近邻（ANN）算法用少量精度换取巨大速度提升：

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW（Hierarchical Navigable Small World）是现代向量数据库中的主导算法。它构建多层图，每个节点连接其近似最近邻。搜索从顶层（稀疏、长跳）开始，下降到底层（密集、短跳）。

```figure
norm-unit-balls
```

## 构建它

### 第 1 步：所有范数与距离函数

完整实现请见 `code/distances.py`。每个函数只用基础 Python 数学从零实现。

### 第 2 步：同一数据，不同距离，不同邻居

`distances.py` 中的演示会创建一个数据集，挑选一个查询点，并展示最近邻如何随距离度量变化。在 L1 下「最近」的点，在 L2 或余弦下可能并不是最近。

### 第 3 步：嵌入相似度搜索

代码包含一个模拟嵌入相似度搜索，分别用余弦相似度和 L2 距离为查询找到最相似的「文档」，展示排名可能不同。

## 使用它

最常见的实际用途：在向量数据库中找相似项。

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

当你调用 `model.encode(text)` 然后搜索向量数据库时，底层做的就是这个。嵌入模型把文本映射成向量，向量数据库计算查询向量与每个存储向量之间的余弦相似度（或点积），并用 ANN 算法避免全量比较。

## 练习

1. 计算 (1, 2, 3) 与 (4, 0, 6) 之间的 L1、L2 和 L-无穷距离。验证对任意两点都有 $L_\infty \le L_2 \le L_1$，并证明这个顺序为何一定成立。

2. 构造两个向量，使余弦相似度很高（> 0.9）但 L2 距离很大（> 10）。从几何上解释发生了什么。再构造两个向量，使余弦相似度很低（< 0.3）但 L2 距离很小（< 0.5）。

3. 实现一个函数，接收数据集和查询点，分别返回 L1、L2、余弦和马氏距离下的最近邻。找出一个数据集，使四个度量给出的最近邻都不同。

4. 用 CDF 法手算 [0.5, 0.5, 0, 0] 与 [0, 0, 0.5, 0.5] 之间的 Wasserstein 距离。再算 [0.25, 0.25, 0.25, 0.25] 与 [0, 0, 0.5, 0.5]。哪个更大？为什么？

5. 实现 MinHash 做近似 Jaccard 相似度。生成 100 个随机集合，计算所有配对的精确 Jaccard，并与使用 50、100、200 个哈希函数的 MinHash 近似比较。绘制近似误差。

### 提示

**练习 1。** 差值向量是 (-3, 2, -3)，所以用 `l1_norm`、`l2_norm`、`linf_norm` 计算该向量的三个范数。证明时比较最大的 $\lvert d_i \rvert$ 与平方和，再比较平方和与绝对值和的平方。

**练习 2。** 余弦相似度忽略长度，所以先让两个向量几乎同向，再把其中一个做得长很多。第二种情况，用两个靠近原点但方向不同的短向量，然后用 `cosine_similarity` 和 `l2_distance` 检查两个条件。

**练习 3。** 对每个 `l1_distance`、`l2_distance`、`cosine_distance`、`mahalanobis_distance` 各调用一次 `find_nearest_neighbor(query, dataset, distance_fn)`。`demo_different_neighbors` 已经展示了一个度量不一致的数据集，先读它，再调整点直到四个答案都不同。

**练习 4。** 写出每个分布的累积和，然后加上两个累积和的绝对差。用 `wasserstein_1d(p, q)` 验证两次手算结果，再根据概率质量需要移动多远来解释较大的值。

**练习 5。** 对每个哈希函数，集合的 MinHash 签名是其元素中的最小哈希值，两个集合在多少个哈希函数上得到相同最小值，比例就近似它们的 Jaccard 相似度。把每个估计值与 `jaccard_similarity` 比较；误差应大致随 $1/\sqrt{k}$ 下降，$k$ 为哈希函数数量。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Norm | 「向量的大小」 | 把向量映射到非负标量的函数，满足三角不等式、绝对齐次性，且只有零向量为 0 |
| L1 norm | 「曼哈顿距离」 | 各分量绝对值之和。优化中会产生稀疏性。对异常值稳健 |
| L2 norm | 「欧几里得距离」 | 各分量平方和的开方。欧氏空间中的直线距离 |
| Lp norm | 「广义范数」 | 各分量绝对值的 $p$ 次幂之和的 $p$ 次根。L1 和 L2 是特例 |
| L-infinity norm | 「最大范数」或「切比雪夫距离」 | 最大绝对分量值。$p$ 趋于无穷时 Lp 范数的极限 |
| Cosine similarity | 「向量夹角」 | 用两边大小归一化后的点积。范围 [-1, 1]。忽略向量长度 |
| Cosine distance | 「1 减余弦相似度」 | 把余弦相似度转成距离。范围 [0, 2] |
| Dot product | 「未归一化的余弦」 | 各分量乘积之和。等于余弦相似度乘以两边大小 |
| Mahalanobis distance | 「考虑相关性的距离」 | 在经数据协方差矩阵白化（去相关并归一化）后的空间里计算 L2 距离 |
| Jaccard similarity | 「集合交叠」 | 交集大小除以并集大小。用于集合，而非向量 |
| Edit distance | 「Levenshtein 距离」 | 把一个字符串转成另一个所需的最少插入、删除、替换次数 |
| KL divergence | 「分布之间的距离」 | 不是真正的距离（不对称）。度量用 $Q$ 编码 $P$ 所需的额外比特 |
| Wasserstein distance | 「推土机距离」 | 把质量从一个分布传输到另一个分布所需的最小工作量。真正的度量 |
| Approximate nearest neighbor | 「ANN 搜索」 | HNSW、LSH、IVF 等算法，比精确搜索快得多地找到近似最近点 |
| HNSW | 「向量数据库算法」 | Hierarchical Navigable Small World 图。用于快速近似最近邻搜索的多层图 |
| L1 regularization | 「Lasso」 | 把权重的 L1 范数加到损失中。把权重推向 0（稀疏性） |
| L2 regularization | 「Ridge」或「weight decay」 | 把权重的平方 L2 范数加到损失中。把权重往 0 收缩，但不产生稀疏性 |
| Elastic Net | 「L1 + L2」 | 结合 L1 和 L2 正则化。比单一方法更好地处理相关特征组 |

## 延伸阅读

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) —— Meta 的十亿级 ANN 搜索库
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) —— 把推土机距离引入 GAN 的论文
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) —— 奠基性 ANN 算法
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) —— Word2Vec，余弦相似度由此成为嵌入默认选择
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) —— scikit-learn 中距离度量与邻域算法的实践指南
