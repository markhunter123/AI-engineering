# 无监督学习

> 没有标签，没有老师。算法自己发现结构。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1 (Norms & Distances, Probability & Distributions), Phase 2 Lessons 1-6
**Time:** ~90 minutes

## 学习目标

- 从头实现 K-Means、DBSCAN 和高斯混合模型，并比较它们的聚类行为
- 使用轮廓分数（silhouette score）和肘部法选择最优 K，评估聚类质量
- 解释 DBSCAN 何时优于 K-Means，并判断哪种算法能处理非球形簇和异常值
- 用聚类方法构建异常检测流水线，标记偏离正常模式的点

## 问题

到目前为止，每节 ML 课都假设数据有标签：“这是输入，这是正确输出。” 但在现实世界中，标签很昂贵。医院有数百万份病历，但没人手动给每个病例打上疾病分类标签；电商网站有数百万次用户会话，但没人手动标注客户细分；安全团队有网络日志，但没人标记每一次异常。

无监督学习无需被告知该找什么，就能发现模式。它将相似的数据点分组，挖掘隐藏结构，并暴露异常值。如果说监督学习是拿着带答案的教科书学习，那么无监督学习就是盯着原始数据，直到模式自己浮现。

问题在于：没有标签，你就无法直接衡量“对”或“错”。你需要不同的工具来判断算法发现的结构是否有意义。

## 概念

### 聚类：把相似的东西归到一起

聚类将每个数据点分配到一个组（簇），使同一组内的点彼此比与其他组的点更相似。问题始终是：“相似”意味着什么？

```mermaid
flowchart LR
    A[Raw Data] --> B{Choose Method}
    B --> C[K-Means]
    B --> D[DBSCAN]
    B --> E[Hierarchical]
    B --> F[GMM]
    C --> G[Flat, spherical clusters]
    D --> H[Arbitrary shapes, noise detection]
    E --> I[Tree of nested clusters]
    F --> J[Soft assignments, elliptical clusters]
```

### K-Means：主力军

K-Means 将数据划分为恰好 K 个簇。每个簇有一个质心（簇的质量中心），每个点属于最近的质心。

Lloyd 算法：

1. 随机选 K 个点作为初始质心
2. 将每个数据点分配到最近的质心
3. 将每个质心重新计算为其分配点的均值
4. 重复步骤 2-3，直到分配不再变化

目标函数（inertia，惯性）衡量每个点到其质心的平方距离之和。K-Means 最小化它，但只能找到局部最小值。不同的初始化可能产生不同结果。

### 选择 K

两种标准方法：

**肘部法：** 对 $K = 1, 2, 3, \ldots, n$ 分别运行 K-Means，绘制 inertia 随 K 变化的曲线，寻找“肘部”——即增加簇数后 inertia 不再显著下降的位置。

**轮廓分数：** 对每个点，衡量它与所属簇的相似度（$a$）与离它最近的其他簇的相似度（$b$）。轮廓系数为 $(b - a) / \max(a, b)$，取值从 -1（分错簇）到 +1（聚类良好）。对所有点取平均得到全局分数。

### DBSCAN：基于密度的聚类

K-Means 假设簇是球形的，并且需要预先指定 K。DBSCAN 不做这两个假设。它将簇视为被稀疏区域隔开的密集区域。

两个参数：
- **eps**：邻域半径
- **min_samples**：形成密集区域所需的最少点数

三类点：
- **核心点**：在 eps 距离内至少有 min_samples 个点
- **边界点**：在核心点的 eps 范围内，但自身不是核心点
- **噪声点**：既不是核心点也不是边界点，即异常值

DBSCAN 将彼此在 eps 范围内的核心点连接成同一个簇，边界点加入附近核心点所在的簇，噪声点不属于任何簇。

优点：能发现任意形状的簇、自动确定簇数量、识别异常值。缺点：在簇密度差异大时表现不佳。

### 层次聚类

层次聚类构建一棵嵌套簇的树（树状图，dendrogram）。

凝聚式（自底向上）：
1. 开始时每个点自成一个簇
2. 合并最近的两个簇
3. 重复直到只剩一个簇
4. 在所需层级切断树状图，得到 K 个簇

簇之间的“距离”可以定义为：
- **单链接（single linkage）**：两个簇中任意两点间的最小距离
- **全链接（complete linkage）**：任意两点间的最大距离
- **平均链接（average linkage）**：所有点对之间的平均距离
- **Ward 法**：使合并后簇内总方差增加最小的合并方式

### 高斯混合模型（GMM）

K-Means 给出硬分配：每个点恰好属于一个簇。GMM 给出软分配：每个点属于每个簇都有一定的概率。

GMM 假设数据由 K 个高斯分布混合生成，每个高斯有自己的均值和协方差。期望最大化（EM）算法在以下两步间交替：

- **E 步**：计算每个点属于每个高斯的概率
- **M 步**：更新每个高斯的均值、协方差和混合权重，以最大化数据的似然

GMM 能建模椭圆形簇（不像 K-Means 只限球形），并能自然处理重叠簇。

### 该用哪种

| 方法 | 最适合 | 避免使用的情况 |
|--------|----------|------------|
| K-Means | 大数据集、球形簇、已知 K | 不规则形状、存在异常值 |
| DBSCAN | 未知 K、任意形状、异常检测 | 密度差异大、维度很高 |
| 层次聚类 | 小数据集、需要树状图、未知 K | 大数据集（$O(n^2)$ 内存） |
| GMM | 重叠簇、需要软分配 | 超大数据集、维度过高 |

### 用聚类做异常检测

聚类天然支持异常检测：
- **K-Means**：离任何质心都很远的点是异常
- **DBSCAN**：噪声点按定义就是异常
- **GMM**：在所有高斯分布下概率都很低的点是异常

```figure
kmeans-step
```

## 构建它

### 第 1 步：从头实现 K-Means

```python
import math
import random


def euclidean_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def kmeans(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n_features = len(data[0])

    centroids = random.sample(data, k)

    for iteration in range(max_iterations):
        clusters = [[] for _ in range(k)]
        assignments = []

        for point in data:
            distances = [euclidean_distance(point, c) for c in centroids]
            nearest = distances.index(min(distances))
            clusters[nearest].append(point)
            assignments.append(nearest)

        new_centroids = []
        for cluster in clusters:
            if len(cluster) == 0:
                new_centroids.append(random.choice(data))
                continue
            centroid = [
                sum(point[j] for point in cluster) / len(cluster)
                for j in range(n_features)
            ]
            new_centroids.append(centroid)

        if all(
            euclidean_distance(old, new) < 1e-6
            for old, new in zip(centroids, new_centroids)
        ):
            print(f"  Converged at iteration {iteration + 1}")
            break

        centroids = new_centroids

    return assignments, centroids
```

### 第 2 步：肘部法与轮廓分数

```python
def compute_inertia(data, assignments, centroids):
    total = 0.0
    for point, cluster_id in zip(data, assignments):
        total += euclidean_distance(point, centroids[cluster_id]) ** 2
    return total


def silhouette_score(data, assignments):
    n = len(data)
    if n < 2:
        return 0.0

    clusters = {}
    for i, c in enumerate(assignments):
        clusters.setdefault(c, []).append(i)

    if len(clusters) < 2:
        return 0.0

    scores = []
    for i in range(n):
        own_cluster = assignments[i]
        own_members = [j for j in clusters[own_cluster] if j != i]

        if len(own_members) == 0:
            scores.append(0.0)
            continue

        a = sum(euclidean_distance(data[i], data[j]) for j in own_members) / len(own_members)

        b = float("inf")
        for cluster_id, members in clusters.items():
            if cluster_id == own_cluster:
                continue
            avg_dist = sum(euclidean_distance(data[i], data[j]) for j in members) / len(members)
            b = min(b, avg_dist)

        if max(a, b) == 0:
            scores.append(0.0)
        else:
            scores.append((b - a) / max(a, b))

    return sum(scores) / len(scores)


def find_best_k(data, max_k=10):
    print("Elbow method:")
    inertias = []
    for k in range(1, max_k + 1):
        assignments, centroids = kmeans(data, k)
        inertia = compute_inertia(data, assignments, centroids)
        inertias.append(inertia)
        print(f"  K={k}: inertia={inertia:.2f}")

    print("\nSilhouette scores:")
    for k in range(2, max_k + 1):
        assignments, centroids = kmeans(data, k)
        score = silhouette_score(data, assignments)
        print(f"  K={k}: silhouette={score:.4f}")

    return inertias
```

### 第 3 步：从头实现 DBSCAN

```python
def dbscan(data, eps, min_samples):
    n = len(data)
    labels = [-1] * n
    cluster_id = 0

    def region_query(point_idx):
        neighbors = []
        for i in range(n):
            if euclidean_distance(data[point_idx], data[i]) <= eps:
                neighbors.append(i)
        return neighbors

    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = region_query(i)

        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seed_set = list(neighbors)
        seed_set.remove(i)

        j = 0
        while j < len(seed_set):
            q = seed_set[j]

            if not visited[q]:
                visited[q] = True
                q_neighbors = region_query(q)
                if len(q_neighbors) >= min_samples:
                    for nb in q_neighbors:
                        if nb not in seed_set:
                            seed_set.append(nb)

            if labels[q] == -1:
                labels[q] = cluster_id

            j += 1

        cluster_id += 1

    return labels
```

### 第 4 步：高斯混合模型（EM 算法）

```python
def gmm(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n = len(data)
    d = len(data[0])

    indices = random.sample(range(n), k)
    means = [list(data[i]) for i in indices]
    variances = [1.0] * k
    weights = [1.0 / k] * k

    def gaussian_pdf(x, mean, variance):
        d = len(x)
        coeff = 1.0 / ((2 * math.pi * variance) ** (d / 2))
        exponent = -sum((xi - mi) ** 2 for xi, mi in zip(x, mean)) / (2 * variance)
        return coeff * math.exp(max(exponent, -500))

    for iteration in range(max_iterations):
        responsibilities = []
        for i in range(n):
            probs = []
            for j in range(k):
                probs.append(weights[j] * gaussian_pdf(data[i], means[j], variances[j]))
            total = sum(probs)
            if total == 0:
                total = 1e-300
            responsibilities.append([p / total for p in probs])

        old_means = [list(m) for m in means]

        for j in range(k):
            r_sum = sum(responsibilities[i][j] for i in range(n))
            if r_sum < 1e-10:
                continue

            weights[j] = r_sum / n

            for dim in range(d):
                means[j][dim] = sum(
                    responsibilities[i][j] * data[i][dim] for i in range(n)
                ) / r_sum

            variances[j] = sum(
                responsibilities[i][j]
                * sum((data[i][dim] - means[j][dim]) ** 2 for dim in range(d))
                for i in range(n)
            ) / (r_sum * d)
            variances[j] = max(variances[j], 1e-6)

        shift = sum(
            euclidean_distance(old_means[j], means[j]) for j in range(k)
        )
        if shift < 1e-6:
            print(f"  GMM converged at iteration {iteration + 1}")
            break

    assignments = []
    for i in range(n):
        assignments.append(responsibilities[i].index(max(responsibilities[i])))

    return assignments, means, weights, responsibilities
```

### 第 5 步：生成测试数据并运行所有算法

```python
def make_blobs(centers, n_per_cluster=50, spread=0.5, seed=42):
    random.seed(seed)
    data = []
    true_labels = []
    for label, (cx, cy) in enumerate(centers):
        for _ in range(n_per_cluster):
            x = cx + random.gauss(0, spread)
            y = cy + random.gauss(0, spread)
            data.append([x, y])
            true_labels.append(label)
    return data, true_labels


def make_moons(n_samples=200, noise=0.1, seed=42):
    random.seed(seed)
    data = []
    labels = []
    n_half = n_samples // 2
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = math.cos(angle) + random.gauss(0, noise)
        y = math.sin(angle) + random.gauss(0, noise)
        data.append([x, y])
        labels.append(0)
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = 1 - math.cos(angle) + random.gauss(0, noise)
        y = 1 - math.sin(angle) - 0.5 + random.gauss(0, noise)
        data.append([x, y])
        labels.append(1)
    return data, labels


if __name__ == "__main__":
    centers = [[2, 2], [8, 3], [5, 8]]
    data, true_labels = make_blobs(centers, n_per_cluster=50, spread=0.8)

    print("=== K-Means on 3 blobs ===")
    assignments, centroids = kmeans(data, k=3)
    print(f"  Centroids: {[[round(c, 2) for c in cent] for cent in centroids]}")
    sil = silhouette_score(data, assignments)
    print(f"  Silhouette score: {sil:.4f}")

    print("\n=== Elbow Method ===")
    find_best_k(data, max_k=6)

    print("\n=== DBSCAN on 3 blobs ===")
    db_labels = dbscan(data, eps=1.5, min_samples=5)
    n_clusters = len(set(db_labels) - {-1})
    n_noise = db_labels.count(-1)
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")

    print("\n=== GMM on 3 blobs ===")
    gmm_assignments, gmm_means, gmm_weights, _ = gmm(data, k=3)
    print(f"  Means: {[[round(m, 2) for m in mean] for mean in gmm_means]}")
    print(f"  Weights: {[round(w, 3) for w in gmm_weights]}")
    gmm_sil = silhouette_score(data, gmm_assignments)
    print(f"  Silhouette score: {gmm_sil:.4f}")

    print("\n=== DBSCAN on moons (non-spherical clusters) ===")
    moon_data, moon_labels = make_moons(n_samples=200, noise=0.1)
    moon_db = dbscan(moon_data, eps=0.3, min_samples=5)
    n_moon_clusters = len(set(moon_db) - {-1})
    n_moon_noise = moon_db.count(-1)
    print(f"  Found {n_moon_clusters} clusters, {n_moon_noise} noise points")

    print("\n=== K-Means on moons (will fail to separate) ===")
    moon_km, moon_centroids = kmeans(moon_data, k=2)
    moon_sil = silhouette_score(moon_data, moon_km)
    print(f"  Silhouette score: {moon_sil:.4f}")
    print("  K-Means splits moons poorly because they are not spherical")

    print("\n=== Anomaly detection with DBSCAN ===")
    anomaly_data = list(data)
    anomaly_data.append([20.0, 20.0])
    anomaly_data.append([-5.0, -5.0])
    anomaly_data.append([15.0, 0.0])
    anomaly_labels = dbscan(anomaly_data, eps=1.5, min_samples=5)
    anomalies = [
        anomaly_data[i]
        for i in range(len(anomaly_labels))
        if anomaly_labels[i] == -1
    ]
    print(f"  Detected {len(anomalies)} anomalies")
    for a in anomalies[-3:]:
        print(f"    Point {[round(v, 2) for v in a]}")
```

## 使用它

在 scikit-learn 中，这些算法只需一行代码：

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score as sklearn_silhouette

km = KMeans(n_clusters=3, random_state=42).fit(data)
db = DBSCAN(eps=1.5, min_samples=5).fit(data)
agg = AgglomerativeClustering(n_clusters=3).fit(data)
gmm_model = GaussianMixture(n_components=3, random_state=42).fit(data)
```

从零实现的版本让你清楚这些库到底在算什么：K-Means 在分配与重新计算之间迭代，DBSCAN 从密集种子扩展簇，GMM 在期望步与最大化步之间交替。库版本增加了数值稳定性、更聪明的初始化（K-Means++）和 GPU 加速，但核心逻辑相同。

## 交付它

本节课产出了 K-Means、DBSCAN 和 GMM 的可用从零实现。这些聚类代码可以作为更高级无监督方法的基础。

## 练习

1. 实现 K-Means++ 初始化：先随机选第一个质心，之后每个质心以到最近已有质心平方距离成正比的概率被选中。比较其与随机初始化的收敛速度。
2. 在代码中加入层次凝聚聚类。实现 Ward 链接并生成树状图（用嵌套合并列表表示）。在不同层级切断，并与 K-Means 结果比较。
3. 构建一个简单的异常检测流水线：在同一数据上运行 DBSCAN 和 GMM，将两种方法都判定为异常的点标记为异常（DBSCAN 中的噪声、GMM 中的低概率点）。测量两者的重叠率，并讨论它们何时会不一致。

## 关键术语

| 术语 | 俗称 | 实际含义 |
|------|----------------|----------------------|
| 聚类 | “把相似的东西分组” | 将数据划分为若干子集，使组内相似度高于组间相似度，具体由某种距离度量定义 |
| 质心 | “簇的中心” | 分配给某簇的所有点的均值；K-Means 用它作为簇的代表 |
| 惯性 | “簇有多紧” | 每个点到其质心的平方距离之和；越小越紧凑 |
| 轮廓分数 | “簇分离得有多好” | 对每个点计算 $(b - a) / \max(a, b)$，其中 $a$ 是簇内平均距离，$b$ 是最近其他簇的平均距离 |
| 核心点 | “密集区域的点” | DBSCAN 中在 eps 距离内至少有 min_samples 个邻居的点 |
| EM 算法 | “软 K-Means” | 期望最大化：迭代计算成员概率（E 步）并更新分布参数（M 步） |
| 树状图 | “簇的树” | 展示层次聚类中簇合并顺序与距离的树形图 |
| 异常 | “离群点” | 不符合预期模式的数据点，DBSCAN 将其标为噪声，GMM 赋予其低概率 |

## 延伸阅读

- [Stanford CS229 - Unsupervised Learning](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) —— Andrew Ng 关于聚类与 EM 的讲义
- [scikit-learn 聚类指南](https://scikit-learn.org/stable/modules/clustering.html) —— 带可视化示例的各种聚类算法实用对比
- [DBSCAN 原始论文（Ester et al., 1996）](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) —— 引入基于密度聚类的论文
