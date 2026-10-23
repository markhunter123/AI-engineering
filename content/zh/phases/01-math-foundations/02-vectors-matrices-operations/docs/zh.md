# 向量、矩阵与运算

> 每个神经网络本质上都是矩阵乘法多走了几步。

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 minutes

## 学习目标

- 构建一个 Matrix 类，支持逐元素运算、矩阵乘法、转置、行列式和逆矩阵
- 区分逐元素乘法与矩阵乘法，并说明各自的适用场景
- 仅使用从零实现的 Matrix 类实现一个稠密神经网络层（`relu(W @ x + b)`）
- 解释广播规则以及神经网络框架中偏置相加的工作原理

## 问题

你想构建一个神经网络。你看到代码里有这样一行：

```
output = activation(weights @ input + bias)
```

这里的 `@` 是矩阵乘法。`weights` 是矩阵，`input` 是向量。如果你不知道这些运算在做什么，这一行就像魔法；如果你知道，它就是一层网络前向传播的全部，只包含三个运算。

模型处理的每一张图都是像素值构成的矩阵。每个词 embedding 都是向量。每个神经网络的每一层都是矩阵变换。不理解矩阵运算就构建 AI 系统，就像不理解变量就写代码一样不可能。

本节课从零开始培养这种运算熟练度。

## 概念

### 向量：有序的数字列表

向量是具有方向和大小的数字列表。在 AI 中，向量代表数据点、特征或参数。

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

二维向量 `[3, 4]` 指向平面上的坐标 (3, 4)。它的长度（模）是 5（勾股数 3-4-5）。

### 矩阵：数字网格

矩阵是二维网格，有行和列。一个 $m \times n$ 矩阵有 $m$ 行、$n$ 列。

```math
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad \text{2x3 matrix (2 rows, 3 columns)}
```

在神经网络中，权重矩阵把输入向量变换成输出向量。一个有 784 个输入和 128 个输出的层使用 128×784 的权重矩阵。

### 为什么形状很重要

矩阵乘法有严格规则：`(m x n) @ (n x p) = (m x p)`。内部维度必须匹配。

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

如果你在 PyTorch 里遇到形状不匹配错误，原因就在这里。

### 运算速查表

| 运算 | 作用 | 神经网络中的用途 |
|-----------|-------------|-------------------|
| 加法 | 逐元素相加 | 给输出加上偏置 |
| 标量乘法 | 每个元素缩放 | 学习率乘以梯度 |
| 矩阵乘法 | 变换向量 | 层的前向传播 |
| 转置 | 行和列互换 | 反向传播 |
| 行列式 | 单一数值摘要 | 判断可逆性 |
| 逆矩阵 | 撤销变换 | 求解线性系统 |
| 单位矩阵 | 什么都不做的矩阵 | 初始化、残差连接 |

### 逐元素乘法 vs 矩阵乘法

这个区别经常让初学者栽跟头。

逐元素：对应位置相乘。两个矩阵形状必须相同。

```math
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} * \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix}
```

矩阵乘法：行与列的点积。内部维度必须匹配。

```math
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} @ \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
```

不同运算、不同结果、不同规则。

### 广播

当你把偏置向量加到输出矩阵上时，形状并不匹配。广播会把较小的数组拉伸以适配较大数组。

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

每个现代框架都会自动完成。理解它能避免形状看起来不对但代码却能运行的困惑。

```figure
vector-projection
```

## 构建它

### 步骤 1：向量类

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### 步骤 2：带核心运算的 Matrix 类

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### 步骤 3：验证它工作

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### 步骤 4：与神经网络连接

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

这就是单个稠密层：`output = relu(W @ x + b)`。每个神经网络的每个稠密层做的都是这件事。

## 使用它

NumPy 用更少的代码、快几个数量级的速度完成上面所有事。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python 中的 `@` 运算符调用 `__matmul__`。NumPy 用 C 和 Fortran 编写的优化 BLAS 例程来实现它。同样的数学，快 100 倍。

NumPy 中的广播：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy 自动把一维偏置广播到两行。这就是每个神经网络框架中偏置相加的方式。

## 交付它

本节课产出一个通过几何直觉教授矩阵运算的 prompt。详见 `outputs/prompt-matrix-operations.md`。

这里构建的 Matrix 类是我们在第 3 阶段第 10 课构建迷你神经网络框架的基础。

## 练习

1. **验证逆矩阵。** 计算 `A @ A.inverse_2x2()` 并确认得到单位矩阵。用三个不同的 2×2 矩阵尝试。当行列式为 0 时会发生什么？

2. **实现 3×3 逆矩阵。** 使用伴随矩阵法扩展 Matrix 类，计算 3×3 矩阵的逆。用 NumPy 的 `np.linalg.inv` 测试。

3. **构建一个两层网络。** 仅使用你的 Matrix 类（不用 NumPy），创建一个两层神经网络：输入 (3) → 隐藏层 (4) → 输出 (2)。初始化随机权重，运行一次前向传播，并验证所有形状都正确。

### 提示

**练习 1.** 对三个矩阵计算 `A @ A.inverse_2x2()`，将每个元素与单位矩阵在 `1e-9` 内比较。对于行列式为 0 的矩阵，例如 [[1, 2], [2, 4]]，`inverse_2x2` 会抛出 `ValueError`，这是预期结果。

**练习 2.** 计算 3×3 余子式矩阵，转置得到伴随矩阵，然后每个元素除以 `determinant()`。检查 `A @ A_inv` 得到单位矩阵，并且每个元素与 `np.linalg.inv(A)` 在 `1e-9` 内匹配。

**练习 3.** 在相乘之前先写出每个矩阵的形状：`W1` 是 4×3，`W2` 是 2×4，如 `demo_neural_network_layer` 所示。3×1 的输入必须得到 4×1 的隐藏向量和 2×1 的输出。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 向量 | "一个箭头" | 有序的数字列表。在 AI 中：高维空间中的一个点。 |
| 矩阵 | "一张数表" | 线性变换。把向量从一个空间映射到另一个空间。 |
| 矩阵乘法 | "把数字乘起来" | 第一个矩阵的每一行与第二个矩阵的每一列求点积。顺序很重要。 |
| 转置 | "翻过来" | 行和列互换。把 $m \times n$ 矩阵变成 $n \times m$。在反向传播中很关键。 |
| 行列式 | "矩阵里的某个数" | 衡量矩阵把面积（2D）或体积（3D）缩放多少。为 0 意味着变换压扁了一个维度。 |
| 逆矩阵 | "撤销矩阵" | 逆转变换的矩阵。只有当行列式不为 0 时才存在。 |
| 单位矩阵 | "无聊的矩阵" | 相当于乘以 1。用于残差连接（ResNet）。 |
| 广播 | "神奇的形状修复" | 通过沿缺失维度重复，把较小数组拉伸成较大数组的形状。 |
| 逐元素 | "普通乘法" | 对应位置相乘。两个数组形状必须相同（或可广播）。 |

## 延伸阅读

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) —— 本节所有运算的视觉直觉
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) —— NumPy 遵循的精确规则
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) —— 面向 ML 的精简线性代数参考
