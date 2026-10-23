# JAX 入门

> PyTorch 会改变张量。TensorFlow 会构建图。JAX 会编译纯函数。最后这一点会改变你对深度学习的思考方式。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 minutes

## 学习目标

- 使用 JAX 的函数式 API（jax.numpy、jax.grad、jax.jit、jax.vmap）编写纯函数风格的神经网络代码
- 解释 PyTorch 的即时可变模型与 JAX 的函数式编译模型之间的关键设计差异
- 应用 JIT 编译和 vmap 向量化，使训练循环比朴素 Python 更快
- 在 JAX 中训练一个简单网络，并将其显式状态管理与 PyTorch 的面向对象方法进行对比

## 问题

你知道如何在 PyTorch 里构建神经网络。你定义一个 `nn.Module`，调用 `.backward()`，再 step 优化器。它能工作，数百万人都在用。

但 PyTorch 的 DNA 里有一个限制：它用 Python 一次一个地即时追踪操作。每个 `tensor + tensor` 都是一次独立的 kernel 启动。每个训练步骤都会重新解释同样的 Python 代码。这直到你需要在 2048 个 TPU 上训练一个 5400 亿参数的模型之前都没问题。到那时，开销会杀死你。

Google DeepMind 用 JAX 训练 Gemini。Anthropic 用 JAX 训练 Claude。这些都不是小规模任务——它们是地球上最大的神经网络训练运行。他们选择 JAX，是因为 JAX 把训练循环视为可编译的程序，而不是一连串 Python 调用。

JAX 是 NumPy 加上三大超能力：自动微分、JIT 编译到 XLA、自动向量化。你写一个处理单个样本的函数；JAX 给你一个能处理 batch、计算梯度、编译成机器码、跨多个设备运行的函数。而这一切都不需要改变原始函数。

## 概念

### JAX 的哲学

JAX 是一个函数式框架。没有类、没有可变状态、没有 `.backward()` 方法。取而代之的是：

| PyTorch | JAX |
|---------|-----|
| `nn.Module` class with state | Pure function: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | JIT compilation via XLA |
| `for x in batch:` manual loop | `jax.vmap(f)` auto-vectorization |
| `DataParallel` / `FSDP` | `jax.pmap(f)` auto-parallelism |
| Mutable `model.parameters()` | Immutable pytree of arrays |

这不是风格偏好，而是编译器的约束。JIT 编译要求纯函数：相同输入永远产生相同输出，没有副作用。正是这种限制，才带来了百倍加速。

### jax.numpy：熟悉的表面

JAX 在加速器上重新实现了 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

函数名相同、广播规则相同、切片语义相同。但数组住在 GPU/TPU 上，每个操作都可以被编译器追踪。

一个关键区别：JAX 数组是不可变的。不能写 `a[0] = 5`，而要写 `a = a.at[0].set(5)`。一开始会觉得别扭，一周后就会豁然开朗——不可变性正是 `grad`、`jit`、`vmap` 可以任意组合的原因。

### jax.grad：函数式自动微分

PyTorch 把梯度挂在张量上（`.grad`）。JAX 把梯度挂在函数上。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 接受一个函数，返回一个计算梯度的新函数。没有 `.backward()` 调用，没有存在张量上的计算图。梯度只是另一个你可以调用、组合或 JIT 编译的函数。

这种组合可以任意嵌套：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导、三阶导、雅可比矩阵、海森矩阵，全部通过组合 `grad` 得到。PyTorch 也能做这些（`torch.autograd.functional.hessian`），但它是后加的；在 JAX 里，这是根基。

约束是：`grad` 只对纯函数有效。函数内部不能写 print（它们会在追踪时运行，而非执行时）。不能修改外部状态。没有显式 key 管理就不能生成随机数。

### jit：编译到 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

第一次调用时，JAX 会追踪函数：它记录发生了哪些操作，但并不真正执行。然后把这个 trace 交给 XLA（Accelerated Linear Algebra），Google 为 TPU 和 GPU 打造的编译器。XLA 会融合操作、消除冗余内存拷贝，并生成优化的机器码。

后续调用会完全跳过 Python，编译后的代码以 C++ 速度在加速器上运行。

JIT 适合的场景：
- 训练步骤（同样的计算重复成千上万次）
- 推理（同一个模型，不同输入）
- 任何被多次调用、输入形状相似的函数

JIT 不适合的场景：
- 值依赖的 Python 控制流（`if x > 0` 中的 x 是被追踪的数组）
- 一次性计算（编译开销超过运行时间）
- 调试（追踪会隐藏真实执行）

控制流限制是真实存在的。`jax.lax.cond` 替代 `if/else`，`jax.lax.scan` 替代 `for` 循环。它们不是可选项，而是编译的代价。

### vmap：自动向量化

你写一个处理单个样本的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 把它提升为处理 batch：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 的意思是：不要对 `params` 做 batch（共享），对 `x` 的第 0 轴做 batch。不需要手写 `for` 循环，不需要 reshape，不需要手动维护 batch 维度。JAX 会自动找出 batch 维度，并对整个计算做向量化。

这不是语法糖。`vmap` 生成融合的向量化代码，比 Python 循环快 10–100 倍。而且它能与 `jit` 和 `grad` 组合：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

逐样本梯度。一行代码。在 PyTorch 里几乎不可能不借助 hack 做到。

### pmap：跨设备的数据并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 把函数复制到所有可用设备（GPU/TPU），并把 batch 切分。函数内部用 `jax.lax.pmean` 和 `jax.lax.psum` 跨设备同步梯度。

Google 用 `pmap`（以及它的继任者 `shard_map`）在数千张 TPU v5e 芯片上训练 Gemini。编程模型是：写好单设备版本，包一层 `pmap`，完成。

### Pytree：通用数据结构

JAX 操作「pytree」——列表、元组、字典和数组的嵌套组合。你的模型参数就是一个 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

JAX 的每个变换——`grad`、`jit`、`vmap`——都知道如何遍历 pytree。`jax.tree.map(f, tree)` 把 `f` 应用到每个叶子。优化器就是这样一次性更新所有参数的：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有 `.parameters()` 方法，没有参数注册。树结构本身就是模型。

### 函数式 vs 面向对象

PyTorch 把状态存在对象里：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 用显式状态的纯函数：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

参数传进去，不存储任何东西，不改变任何东西。这让每个函数都可测试、可组合、可编译。这也意味着你自己管理参数——或者用 Flax、Equinox 这样的库。

### JAX 生态系统

JAX 给你原语，库给你易用性：

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | Neural network layers | `nn.Module` with explicit state |
| **Equinox** (Patrick Kidger) | Neural network layers | Pytree-based, Pythonic |
| **Optax** (DeepMind) | Optimizers + LR schedules | Composable gradient transforms |
| **Orbax** (Google) | Checkpointing | Save/restore pytrees |
| **CLU** (Google) | Metrics + logging | Training loop utilities |

Optax 是标准优化器库。它把梯度变换（Adam、SGD、裁剪）与参数更新分离，使得组合变得轻而易举：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 何时用 JAX 而非 PyTorch

| Factor | JAX | PyTorch |
|--------|-----|---------|
| TPU support | First-class (Google built both) | Community-maintained (torch_xla) |
| GPU support | Good (CUDA via XLA) | Best-in-class (native CUDA) |
| Debugging | Hard (tracing + compilation) | Easy (eager, line-by-line) |
| Ecosystem | Research-focused (Flax, Equinox) | Massive (HuggingFace, torchvision, etc.) |
| Hiring | Niche (Google/DeepMind/Anthropic) | Mainstream (everywhere) |
| Large-scale training | Superior (XLA, pmap, mesh) | Good (FSDP, DeepSpeed) |
| Prototyping speed | Slower (functional overhead) | Faster (mutate and go) |
| Production inference | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| Who uses it | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

诚实答案：除非有明确理由，否则用 PyTorch。选择 JAX 的理由是——你能用 TPU、需要逐样本梯度、要在超大规模上做多设备训练，或者你在 Google/DeepMind/Anthropic 工作。

### JAX 中的随机数

JAX 没有全局随机状态。每次随机操作都需要显式的 PRNG key：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

一开始很烦人。但它保证了跨设备和跨编译的可复现性——这是 PyTorch 的 `torch.manual_seed` 在多 GPU 设置里无法保证的。

```figure
batchnorm-effect
```

## 构建它

### 第 1 步：设置与数据

我们用 JAX 和 Optax 在 MNIST 上训练一个 3 层 MLP。784 个输入，两个隐藏层分别为 256 和 128 个神经元，10 个输出类。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 第 2 步：初始化参数

没有类，只有一个返回 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

手动完成 He 初始化。三个 PRNG key 从一个种子中分裂出来。每个权重都是嵌套字典里的不可变数组。

### 第 3 步：前向传播

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

纯函数。参数进，预测出。没有 `self`，没有存储状态。`loss_fn` 从零实现了交叉熵——softmax、log、负均值。

### 第 4 步：JIT 编译的训练步骤

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 一次返回损失值和梯度。`@jax.jit` 装饰器把两个函数都编译到 XLA。第一次调用之后，每个训练步骤都不会再触碰 Python。

### 第 5 步：训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 个 epoch，测试准确率约 97%。第一个 epoch 较慢（JIT 编译），第 2–10 个 epoch 很快。

注意少了什么：没有 `.zero_grad()`，没有 `.backward()`，没有 `.step()`。整个更新是一个组合函数调用。梯度被计算、被 Adam 变换、被应用到参数上——全部发生在 `train_step` 内部。

## 使用它

### Flax：Google 的标准库

Flax 是最常见的 JAX 神经网络库。它把 `nn.Module` 带了回来，但采用显式状态管理：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

结构与 PyTorch 相同，但 `params` 与模型是分离的。`model.init()` 创建参数，`model.apply(params, x)` 运行前向传播。模型对象本身没有状态。

### Equinox：更 Pythonic 的选择

Equinox（Patrick Kidger 出品）把模型本身表示成 pytree：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

模型本身就是 pytree。不需要 `.apply()`。参数就是模型的叶子节点。这更接近 JAX 的思维方式。

### Optax：可组合的优化器

Optax 把梯度变换与更新解耦：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

梯度裁剪、学习率 warmup、权重衰减，全部以变换链的形式组合。每个变换看到梯度、修改它、再传给下一个。没有单一的优化器类。

## 交付它

**安装：**

```bash
pip install jax jaxlib optax flax
```

如需 GPU 支持：

```bash
pip install jax[cuda12]
```

如需 TPU（Google Cloud）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**性能注意事项：**

- 第一次 JIT 调用较慢（编译开销）。基准测试前先做 warm up。
- 避免在 JIT 内部对 JAX 数组做 Python 循环。用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 可以在 JIT 内部工作，普通 `print()` 不行。
- 用 `jax.profiler` 或 TensorBoard 做性能分析。XLA 编译可能隐藏瓶颈。
- JAX 默认预分配 75% 的 GPU 内存。设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 可关闭。

**Checkpointing：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课产物：**
- `outputs/prompt-jax-optimizer.md` —— 一个用于选择合适 JAX 优化器配置的 prompt
- `outputs/skill-jax-patterns.md` —— 一份涵盖 JAX 函数式模式的技能参考

## 练习

1. 给 MLP 加上 dropout。在 JAX 里，dropout 需要 PRNG key——把 key 传进前向传播，并在每个 dropout 层 split 出新的 key。对比加与不加 dropout 的测试准确率。

2. 用 `jax.vmap` 计算 32 张 MNIST 图片的逐样本梯度。计算每张图片的梯度范数。哪些样本梯度最大，为什么？

3. 把手动前向函数换成通用的 `mlp_forward(params, x)`，使其适用于任意层数。用 `jax.tree.leaves` 自动判断深度。

4. 对训练步骤做基准测试，比较有 `@jax.jit` 和没有的情况。各跑 100 步。你的硬件上加速比是多少？第一次调用的编译开销有多大？

5. 通过组合 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 实现梯度裁剪。对比加与不加裁剪的训练效果。绘制训练过程中的梯度范数，观察影响。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| XLA | "The thing that makes JAX fast" | Accelerated Linear Algebra -- a compiler that fuses operations and generates optimized GPU/TPU kernels from a computation graph |
| JIT | "Just-in-time compilation" | JAX traces the function on first call, compiles to XLA, then runs the compiled version on subsequent calls |
| Pure function | "No side effects" | A function where the output depends only on inputs -- no global state, no mutation, no randomness without explicit keys |
| vmap | "Auto-batching" | Transforms a function that processes one example into one that processes a batch, without rewriting |
| pmap | "Auto-parallelism" | Replicates a function across multiple devices and splits the input batch |
| Pytree | "Nested dict of arrays" | Any nested structure of lists, tuples, dicts, and arrays that JAX can traverse and transform |
| Tracing | "Recording the computation" | JAX executes the function with abstract values to build a computation graph, without computing real results |
| Functional autodiff | "grad of a function" | Computing derivatives by transforming functions, not by attaching gradient storage to tensors |
| Optax | "JAX's optimizer library" | A composable library of gradient transformations -- Adam, SGD, clipping, scheduling -- that chain together |
| Flax | "JAX's nn.Module" | Google's neural network library for JAX, adding layer abstractions while keeping state explicit |

## 延伸阅读

- JAX 官方文档：https://jax.readthedocs.io/ —— 关于 grad、jit、vmap 的优秀教程
- "JAX: composable transformations of Python+NumPy programs"（Bradbury et al., 2018）—— 解释设计哲学的原始论文
- Flax 文档：https://flax.readthedocs.io/ —— Google 的 JAX 神经网络库
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations"（2021）—— Flax 的 Pythonic 替代品
- DeepMind, "Optax: composable gradient transformation and optimisation" —— 标准优化器库
- "You Don't Know JAX"（Colin Raffel, 2020）—— 来自 T5 作者之一的 JAX 陷阱与模式实用指南
