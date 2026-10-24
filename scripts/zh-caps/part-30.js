/** zh translations — figures-dl (9 figures) */
module.exports = {
  'perceptron-boundary': 'perceptron 用 w·x + b 的符号做预测，所以决策面是一条直线。拖动权重和偏置，旋转并平移这条线，直到每个蓝色圆都在正侧、每个灰色方块都在负侧。',
  'mlp-forward': '两个输入穿过固定权重进入三个隐藏单元，各自被 tanh 压缩，再合成一个输出。蓝色边是正权重，金色是负权重；节点底色显示每个单元对你设定的输入反应多强。',
  'backprop-vanishing': 'backprop 每层乘一个 activation 导数。Sigmoid 的导数封顶 0.25，tanh 也到不了一，于是在深网络里连乘积塌向零（注意对数轴）。ReLU 对活跃单元保持导数为一——深度训练因此才变得可行。',
  'optimizer-trajectory': '损失面是一道窄峡谷：沿 x 平缓，沿 y 陡二十倍。朴素 SGD 在陡壁间来回弹跳、沿谷底缓慢爬行。Momentum 把弹跳抹平；Adam 对每个方向归一化，两个轴以相近的速度收敛。',
  'weight-init-variance': '每层把 activation 方差乘上 n·Var(w) 的增益。天真的大权重让增益随宽度增长，activation 爆炸（对数轴）。Xavier 取 Var(w)=1/n，He 对 ReLU 取 2/n，都把增益按在 1 附近，信号幅度于是不随深度衰减。',
  'dropout-mask': 'Dropout 每步把比例为 p 的单元清零，网络因此无法依赖任何单个单元。因为只有幸存者传递信号，它们被放大 1/(1−p) 以保持期望 activation 不变；测试时整层全开、不再缩放。',
  'batchnorm-effect': '不管下一层送上来怎样的均值和宽度（灰色），batch norm 减去 batch 均值、除以 batch 标准差，把分布掰回零均值单位方差（蓝色）。可学习的 γ 和 β 随后让网络按需重新拉伸——如果别的尺度更有用。',
  'learning-curves': '灰色是训练损失，蓝色是验证损失。容量越大训练损失越低，但验证损失触底后随模型开始背噪声而回升。不断张开的缺口就是过拟合信号；金线标记 early stopping 会冻结模型的位置。',
  'gradient-clipping': 'gradient 范数突然飙升时，一步就能把权重扔下悬崖。clipping 把任何范数超过阈值 τ 的 gradient 等比缩回 τ：方向不变，幅度封顶。τ 以下原样通过；τ 以上更新被驯服为 min(‖g‖, τ)。',
};
