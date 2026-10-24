/** zh translations — figures-foundations2 (8 figures) */
module.exports = {
  'data-augmentation': '增广对每张训练图像施加保持标签的变换——翻转、旋转、裁剪、色彩偏移——一个有标签示例变成许多个。网络从不会两次见到完全相同的输入，学会的特征能在这些变化下存活——等效数据集因此扩大，过拟合被抑制，还不用多收集一条数据。',
  'transfer-learning': '预训练 backbone 已经认识通用特征——边缘、纹理、形状。迁移学习冻结这些低层，只训练顶部几层加一个新 head 来适配目标任务。可训练参数更少，意味着要存的梯度更少、要拟合的数据远更少、收敛更快；数据集很小时多冻几层，数据集又大又不同时少冻几层。',
  'batchnorm-inference': '训练时 BatchNorm 用当前 mini-batch 的均值和方差（橙色）归一化每个 activation，同时悄悄累计滑动平均。推理时切换到冻结的滑动平均（蓝色），单个输入也按确定性处理。小 batch 会让 per-batch 统计量噪声很大——这正是极小 batch 伤害 BatchNorm、并催生 Group Norm 或 Layer Norm 的原因。',
  'ctc-collapse': 'CTC 让声学模型对每帧发一个标签，而不需要知道对齐。解码按顺序跑两步：先把连续相同符号折叠成一个，再删掉 blank token。blank 至关重要——它隔开两个真实的「l」帧，「hello」的两个 l 才不会被合并成一个。',
  'mfcc-pipeline': 'MFCC 跑一条固定 pipeline：取 spectrogram，把它弯折到 mel 间隔的滤波器组上，取对数模拟响度感知，再做 DCT。DCT 把平滑的谱包络压进前几个系数——只保留最低的 12–13 个，就抓住了区分音素的声道形状，同时把音高和噪声留在更高系数里扔掉。',
  'autoencoder-bottleneck': '自编码器把输入挤过一个狭窄瓶颈，在另一侧重建。宽瓶颈什么都拷贝、什么都没学；极窄的瓶颈逼网络丢弃细节、只留主导结构，重建误差随压缩变强而上升。甜点位是留住信号、扔掉噪声——那段学到的编码就是有用的表示。',
  'normalizing-flow': 'normalizing flow 把一个简单的基础密度（灰色高斯）经可逆函数映射成复杂目标（蓝色）。因为映射可逆，换元公式给出精确密度：除以雅可比行列式的绝对值——对数域就是减去 log|det J|。映射拉伸空间处密度变稀，压缩处密度堆积——因为一切都是精确的，flow 可以用最大似然训练。',
  'score-matching': 'score-based 模型学的是 score——log 密度的梯度——画出来就是指向数据稠密处的灰色向量场。生成不需要显式密度：从噪声出发，沿 score 反复走步（Langevin 动力学，每步加一点噪声）。橙色样本跟着箭头走出空旷区，落进一个高密度模式——diffusion 模型采样正是这么画的。',
};
