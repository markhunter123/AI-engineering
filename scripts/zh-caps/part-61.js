/** zh translations — figures-vision-speech (8 figures) */
module.exports = {
  'convolution-kernel': '卷积把一个小 kernel 盖在图像每个 3x3 patch 上、求逐元素乘积的和。边缘 kernel 抵消平坦区、点亮竖直条带；blur 对邻居取平均；sharpen 让中心相对周围放大。输出每边小两个像素——因为窗口无法在边界上居中。',
  'pooling': 'pooling 把每个窗口概括成一个值来下采样特征图。2x2 窗口、步长 2，两个维度各减半，4x4 网格变成 2x2。max pooling 转发每个窗口里最强的 activation（高亮）；average pooling 转发均值。结果更小、对平移更宽容、对下一层更便宜。',
  'receptive-field': 'CNN 深处的神经元看不见整幅图，只看得见透过下面每一层喂给它的 patch。每个 kernel 把那个 patch 加宽 (kernel − 1)，任何 stride 把后续所有层的触达乘大。叠小 kernel 是网络廉价攒出大感受野的方式——不需要一个巨型滤波器。',
  'conv-output-size': '卷积的输出空间尺寸服从一个公式：floor((W − K + 2P) / S) + 1。padding P 加边，让 kernel 够得着边缘、保住尺寸；stride S 跳位、缩小输出。「same」padding 选 P 使输出匹配输入；「valid」padding 用 P = 0、每边缩 K − 1。',
  'cnn-param-count': '一个卷积层花 (K·K·Cin + 1)·Cout 个权重，而且那一个 kernel 在每个空间位置复用。稠密层接同样的特征图，每对输入输出像素都要一个独立权重——随图像尺寸爆炸。权重共享正是 CNN 小到能在图像上训练的原因。',
  'spectrogram-window': 'spectrogram 把信号切成窗、对每窗做 Fourier 变换。短窗钉死「什么时候发生」，能量却摊到频率上；长窗把音高解得细，却沿时间抹开。两种分辨率的乘积固定——每个选择都是在拿一个换另一个。',
  'mel-scale': '人耳对低频解得细、对高频解得粗。mel 尺度把线性 Hz 拗弯来匹配：1 kHz 以下近乎线性，以上急剧压缩——从 8 到 9 kHz 跨过的 mel 远少于 200 到 300 Hz。mel 滤波器组把分辨率花在耳朵真正在乎的地方。',
  'nyquist-aliasing': '灰色是真实信号，蓝点是采样点，虚线是这些点暗示的波。只要采样率超过信号频率的两倍，采样点就能重建原波。把频率推过 Nyquist（fs/2），同样的点描出一条更低的假频率——这就是混叠，也是音频采样前先低通滤波的原因。',
};
