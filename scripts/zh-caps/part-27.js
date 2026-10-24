/** zh translations — figures-genai-rl (9 figures) */
module.exports = {
  'diffusion-denoise': 'diffusion 模型学的是逆转加噪过程。t = T 时信号是纯噪声；t 一路降到 0，模型逐步去噪，底下的信号（灰色虚线）重新浮现。每个 x_t 都是干净信号与同一份噪声的固定混合，权重由 schedule 决定。',
  'noise-schedule': 'alpha_bar 是第 t 步还存活的信号比例，它与剩余噪声之比就是信噪比。linear schedule 在前几步就把信号抹掉大半；cosine schedule 中段衰减得更平缓，让有用信号留得更久，给模型更有信息量的中间目标。',
  'vae-latent-grid': 'VAE 把输入映进一个平滑的 latent 空间，再把空间中的点解码回输出。左边的方块是这个空间的一片切片，圆点是你的 latent code。挪动它，右边的解码形状随之连续形变——因为 decoder 训练成相邻的编码产出相邻的输出。',
  'gan-minimax': 'GAN 是两人博弈：generator（蓝）设法骗过 discriminator（灰），discriminator 设法分辨真假。两者必须一起变强。discriminator 大获全胜，它给 generator 的梯度就消失；generator 一骑绝尘，又可能坍缩到少数几个输出上。健康的训练停在虚线标出的均衡附近。',
  'qlearning-gridworld': 'Agent 学会抵达目标、避开深坑。格子底色是学到的 state value（蓝好琥珀坏），箭头是 greedy policy。早期数值都贴近零；随着训练，价值从目标向外传播，箭头排成一条绕开深坑的路径。',
  'value-iteration-gamma': 'value iteration 把 reward 从目标一格一格向回推。这条链上唯一的 reward 在最右格，所以每个状态的价值就是 gamma 的「到目标距离」次方。gamma 太小，远处的 reward 几乎一文不值（短视）；gamma 接近 1，价值能沿链条一路传回。',
  'epsilon-greedy': 'Agent 以 epsilon 概率随机探索一个动作，否则按当前最优估计利用。长条显示 explore/exploit 的配比，曲线是累计 regret——没有永远选最优臂而放弃的那部分 reward。固定 epsilon 的 regret 永远堆积；衰减 schedule 先探索后利用，regret 走平。',
  'discount-horizon': 't 步之后的 reward 按 gamma^t 加权，权重几何式衰减（长条）。虚线是有效视野 1/(1 − gamma)，权重在那里降到约 1/e。调大 gamma，视野拉长，Agent 开始在乎遥远得多的未来 reward。',
  'policy-gradient-landscape': 'policy gradient 方法沿着能抬高期望 reward 的方向调整策略参数 theta。灰色曲线是 reward 地形，圆点是一步步爬向峰顶的上升轨迹。步长太小爬得慢；太大就冲过头；左边的小凸起若是起点，整段攀登会被困在那里。',
};
