/** zh translations — lesson-figures built-in widgets (16 figures) */
module.exports = {
  'kv-cache-sizer': '缓存为每个 token、每层、每个 kv-head 保存一个 key 和一个 value。它随序列长度和 batch 线性增长——把 GPU 填满的是高 batch 下的长上下文，不是权重。',
  'gradient-descent': '每一步沿梯度乘学习率的方向下山。太小则爬行；太大则越过谷底、发散。训练就是在两者之间寻找那个学习率。',
  'softmax-temperature': '温度在指数之前先除 logits。低于 1 时分布向头部 token 收尖；高于 1 时向均匀摊平。T→0 时是 argmax；T→∞ 时是抛硬币。',
  'bias-variance': '灰色是训练误差，蓝色是测试误差。简单模型错过信号（高偏差）；复杂模型拟合噪声（高方差）。测试误差是两者之和，在两种压力平衡处最低。',
  'l2-regularization': 'L2 把权重范数的平方加进损失。调大 λ 把每个系数往零拉、让模型更平滑。太小则过拟合；太大则连信号一起忘掉。',
  'lr-schedule': '学习率很少一直固定。短暂 warmup 避开早期不稳定；随后 cosine 或 step 衰减把学习率退火到零，让训练后期落进一个好的极小值。',
  'sampling-decoder': '解码按顺序跑三道过滤器。temperature 重塑分布，top-k 封顶候选数量，top-p 保住覆盖概率 p 的最小集合。幸存者重新归一化后从中采样。',
  'scaling-laws': 'Chinchilla 拟合从参数量和 token 数预测损失。在固定算力预算下，损失在每参数约 20 token 附近最低。大多数早期大模型严重训练不足：参数太多，token 太少。',
  'quantization': '每个权重按它的位宽付存储费。位数减半，内存减半、吞吐约翻倍，而损失的精度随之增长。8-bit 几乎免费；4-bit 需要精细方案；再往下精度就守不住了。',
  'rope-explorer': 'RoPE 把每一对维度旋转一个随位置增大的角度。低维用高频（转得快，编码近处顺序）；高维用低频（转得慢，编码长程距离）。调大 base 会拉长每一个波长、扩展可用上下文。',
  'lora-params': 'LoRA 冻结 d×d 的权重，只训练一个低秩增量 B·A——仅 2·d·r 个参数。可训练比例是 2r/d，所以在 4096 维模型上 rank 取 8，训练的权重不足百分之一，质量却大体保住。',
  'precision-recall-threshold': '灰色是负类，蓝色是正类；橙线是阈值。右移它，正类预测变少：precision 升、recall 降。F1 是两者的调和平均，在两条曲线交叉处最高。',
  'cross-entropy-loss': '交叉熵收取模型压在正确答案上的概率的 −log。又对又自信几乎免费；又错又自信倾家荡产。正是这种不对称逼着模型学会校准，而不只是答对。',
  'cosine-similarity': 'embedding 之间比的是角度，不是距离。两个向量同向时 cosine 为 1，正交（无关）为 0，相反为负。模长被约掉——所以长文档和短查询照样能匹配。',
  'tokenizer-tradeoff': '更大的词表把文本切成更少的 token，序列更短、注意力更便宜。但 embedding 表和输出表随词表变大，收益又用参数还回去。真实的 tokenizer 落在两种压力的平衡处，大约 32K 到 128K。',
  'rag-chunking': '小块能精确定位相关段落，却把上下文打碎、让待索引的向量翻倍。大块保住上下文完整，却稀释每一次匹配、灌进 prompt 的 token 爆炸。overlap 软化边界，代价是更多的小块。',
};
