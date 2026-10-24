/** zh translations — figures-llms2 (8 of 9; rlhf-pipeline has no .lf-cap caption) */
module.exports = {
  'rmsnorm-vs-layernorm': '灰色是原始特征向量，蓝色是归一化输出。LayerNorm 先减均值再除标准差，把每个向量重新居中到零。RMSNorm 完全跳过均值，只除均方根——更便宜，还保留任何常数平移。加一个平移，看 LayerNorm 把它吸收掉，而 RMSNorm 放它通过。',
  'swiglu-ffn': '普通 FFN 把输入过一个矩阵加 ReLU：零点处一道硬折角。SwiGLU 从同一输入分成两路——值路 x·W 与门路 swish(x·V)——再相乘。门按坐标平滑地把值放大或缩小，给网络一个柔软、可学习的开关；现代开源模型用它取代了平平无奇的 ReLU。',
  'dpo-loss': 'DPO 跳过单独的奖励模型：直接训练 policy，让 chosen 的得分超过 rejected——两者都相对一个冻结的 reference 度量。损失是 β 倍该差距的 −log σ。正差距（chosen 领先）把损失推向零；负差距产生大梯度。β 控制隐式 KL 约束把 policy 往 reference 拉回的力度。',
  'paged-kv-cache': '连续 KV cache 为每个请求预先保留完整的最大序列长度，大部分位置空着（灰色）。PagedAttention 把缓存放进按需分配的定长页：只有最后一页部分空闲（橙色）。内部浪费从「max − length」降到至多一页——同样的 GPU 上，分页缓存因此装得下多得多的并发序列。',
  'expert-capacity': 'MoE 层里每个 expert 分到固定数量的 token 槽位，由 capacity factor 决定。路由不均，热门 expert 溢出，多出的 token 被丢弃（虚线上方的橙色）。factor 太低，丢很多 token；太高，轻负载的 expert 闲置、padding 算力被浪费。调 factor 就是让两头都小。',
  'sliding-window-attention': '行是 query，列是 key。蓝格是 token 真正注意到的配对；灰格在因果三角形内、却被窗口切掉；白色是未来，永远被掩。全因果注意力填满整个下三角，代价 O(N²)。宽度 w 的滑动窗口只保留带状对角线，降到 O(N·w)——长上下文因此养得起。',
  'differential-attention': '差分注意力计算两份独立的 softmax 映射，把第二份乘以学到的 λ 后从第一份里减去。两份都携带同样的宽泛注意力噪声，减法把它当共模消掉，而真正的信号峰（这里是 token 2）存活。λ 调大，减得更狠——注意力质量因此收拢到相关 token 上，不再摊在无关上下文里。',
  'weight-tying': '输入 embedding 把每个 token id 映射成一个 d 维向量；输出投影把隐藏向量映射回词表上每个词条的 logit。两者都是 vocab×d 矩阵、扮演互逆的角色，所以很多模型把它们绑在一起：输出层复用转置后的 embedding。整整一块 vocab×d_model 的参数被省掉——词表上万 token 时这是笔大钱。',
};
