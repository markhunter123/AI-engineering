/** zh translations — figures-transformers (9 figures) */
module.exports = {
  'attention-heatmap': '每个 query token 对每个 key token 做点积打分，除以温度，再对行做 softmax 使权重和为一。越深的格子分到的注意力越多。温度调低，焦点锐化到单个 key；温度调高，焦点摊开。',
  'multihead-split': 'multi-head attention 把模型维度切成等份，每头一份，所以头数必须整除 d_model。每头在自己的子空间里注意；切片再拼接回 d_model——加头不花额外的总宽度。',
  'causal-mask': 'causal mask 在 softmax 之前把所有未来位置的分数设成负无穷，于是每个 token 只注意自己和它之前的 token。灰色上三角是被禁的未来。正是这一条约束，让 transformer 可以从左到右生成而不偷看前方。',
  'softmax-attention-scaling': '点积随 head 维度增长，不缩放的话分数变大、softmax 饱和到一个 token 上，梯度被杀死。除以 d_k 的平方根抵消这种增长，注意力分布在任何维度上都保持校准。',
  'word-vector-arithmetic': 'Word2Vec 把词摆成「关系即方向」的样子。man 到 king 的向量与 woman 到 queen 的向量相同，所以从 king 减去 man、加上 woman，几乎正好落在 queen 上。语义成了几何，类比成了向量加法。',
  'bpe-merge': 'Byte-pair encoding 从字符出发，反复把最高频的相邻对合并成新符号。每次合并给词表加一个条目、把语料缩短一点。「es」「est」这类常见序列变成单个 token，高频文本打包进更少的 token，生僻词则还能退回碎片。',
  'gqa-kv-sharing': '每个 query 头保留自己的投影，但多个头可以共享一个 key-value 头。每个 query 一个 kv 是完整 multi-head；全体共用一个是 multi-query；中间按组共享就是 grouped-query attention。kv 头变少，cache 按 query 头与 kv 头之比缩小，质量基本保住。',
  'transformer-residual': 'transformer block 有两个子层：self-attention，再一个 feed-forward network，各自包着 add-and-norm。虚线是残差旁路——把输入 x 原样带过去，在子层之后加回来。几百个 block 能叠起来而梯度不在半路死掉，靠的就是这些旁路。',
  'flash-attention-memory': '标准 attention 要把完整的 N×N 分数矩阵写进内存，开销随序列长度平方增长、很快盖过一切。FlashAttention 分块计算、从不物化那个矩阵，内存随长度线性增长。两条曲线迅速拉开：长上下文时省的是数量级。',
};
