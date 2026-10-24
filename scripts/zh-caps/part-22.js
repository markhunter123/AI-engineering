/** zh translations — figures-capstone-c (8 figures) */
module.exports = {
  'cc-embedding-lookup': 'token id 不是算术对象，是索引。id 从 token 表里选出一行，返回一个稠密向量——模型把它当作这个 id 的含义。位置也没有天生的向量，所以一个并行的 positional embedding 按槽位查出、逐元素相加。得到的就是第一个注意力 block 消费的 (B, T, D) 张量。',
  'cc-transformer-block': 'block 恰好有两个子层、两条残差路径。输入分叉：一份穿过 LayerNorm 再进注意力，另一份直通加法。同样的分叉在 MLP 周围重复。干净的旁路让梯度能触到深堆栈的底部；把 norm 放进旁路之内（pre-LN），堆栈才能不靠 warmup 拐杖训练起来。',
  'cc-gpt-assembly': '整个 124M 模型是四块：token 表、加进来的位置表、依次点亮的十二个相同 block、一个最终 LayerNorm，加一个语言模型 head。head 不是新权重——它复用转置的 token embedding 矩阵，所以 weight tying 在这个规模省下约 3800 万参数，总数正好落在参考实现的数字上。',
  'cc-weight-remap': '公开 checkpoint 携带的是原实现的参数名，不是你的。加载器是一个字符串到字符串的映射器，喂给唯一的形状检查：匹配的张量在 no_grad 下拷进本地名字，不匹配的记日志并拒绝——绝不盲目拷贝。每次赋值都有记录，加载错误会出现在报告里，而不是生成时变成沉默的乱码。',
  'cc-sft-loss-mask': '指令微调把每个样本连同边界 token 打包成一条序列，但指令是给定而非要学的。collate 函数把这些位置设为 ignore_index −100，交叉熵完全跳过它们。只有 response 边界之后的 token 参与损失——梯度教模型的是「产出答案」，不是「背下 prompt」。',
  'cc-hdf5-corpus': '一次写一篇文档会把文件打碎；一次性全写完，一崩就丢整个 shard。诚实的纪律是先缓冲再扩展：token 攒到与 chunk 大小一致，把数据集恰好扩一个 chunk，写入新区间。训练时，memory-mapped 切片把一个 hyperslab 从页缓存直接拷进 batch 缓冲。',
  'cc-grad-accumulation': '加速器装得下 32 个样本，损失曲线却要 512——那就逐 micro-batch 跑 backward，让梯度在参数缓冲里求和。每份损失除以累计次数，让运行总和等价于一次全量 batch 的 backward；缓冲满时优化器恰好步进一次，同步推迟到最后一个 micro-batch。',
  'cc-atomic-checkpoint': '恢复需要完整的训练状态——模型、优化器、调度器、步数计数、以及每个随机性来源的 RNG 状态——否则恢复后的损失曲线会偏离不间断的那条。保存先写临时文件、fsync、再 rename 覆盖最终名字。POSIX rename 原子生效，写到一半崩溃留下的也是上一个完好 checkpoint，不是半具尸体。',
};
