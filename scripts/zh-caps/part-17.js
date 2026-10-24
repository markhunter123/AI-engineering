/** zh translations — figures-infra (9 figures) */
module.exports = {
  'data-parallel': '每张 GPU 持有完整的模型副本，处理全局 batch 的不同切片。backward 之后，all-reduce 把所有 GPU 的梯度求和，副本保持一致。吞吐量随 GPU 数接近线性扩展，但内存并不下降——每张卡仍然存着整个模型。',
  'tensor-parallel': '把权重矩阵按列切块，一次大矩阵乘法就被拆到多张 GPU 上。每张 GPU 用自己的切片乘完整输入、产出部分输出，再由 all-gather 把切片缝成完整结果。每张 GPU 的参数量按卡数下降——一层大到单卡装不下的模型就是这样被服务的。',
  'pipeline-parallel': '模型被切成若干阶段，每张 GPU 一个，micro-batch 像流水线一样流过。流水线填充与排空时，部分阶段闲着——那个灰色气泡。气泡占比是（阶段数 − 1）除以（micro-batch 数 + 阶段数 − 1），所以喂更多 micro-batch 能把固定的填充排空成本摊到接近零。',
  'zero-sharding': '朴素数据并行在每个 GPU 上复制完整的优化器状态、梯度和参数。ZeRO 分阶段消灭冗余：stage 1 切分沉重的 Adam 优化器状态，stage 2 加上梯度，stage 3 连参数本身也切。每升一级，单卡内存再降一截——用一点通信换取训练大得多的模型的能力。',
  'gpu-memory-breakdown': '训练内存远不止权重。混合精度 Adam 里，每个参数要付：fp16 权重两字节、梯度两字节、优化器状态约十二字节——所以在第一个 activation 出现之前，固定成本已约每参数十六字节。activation 再随 batch 大小增长。这就是为什么推理装得下的模型，单卡训练可能远远装不下。',
  'throughput-latency': '蓝色是吞吐量，灰色虚线是单请求延迟。更大的 batch 让 GPU 更忙，每秒总 token 数爬升；但每个请求在别人后面等得更久，延迟也上升。拐点（橙色）是吞吐几乎不再增长、而延迟持续爬升的地方——大多数 serving 系统瞄准的 batch size 就在这里。',
  'autoscaling': 'autoscaler 增删副本，让到达的负载留在容量之内、延迟压在目标之下。副本数等于负载除以单副本可服务的量，向上取整。QPS 升，副本转起来；QPS 降，副本缩回去——延迟和成本正是这样被同时按住的。',
  'cost-per-token': 'serving 经济学归结为两个数：GPU 每小时多少钱，这一小时产出多少 token。每百万 token 成本等于时价除以每小时服务的 token 数、再按一百万缩放。吞吐翻倍，单价减半——batching、量化、更快的 kernel 全都直接换算成更低的每 token 价格。',
  'roofline': '算术强度是 kernel 每移动一个字节做了多少 FLOPs。左侧斜屋顶表示性能被内存带宽封顶；右侧平屋顶表示被原始算力封顶。屋脊是两者交汇处。落在屋脊左下方的 kernel（橙色）是 memory-bound，解法是提高数据复用，不是换更快的芯片。',
};
