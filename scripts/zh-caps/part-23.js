/** zh translations — figures-capstone-i (9 figures) */
module.exports = {
  'ci-chunk-boundaries': '每种策略在同一份文档的不同偏移处下刀。固定的 512-token 窗口直接切穿金标准答案区间——两半被嵌入进不同的聚类，retriever 还没跑 recall 就掉了。句子级和结构化 markdown 切分器尊重文本自带的边界。',
  'ci-rag-metric-ladder': '评估器把排序列表走一遍。每经过一份金标准文档就更新 precision 与 recall；第一份命中的位置定下 MRR；带折扣的位置定下 nDCG。faithfulness 和 answer relevance 是事后在生成的答案上算的——所以检索满分照样可能答错。',
  'ci-task-spec-gate': 'tasks.jsonl 的每一行在到达 runner 前独立校验：必填字段齐全、metric_name 在封闭词表内、target 类型正确。格式错误的记录被改道并记日志，运行继续。正是冻住这道闸门，71 到 75 课才能靠单个字段分发。',
  'ci-leaderboard-ci': 'aggregator 把逐任务 EvalRun 记录聚成每模型均值，再对任务做有放回重采样，给每个均值套上 bootstrap 置信区间。模型 A 的点估计领先模型 B，但区间重叠——真正决定排名是真实还是采样噪声的，是成对差值的 CI。',
  'ci-ring-allreduce': '每个 rank 拥有张量的一块，只和环形邻居通话。reduce-scatter 阶段，一块数据每跳一次累计部分和、共跳 N−1 次；allgather 阶段，完成的和再轮转回一圈。没有 root、没有瓶颈：每 rank 流量固定在 2T(N−1)/N 字节，多少 rank 加入都不变。',
  'ci-ddp-grad-sync': '每个参数上的 backward hook 把它的梯度丢进桶。1 号桶一满，它的 allreduce 就上线——此刻第 2、1 层还在反向传播，通信藏在计算背后。构建时的一次 broadcast 让所有副本一致；逐步求和的梯度让它们每一步保持一致。',
  'ci-sharded-checkpoint': '每个 rank 同时把自己的 shard 流式写到临时文件——写带宽随集群扩展，而不是堵在单一 gather 上。所有 shard 落地之后，rename 才把临时文件和 manifest 一次性翻成正式版。恢复先读 manifest，核对 world size 与哈希，任何不匹配都大声失败。',
  'ci-distributed-assembly': '每个部件都单独构建并测试过；结业项目证明它们能组合。DDP broadcast 同步一次初始权重，ZeRO-1 的步进每轮取代 optimiser.step，分片 checkpoint 在第 10 步触发。只有四个不变量在每个 rank 上全部成立，运行才会在 20 步后自动终止。',
  'ci-refusal-quadrant': '评估器把助手当作 prompt 安全性上的二分类器。安全 prompt 被拒——落进过度拒绝格；不安全 prompt 被回答——落进拒绝不足格。两个非对角色都是 bug，按分类学类别分开度量，ECE 追踪模型是否知道自己何时是对的。',
};
