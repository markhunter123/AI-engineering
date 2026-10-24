/** zh translations — figures-capstone-d (8 figures) */
module.exports = {
  'cd-hyde-vector': '原始 query 向量指向错误的区域，正确答案的文档根本进不了 top-N。HyDE 让模型写出「能回答这个问题的文档」，改为嵌入那份假想答案——检索向量正好落在真实文档所在的位置。',
  'cd-bleu-overlap': 'BLEU 就是数数。候选 n-gram 每有一个出现在参考里就得一分；截断后的计数除以候选长度，即修正 n-gram precision。候选比参考短要吃 brevity penalty，所以残根上的高 precision 照样输。ROUGE-L 则奖励最长公共子序列。',
  'cd-reliability-diagram': '把预测按声明的置信度分桶，画出实际准确率。校准好的模型贴着对角线。橙色缺口是它要的信心超过挣来的信心的地方；expected calibration error 就是这些缺口按桶加权的平均。Perplexity 问的是 held-out 文本像不像话；calibration 问的是信心诚不诚实。',
  'cd-zero-shard': '朴素 DDP 在每个 rank 上复制整套优化器状态——栈里最大的一块分配。ZeRO 第 1 阶段只给每个 rank 1/N 的 Adam moments。reduce_scatter 把各自的 gradient shard 送到每个 rank，它本地 step，然后 allgather 把更新后的参数 shard 广播回去、每个 rank 重建完整模型。内存线性下降；线上开销等于一次 allreduce。',
  'cd-pipeline-bubble': '每级 stage 住在自己的 rank 上；一个 microbatch 进入 stage 0，把 activation 递给 stage 1，沿时间对角线传递下去。阴影三角是 bubble——pipeline 注满和排空时的空转。每步的 microbatch 越多 bubble 越小：(N−1)/(M+N−1) 从 M=4 的 43% 降到 M=64 的 5% 以下。',
  'cd-attack-taxonomy': '没有分类学的安全 harness 就是掷硬币。六个类别各自点名攻击滥用的一条信任边界——从 assistant persona 到它的 next-token 决策。给边界命名，攻击流才变成直方图，直方图才变成覆盖率图，覆盖率图才变成下一个 sprint。',
  'cd-output-router': '只查输入不够；输出侧的分类器看的是真实回复，问的是能不能发。三个独立分类器（toxicity、PII、leakage）各回一个严重度，router 按最坏发现选动作：log、warn、redact 或 block。分类器与流式并行运行，延迟藏在最后一次 flush 后面。',
  'cd-constitution-loop': '一条规则要有名字、谓词和解释——缺一样就只是种感觉。constitution 以 YAML 存在版本控制里。引擎标记每条违规，fixer 提出修订，第二遍确认修订满足所有规则之后，回复才被接受或上报。',
};
