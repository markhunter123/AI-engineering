/** zh translations — figures-autonomous2 (9 figures) */
module.exports = {
  'task-decomposition': '长程 Agent 不正面硬啃复杂目标。它把目标分解成子任务，子任务再分解，直到叶子都是能直接执行的步骤。分支更宽、树更深，计划得更彻底，但要追踪的工作量成倍涨——所以有用的计划保持浅。',
  'reflection-loop': '自我改进循环：行动、评估结果、批评哪里出了错、在下一次尝试前修订。每一轮都抬高质量，但收益几何式缩水、很快走平。功力在于知道反思什么时候开始不回本。',
  'memory-consolidation': '最近的步骤以细致的 episodic 记录活着。缓冲越过阈值，最老的 episode 被压缩成保留大意、腾出窗口的紧凑 semantic 摘要。长期运行的 Agent 靠巩固活下来，而不是永远记住每个 token。',
  'world-model-rollout': '基于模型的规划在触碰世界之前先模拟未来。从当前状态出发，Agent 想象每个候选动作会到的状态，用学到的模型把它们向前推演，然后才落子最好的一步。rollout 越深计划越好，想象之树却指数生长。',
  'autonomy-oversight': 'human-in-the-loop 是个旋钮，不是开关。一条自主阈值让低风险动作无人值守地跑，超过它的统统停下来等人工批准。要速度调高旋钮，要控制调低。无论旋钮停在哪，上生产都要靠近最高档。',
  'pass-at-k': 'pass@k 问的是 k 个独立采样里是否至少一个解出任务。若每次尝试成功率是 p，全失败的概率是 (1−p)^k，pass@k 就是一减去它。再弱的模型，采样一多也陡峭攀升——所以 best-of-k 是这么便宜的杠杆，pass@1 和 pass@k 讲的是不同的故事。',
  'eval-harness-matrix': 'eval harness 让每个任务对着每个模型变体跑，把 pass/fail 记进网格。顺着列读，看出哪些任务难；顺着行读，给一个变体算出总分。这张矩阵把模糊的直觉变成一个能拿去做回归测试的数字。',
  'canary-rollout': 'canary 发布把一小片流量切给新版本，其余留在久经考验的版本上。盯住 canary 的错误率对 SLA；一旦越线，流量立刻退回稳定版。这里 canary 跑得火烫，所以扩大切分比例抬高混合错误率、回滚始终保持待命。',
  'trace-spans': '分布式 trace 是铺在时间轴上的 span 树。root span 盖住整个请求；LLM 调用、retrieval、工具调用各自的 child span 按起始时间和时长嵌在里面。读 gantt 图就知道延迟到底去了哪儿——这是每起生产事故的第一个问题。',
};
