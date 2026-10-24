/** zh translations — figures-claude-certifications (6 figures) */
module.exports = {
  '00-certification-route-map': '准备就绪不是蓝图看起来多眼熟，而是你能解释什么、能在时间压力下做什么、能拿什么 artifact 证明。',
  '04-context-cache': 'Prefix caching 跳过重复的 prompt 计算。semantic caching 为相似 query 复用旧答案。前者精确、在 provider 侧；后者近似、在应用侧——所以它的阈值是一个安全决策。',
  '10-tool-loop-budget': '工具循环需要显式的调用预算、终止条件和类型化的失败。更多重试能抬完成率，也同时抬高延迟、成本和副作用风险。',
  '17-session-context-budget': '会话记忆应该保留决策、约束和未决状态，而不是回放每个 token。compaction 是一个信息设计问题。',
  '24-rag-ranking': '检索质量不只是最近邻相似度。选出的证据还必须覆盖答案打算断言的那些内容。',
  '26-latency-cost-slo': '平均数掩盖架构。模型延迟、cache 行为和允许的服务目标必须当作一个系统来测量。',
};
