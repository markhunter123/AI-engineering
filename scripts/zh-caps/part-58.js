/** zh translations — figures-swarms2 (8 figures) */
module.exports = {
  'swarm-consensus-wave': '决策沿环涟漪般扩散，诚实的 Agent 收敛到共享值。单个拜占庭节点（X）拒绝翻转，朴素多数决仍可能被俘获。经典 BFT 容忍 f < n/3 个这种节点；对 LLM Agent 来说，开放问题是相关故障，不是任意故障。',
  'swarm-auction': 'Agent 为任务竞价，报价随轮次上涨。出价最高者（a1）赢得任务，却只付第二高的价——虚线。在第二价格机制下，诚实报价是占优策略——机制设计因此偏爱用它给 Agent 之间分配工作和 token。',
  'swarm-stigmergy': '没有哪个 Agent 在规划路线。每个 Agent 走过时留下信息素、偏爱更浓的痕迹，最短路因此积聚流量、绕路蒸发。ACO 把这一套变成 Agent 路由：痕迹记录哪个 Agent 处理过哪种任务类型，衰减让更好的路线可以被重新发现。',
  'swarm-hierarchy-token': 'manager 拆分目标，经 sub-manager 层层委派给 worker；结果沿同样的边返回。风险在于：LLM manager 每轮都要重新推理整棵树，context 的一点漂移就会错配工作、结构陷入循环。很多时候，一条扁平序列反而更强。',
  'swarm-message-bus': 'Agent 不再互传原始字符串，而是在共享总线上讲带类型的协议。MCP 载工具调用，A2A 载委派任务（和回复），ANP 载身份。分开车道，多 Agent 系统才可审计，不同团队造的 Agent 才能互操作。',
  'swarm-roles': 'Agent 更多没用，Agent 不同才有用。计划、工件、主观批评、确定性检查是各自带工具的独立角色。verifier 是承重墙：MAST 把几乎每一起多 Agent 失败都追溯到缺失或损坏的验证。一个 reject 就把工件送回去重新计划。',
  'swarm-blackboard': 'Agent 通过中央黑板共享事实，而不是复制消息。危险是记忆投毒：一个 Agent 写进一段幻觉，下游每个读者都把它当作已验证采纳，准确率无声下滑。撑得住的缓解是来源记录、不可写的 verifier、以及每 Agent 独立视图。',
  'swarm-speaker': 'Agent 对一个共享池做反应，而不是对固定图。一个选择器——轮转、LLM 或自定义规则——挑出下一个发言者，token 在候选间跳转、最终落定一个。这避免了在 N 个 Agent 之间硬编码每种交接的边爆炸。',
};
