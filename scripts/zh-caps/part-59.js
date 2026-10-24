/** zh translations — figures-swarms3 (8 figures) */
module.exports = {
  'sw-contract-net': 'FIPA contract-net 协议把任务分配变成密封拍卖。manager 广播征集提案，空闲 Agent 回以报价，manager 把合同判给最优报价。MCP 的 tools/call 和现代任务市场，都是这个 1980 年代机制的 JSON 版重述。',
  'sw-work-stealing': '集群没有中央调度器。任务落在共享队列上，空闲 worker 自己拉走下一件。协调活在队列语义里，系统扩展到队列扩展为止。代价是确定性：你用一份连贯的全局计划换来了吞吐。',
  'sw-handoff-routing': 'OpenAI Swarm 把编排压缩成两个原语：routine（prompt 加工具）和 handoff（一个返回下一个 Agent 的工具）。没有状态机。模型靠调用正确的 handoff 来路由，此刻握着对话的 Agent 就是当家的。',
  'sw-agent-card-discovery': 'A2A 是 Agent 之间的横向线协议。客户端先从 well-known URL 取一张 Agent Card，弄清远端 Agent 会什么，再发一条消息——远端 Agent 把它变成任务。任务走一段不透明生命周期（TASK_STATE_SUBMITTED、TASK_STATE_WORKING、TASK_STATE_COMPLETED），返回 artifact。它就是 HTTP 加 REST，把 Agent 重新表述成一等公民的对等端。',
  'sw-debate-topology': '聚合 N 个 Agent 不是简单多数决，接线方式很重要。star 全部经过枢纽，chain 传递接力棒，tree 分叉，graph 让所有人互辩。MultiAgentBench 发现 graph 最适合研究型任务，代价是超过约四个 Agent 后协调税开始爬升。',
  'sw-theory-of-mind': '真正的协调需要 Agent 彼此建模。高阶 theory of mind 是推理「一个 Agent 认为第三个 Agent 相信什么」。Riedl 2025 发现：只有在 theory-of-mind prompt 下，这才产生真正面向目标的分化；撤掉 prompt，表面上的协调在统计控制下不复存在。',
  'sw-ctde': 'Centralized Training, Decentralized Execution 是合作式 MARL 的脊梁。训练时 critic 看得见每个 Agent 的状态和动作，治好了困扰独立学习者的非平稳性。测试时 critic 退场，每个 actor 只凭自己的局部观测运行。MADDPG、QMIX、MAPPO 是同一拆分的三种演绎。',
  'sw-checkpoint-replay': '持久化执行是多 Agent 系统扩展出单台笔记本的关键。运行时每步之后按线程 id 写一个 checkpoint。worker 中途崩溃时租约释放，另一个 worker 接起任务、从最后一个已提交的 checkpoint 续跑，而不是从头再来。',
};
