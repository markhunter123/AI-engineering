/** zh translations — figures-agents2 (8 figures) */
module.exports = {
  'rewoo-plan': 'ReWOO 把推理和观察分开。planner 一遍写完整条工具调用链，worker 依序执行、步骤之间不再调模型。ReAct 则在每个动作前都调一次模型、重读整条不断变长的轨迹——跑得越久 token 花得越多。',
  'tree-of-thoughts': 'Tree of Thoughts 把单条链变成搜索。每一步模型提出几个候选想法；打分器排序，beam 只留最好的几个继续展开。剪掉弱枝让指数级的树变得养得起，同时 Agent 还能从坏的一条推理线回头。',
  'self-refine': 'Self-Refine 让同一个模型写答案、批评自己的作品、再修改，把输出循环回输入。头几轮抓明显的毛病、涨得最多；后面几轮越来越没得改，质量爬向天花板然后走平。功力在于知道额外几轮什么时候开始不配它们的 token。',
  'memory-blocks': '结构化 Agent 记忆把常驻上下文窗口的固定 core block，与磁盘上无界的 archival store 分开。Agent 读写自己的记忆：core 满了就总结，或把最不相关的条目逐出进 archival，需要时再把别的换页进来。Agent 就这样在有限窗口里装下实际上无限的历史。',
  'voyager-skills': '终身 Agent 把每个解过的任务变成一个命名的、可复用的 skill 存起来。episode 越攒越多，库越长越大；后面的任务靠组合架上已有的 skill 更快解决，而不是从零开始。能力复利：每次成功都让下一个问题更便宜。',
  'langgraph-state': '有状态图把 Agent 建模成节点加边，一个共享 state 对象贯穿每个节点。每个节点读状态、干活、写回。条件边按状态分支，评分不及可以绕回重写而不是收工。图让控制流显式、状态步步可查。',
  'multi-agent-debate': '辩论里，几个 Agent 先各自独立作答，然后互读对方的记录、跨轮修订。分歧迫使每个人辩护或修正自己的推理，立场通常收敛向一个更好的共同答案。头一两轮就拿走了大部分准确率收益；再往后大家一致了，额外轮次主要在烧 token。',
  'orchestration-pattern': '三种常见的多 Agent 拓扑，各自用韧性换控制。supervisor 把路由集中在 O(N) 条边上，却是单点故障。peer-to-peer 集群没有瓶颈，但付 O(N²) 的协调成本。hierarchy 透过层层 supervisor 分派，扇出可扩展，代价是每层加一截延迟。选哪个，取决于 Agent 之间必须说多少话。',
};
