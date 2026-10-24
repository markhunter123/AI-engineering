/** zh translations — figures-agents-alignment (9 figures) */
module.exports = {
  'agent-loop': 'Agent loop 是一条因果轨迹，不是三个轮流强调的方框。每个 thought 选一个 action，工具调用产出 evidence，observation 更新 context，而新的 context 改变下一个 thought。',
  'react-trace': 'ReAct 把推理与行动交错。每个 Thought 决定做什么，每个 Action 调一个工具，每个 Observation 把结果喂回来。把推理显式写出来，Agent 才能从走错的一步回头，而不是将错就错。',
  'tool-routing': 'router 拿 query 和每个已注册的工具描述打分、选最接近的。function 名字和描述不是装饰：它们正是 router 决定调用哪个工具所用的信号。',
  'swarm-messages': '如果每个 Agent 都跟其他每个 Agent 说话，消息数按 N·(N−1) 增长，朴素 broadcast 是平方级扩展。所有流量改经 supervisor 中转，边数降为线性——大型系统因此都把协调集中起来。',
  'supervisor-hierarchy': 'supervisor 把任务拆给 worker Agent，worker 自己还可以再 supervise。Agent 总数是分支因子沿深度的几何和——很小的扇出也会让人数迅速爆炸。树要浅。',
  'rlhf-reward-kl': 'RLHF 最大化 reward 减去一项把 policy 拉近 reference 模型的 KL 惩罚。β 太小，惩罚咬不住，policy 就追着代理 reward 漂移、钻奖励模型的空子。KL 项就是拴住 reward hacking 的那条绳子。',
  'dpo-margin': 'DPO 直接在偏好对上训练，没有单独的奖励模型。损失是 β 倍 chosen 与 rejected 之间隐式 reward 差距的 −log σ。正差距越大损失越接近零；负差距（rejected 排更高）重罚。',
  'context-budget': '每一轮都往固定窗口里追加 token。累计不断攀升直到逼近上限，此时 Agent 必须把旧轮次压缩成摘要，或者移交一个全新 context。长会话的生死全看怎么管理这笔预算。',
  'guardrail-gates': '安全以有序闸门运行：模型前的输入过滤器、请求上的 policy 检查、响应上的输出过滤器。第一道绊响的闸门就拦下请求——不安全 prompt 到不了模型，不安全输出到不了用户。',
};
