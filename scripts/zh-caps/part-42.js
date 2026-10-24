/** zh translations — figures-agents3 (8 figures) */
module.exports = {
  'htn-tree-expand': 'HTN planner 把复合任务展开成方法，每个方法再展开成子任务，递归到每个叶子都是前置条件成立的原子 operator。树自顶向下长出；计划就是原子叶子的从左到右读序。',
  'workflow-chain': '最简单的 workflow：一条固定的线性模型调用链，上一个输出是下一个输入，步骤之间可插程序化的闸门。图由工程师全权所有，调试便宜、运行可预测。只有步骤无法预知时，才伸手去拿 Agent。',
  'actor-mailbox': '每个 Agent 是一个 actor：私有状态、一个信箱、一个 handler。actor 之间从不共享内存，只发消息，运行时把投递和处理解耦。崩溃只隔离到一个 actor，并发是天然的，搬到分布式部署只是换一种传输方式。',
  'debate-converge': '独立的模型实例各自提出答案，然后互相阅读、批评，数轮之间向一致更新。浮出错误的是分歧，而不是一条单独的 chain of thought。稀疏拓扑（不是图示的全连接）能用几分之一的 token 成本打平准确率。',
  'computer-use-cursor': '视觉式 computer use 从截图读像素、发出与分辨率无关的坐标，再发键盘鼠标命令——不用 accessibility API。屏幕上的一切都是不可信输入；只有用户的直接指令算授权，所以每一步动作前都有安全检查把关。',
  'voice-pipeline': 'voice agent 是一条以帧为单位的 pipeline，不是文本外挂 TTS：voice activity detection、speech-to-text、LLM、text-to-speech，全部压在严酷的 ~600ms 预算里。partial 音频是默认，barge-in 取消逆流而上——每一段都必须流式，而不是等完整一轮。',
  'injection-hijack': '间接 prompt injection 把指令埋进 Agent 检索到的内容里。模型无法可靠区分用户意图与检索文本，恶意 token（红）就把 Agent 劫持到攻击者选定的工具。把所有检索内容当成 tool-use 表面上的任意代码——任何调用提交前先校验。',
  'failure-cascade': 'Agent 的失败不是随机噪声，它们落入反复出现的模式。级联错误代价最高：一个幻觉步骤写进下一步的输入，一个错误动作沿 plan、retrieval、reasoning、action 层层传播。给模式命名，你才能监控它、并及早剪断链条。',
};
