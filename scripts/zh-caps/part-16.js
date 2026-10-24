/** zh translations — figures-capstone-f (10 figures) */
module.exports = {
  'cf-scene-index': '摄入把视频切成场景，每个场景并排存三个向量：caption embedding、关键帧 embedding、转写 embedding。一次查询同时对三者发起，结果合并，答案以 top 场景内的一个 (start, end) 窗口返回，而不是整个文件。',
  'cf-mcp-gate': '每条 JSON-RPC 消息各走一次 POST，携带协议版本和 client capabilities。闸门验证 issuer、audience、scope、tool 和参数；有后果的调用还需要一条绑定到这个精确动作的批准记录。注册表索引 server.json 的发布元数据；另有一个独立的 server/discover 探测验证在线端点实际支持什么。',
  'cf-spec-decode': 'draft head 提议五个候选 token；目标模型一次 verify pass 给全部打分。被接受的前缀顶替三个串行 decode 步，被拒绝的后缀被丢弃重采样。接受率决定加速比——而被拒绝时那次更大的 verify pass，恰恰是 p99 延迟需要单独一份报告的原因。',
  'cf-safety-stack': '干净的请求一路直落：sanitize、rails、classifier 闸门、模型、输出过滤。一个 jailbreak 在第二层就被 classifier 闸门抓住并弹开。红队靶场从外面持续试探每一层；任何被输出过滤标为高风险的内容都绕道进入人工审核队列。',
  'cf-issue-to-pr': '一个带标签的 issue 触发 GitHub App webhook；dispatcher 在入队前检查每个仓库的每日预算。沙箱从零复现构建，并扣住任务直到完整测试套件通过。只有绿色的 CI 闸门才开 PR；禁止 force-push 的是分支保护，不是 Agent 的自觉。',
  'cf-tutor-loop': 'tutor 从不倒答案：学习者的每次回复都换来一个引导性提问或一段搭手架提示。每次交互都更新当前概念的掌握概率；进度条填满后，policy 沿课程图里的先修边走过去，点亮下一个概念。',
  'cf-loop-contract': '这个 loop 是确定性状态机，不是聊天式的 while 循环。一枚 run token 从 IDLE 走过 PLANNING、EXECUTING、AWAITING_TOOL 和 REFLECTING，按预算许可的轮数走内层 execute-reflect 循环，最后落进 DONE。每次转移都往流上发一个带类型的事件，UI 和 tracer 因此只需订阅，而不用去撬开 loop。',
  'cf-registry-validate': '注册表把名字、schema、handler 一次性钉死，dispatcher 从此信任它。坏的调用永远到不了 handler：schema 检查用一个 json-pointer 路径把它弹回去，模型一轮就能修好。修正后的调用过同一个纯验证器，然后才碰代码。',
  'cf-jsonrpc-frames': '每条消息就是一行上的一个 JSON 对象。request 带 id，得到恰好一条同 id 的 response；notification 不带 id，必须什么也得不到。乱码行换来 id 为 null 的 -32700 parse error，紧接着的下一行照常解析——一条坏帧永远不会毒死整条流。',
  'cf-dispatch-retry': '第一次尝试跑到它的 per-call 超时，返回一个带类型的错误而不是把 loop 挂死。每次重试前，backoff 加抖动翻倍；与在途尝试赛跑的重复调用在幂等键上坍缩进同一次。无论发生什么，loop 收到的都是一种信封形状：result 或映射后的错误，绝不是原始堆栈。',
};
