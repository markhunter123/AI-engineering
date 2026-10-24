/** zh translations — figures-tools2 (8 figures) */
module.exports = {
  'tp-tool-loop': '每套 tool calling 栈都跑同一个四步循环。host 描述工具，模型决定调哪个，host 真实执行，模型在下轮之前观察结果。数据包绕圈，是因为循环重复到模型不再需要调用为止。',
  'tp-parallel-fanout': '单轮模型可以同时发出几个独立的工具调用。host 并发执行，总延迟从所有调用之和塌缩到最慢单个的时长。这里三个天气查询各有各的时钟，但这一轮只等最长的那个。',
  'tp-schema-routing': '选择是一个匹配问题。模型读每个工具的名字和描述，把 query 路由到最贴合的。含糊或互相重叠的描述让光束游移、选错；锋利的「何时用 X、何时不该用」措辞把它锁死在一个工具上，准确率抬升十到二十个点。',
  'tp-client-merge': '真实的 host 同时加载多个 MCP server，把各自发现的工具列表摊平成模型看到的一个命名空间。client 对每个 server 调 server/discover，给名字加前缀避免冲突，并记住每个工具属于哪个 server——调用才能路由回正确的进程。',
  'tp-transport-handshake': '两种传输、两种部署形态。stdio 用 stdin 和 stdout、配一个本地子进程。MCP 2026-07-28 Streamable HTTP 是无状态的：每条 JSON-RPC 消息各自 POST 到同一个端点，该请求返回 JSON 或 request 范围的 SSE。没有 Mcp-Session-Id、独立 GET 流，也没有会话 DELETE。',
  'tp-task-lifecycle': '官方 tasks 扩展让长时间运行的工作返回一个 task 句柄，而不是把请求一直挂着。client 用 tasks/get 轮询；终态的 task 自带最终结果。task 进入 input_required 时，tasks/update 提供对未决 inputRequests 的响应，tasks/cancel 表达取消意图。当前扩展没有 tasks/list 或 tasks/result 方法。',
  'tp-router-failover': 'routing gateway 在多个 provider 之上给一个统一 API 表面。最高优先级的 provider 出错时，请求沿 fallback 链重试直到有人应答——不用重新部署。同一层还按请求追踪成本和 token，让每个工作负载落在「够用中最便宜」的模型上。',
  'tp-tool-poisoning': '工具描述是 prompt 的一部分。恶意 server 可以埋一句用户永远看不到的指令，让模型去读一个秘密、悄悄泄漏出去。接口看起来完全正常——所以防御是给描述做 hash pinning、在 CI 里扫描 injection 模式，而不是信任。',
};
