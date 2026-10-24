/** zh translations — figures-tools3 (11 figures; wins over figures-mcp duplicates) */
module.exports = {
  't3-dispatch-loop': 'server 从 stdin 每次读一行 JSON 对象。带 id 的消息是 request，必须在 stdout 上恰好产生一条携带同一 id 的 response。notification 没有 id，也不产生任何输出。任何打到 stdout 上的其他东西都会污染线路——这就是 debug 输出走 stderr 的原因。',
  't3-primitive-sort': '不是所有东西都是 tool。模型应该按查询决定是否调用它——那是 tool。用户把它作为上下文附加——那是 resource。可复用单元是整个工作流——那是 prompt。按这个方式整理一个笔记 server，能消掉纯读取场景的模型往返，让每项能力落在宿主本来就为它构建好的 UX 界面上。',
  't3-sampling-flip': '对新的 MCP 设计而言 sampling 已废弃；直接调用模型供应商即可。兼容性 server 在当前协议下不会发出反向 request：它返回 resultType 为 input_required 的 sampling inputRequest，client 在自己的策略下取得补全，然后带着 inputResponses 和逐字节相同的 requestState 重试原方法。',
  't3-roots-boundary': '把被授权的目录或资源 URI 作为显式 tool 参数、资源引用或 server 配置传入。server 在访问之前解析并约束每一条路径：内部读取成功，外部路径被拒绝。MCP Roots 仅在其废弃窗口内可用，且绝不替代授权或 OS 沙箱。',
  't3-ui-sandbox': 'tool 结果指名一个 ui:// 资源，宿主把它的 HTML 渲染在沙箱化 iframe 里。iframe 拿到锁死的 CSP，除非元数据明确授予否则没有网络——进出沙箱墙的唯一通道是一套极小的 postMessage JSON-RPC 方言。同一份 HTML bundle 在所有兼容 client 里渲染一致。',
  't3-scope-stepup': 'scope 为 notes:read 的 token 撞上一个需要 notes:write 的动作。server 不让调用失败，也不重跑整套 OAuth 流程，而是以 403 应答，并用 WWW-Authenticate header 点名缺失的 scope。client 请用户只为这一增量授权，然后带着升级后的 token 重试。最小权限保持默认，因为升权的代价这么低。',
  't3-gateway-funnel': '对每个开发者来说，网关看起来就是一个 MCP server。内部：每次调用先认证，再对照每用户 RBAC 检查、限速、与固定的 tool-hash manifest 比对、写入审计日志，然后才路由到拥有该 tool 的后端。策略住在同一个地方，而不是五千份 IDE 配置里。',
  't3-jwks-rotate': '授权服务器按计划把签名密钥从 kid A 滚动到 kid B。启动时只取过一次 JWKS 的资源服务器从那一刻起拒绝所有 token。生产形态是：缓存密钥集 + 刷新任务在旧密钥过期前覆盖缓存 + 缓存未命中时的回退拉取——新密钥签发的 token 在凌晨 3 点也能通过验证，无需重启。',
  't3-span-waterfall': '一个 trace id 覆盖整轮。agent span 最先打开、最后关闭；它内部，一次 LLM 调用、一次 tool 执行和被包裹的 MCP dispatch 各自拥有带 gen_ai 属性的 span。当后端冷启动，被拉长的正是 mcp.call 那根条——而这恰恰是光靠日志回答不了的问题。',
  't3-skill-layers': 'AGENTS.md 告诉任何 Agent 这个项目如何运作。SKILL.md 把一个工作流打包成 frontmatter 加正文，运行时渐进披露。MCP 提供 skill 调用的 tools。因为每一层都是开放格式的普通文件，同一套 bundle 直接落进 Claude Code、Cursor 和 Codex，而不是复制三份各自漂移。',
  't3-capstone-chain': '每个无状态 MCP 请求都携带 version 与 capabilities。网关做认证并施加策略；server 可以返回官方 task 扩展句柄，client 用 tasks/get 轮询；独立工作经 A2A SendMessage 委派。最终的 ui:// 报告与每个边界 span 保持关联，不依赖任何协议会话。',
};
