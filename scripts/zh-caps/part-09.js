/** zh translations — figures-misc2 (8 figures) */
module.exports = {
  'mx-propose-then-commit': '2026 年的 HITL 形态不是同步的 Approve 弹窗。被提议的动作先带着幂等键持久化，展示意图、影响半径和回滚计划，仅在明确确认后提交，然后再验证一遍确认副作用确实发生了。真正的失败模式是橡皮图章：看都不看就点 Approve。带明确清单的挑战—应答机制是有据可查的缓解手段。',
  'mx-priority-tiers': '2026 年的 Claude Constitution 把行为排成四层：安全与支持人类监督最高，伦理次之，Anthropic 准则第三，helpfulness 最后。层与层冲突时高层胜出——与 Unix 优先级或网络 QoS 同构。硬编码禁令凌驾于整个栈之上、不可覆盖；其余一切都在层级内按理由裁决，用可审计性换取对未见过情况的泛化。',
  'mx-research-loop': 'AI Scientist v2 在没有人类模板的情况下闭合研究循环：生成想法、检查新颖性、在沙箱里起草并运行实验、让视觉语言模型点评图表、撰写论文，并在内部评审中迭代。一篇生成的论文通过了 ICLR 2025 workshop 评审。独立评估发现 42% 的实验死于代码错误、新颖性检查常把成熟方法误标为新方法——所以循环里决定可靠性的是沙箱这一侧的边缘。',
  'mx-speculative-tree': 'EAGLE-3 在目标模型的隐藏状态上训练一个 draft head，其分布紧贴目标，通用聊天里接受率 alpha 落在 0.6 到 0.8 区间。draft 一次提出 K 个 token；目标模型用一次 forward 验证全部 K 个；被接受的 token 近乎免费。但每个被拒绝的 draft 都要再花一次目标模型 forward——所以当 alpha 低于约 0.55 且高并发时，这项技术是净亏损。先在真实流量上量 alpha，再拨开关。',
  'mx-gateway-fallback': '网关坐在你的应用与供应商之间，把路由、回退、重试、限速、密钥引用和可观测性收进一个 OpenAI 兼容 API。当主供应商返回 429 或 5xx，网关改道 fallback 供应商，请求依然完成。2026 年的格局：LiteLLM（MIT 协议、100+ 供应商、接近 2000 RPS 时崩）、Portkey（控制面、guardrails）、Kong AI（自家基准里最快）、Bifrost（自动重试）。数据驻留要求决定自托管还是托管。',
  'mx-sequential-test': 'evals 回答「模型能不能干这活」；A/B 测试回答「用户在不在乎」。固定样本量测试惩罚偷看，所以 2026 年的平台用带始终有效边界的 sequential testing：边界早期宽、随时间收紧。累计统计量随机游走，越过边界就停止并上线；一直待在界内就判平。CUPED 降方差，Benjamini-Hochberg 校正同时测多个变体带来的多重比较。',
  'mx-schema-funnel': 'LLM 返回字符串，你的应用需要类型化 JSON。在 prompt 里加一句「用 JSON 回答」大约九成时候有效，剩下一成直接崩溃。constrained decoding 在 token 层面补上这个缺口：有限状态机或文法对 logit 加掩码，任何会破坏 schema 的 token 在被采样之前就被禁止。输出从构造上就是合法 JSON，不靠后处理；Pydantic 层做类型校验，偶尔失手就重试。',
  'mx-tool-call-loop': 'LLM 只会生成文本，所以它查不了天气也查不了数据库。function calling 是弥合差距的协议：模型输出结构化 JSON，点名要调用哪个函数、传什么参数；你的代码执行它；结果回灌进对话。模型是大脑，工具是双手，这个循环是神经系统。它不断重复，直到模型不再请求调用、给出最终答案——期间有并行调用与无限 tool loop 的防护。',
};
