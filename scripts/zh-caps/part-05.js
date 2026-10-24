/** zh translations — figures-autoswarm5 (12 figures) */
module.exports = {
  'a5-scaffold-delta': 'Claude Sonnet 4.5 在 SWE-agent v1 脚手架内的 SWE-bench Verified 得分 43.2%，放进 Cline 自主脚手架则达到 59.8%。权重相同，差距 16.6 个百分点。检索层、planner、沙箱和「编辑—验证」循环如今与模型本身同等重要。',
  'a5-guard-sieve': 'Llama Guard 守在模型两侧，按 MLCommons S1–S14 危害分类法对输入和输出分类。明显的滥用被廉价地拦下。但 Huang 等人 2025 年测得：六个防护系统上 emoji 走私攻击成功率均为 100%——分类器是一层防线，不是解决方案。',
  'a5-rsp-ladder': 'RSP v3.0 把 AI R&D-4 定为下一档阈值：一个能以有竞争力的成本自动化相当比例 AI 研究的模型。Claude Opus 4.6 位于其下，不过 Anthropic 承认，要自信地排除它正变得越来越难。承诺现在拆分为单边行动与行业建议两部分，2023 年的暂停条款已删除；SaferAI 把该政策评分从 2.2 降到 1.9。',
  'a5-tracked-vs-research': 'OpenAI Preparedness v2 把类别一分为二。Tracked Categories 触发 Capabilities 与 Safeguards 报告，部署前由安全顾问组评审。Research Categories——包括 Long-range Autonomy 与 Sandbagging——仅做监控，缓解措施也只是「潜在」。DeepMind FSF v3 做了同样的动作：把自主性并入其 ML R&D 与 Cyber 领域。',
  'a5-horizon-fit': 'METR 让模型跑遍 HCAST、RE-Bench 与 SWAA 任务——横跨几分钟到几小时的专家工时——再对成功率与 log 专家完成时间拟合一条 logistic 曲线。50% 通过点就是 time horizon。它是在没有真实后果的条件下测得的理想化上界，不是对部署表现的预测。',
  'a5-four-risks': 'CAIS 框架把灾难性 AI 风险分为恶意使用、AI 竞赛、组织风险与失控 AI 四类。类别相互重叠：一个在竞赛中为速度牺牲审计的实验室发布的失控 AI，同时属于全部四类。组织风险是从业者真正能动手改变的一格——这正是它被高亮的原因。',
  'a5-primitive-radar': 'Agent、handoff、共享状态、orchestrator：四个原语撑起整个设计空间。OpenAI Swarm 靠 agents 加 handoffs，状态留给调用方。LangGraph 的重心在 StateGraph 与确定性的图 orchestrator。CrewAI 偏向角色化 agents 加一个 manager 进程。每个新框架发布都只是同一张雷达图上的又一个形状。',
  'a5-og-narrator': 'LLM 把「决定报价」和「叙述报价」混在一起，在严格参数化的议价里只成交 26.7%；扩大规模也治不了。OG-Narrator 把两者拆开：确定性的报价生成器计算每一次数字动作，LLM 只负责写 accompanying message。成交率跳到 88.9%，与 Contract Net 的结论呼应：让机制与通信层分离。',
  'a5-memory-reflection': 'Smallville Agent 维护一条 append-only 的记忆流，按新近度、重要性与相关性打分。Agent 周期性地把近期记忆合成为一次 reflection，reflection 再落回流里，驱动从日级别到动作级别的自上而下计划。三者缺一，可信度都会下降：正是这个循环让一个被种下的派对点子传遍了 24 个无脚本 Agent。',
  'a5-retry-cascade': 'MAST 统计出 1642 条多 Agent 执行 trace 中 36.94% 存在协作失败，而重试风暴是教科书式的级联：支付失败触发订单重试，每个订单重试又触发库存重试，库存服务在几秒内看到 10 倍负载。层与层之间的 circuit breaker 是把放大链条变回一次受控失败的缓解手段。',
  'a5-bench-gap': '前沿模型在 SWE-bench Verified 上得分 70% 以上，在 SWE-bench Pro 上约 23%——后者的 1865 道题要求跨 41 个仓库改动 10 行以上。Verified 接近饱和、部分被污染、还被一两行改动的简单尾部稀释。Pro 是未被污染的现实检验：任何排行榜主张都要对照这两列来读。',
  'a5-orchestrator-scale': 'Anthropic 的 Research 系统按查询复杂度伸缩投入：简单查询用单个 Agent 加少量 tool call；中等查询三个；复杂研究则十个以上并行 subagent。它比单 Agent Opus 4 高出 90.2%，token 用量一项就解释了 BrowseComp 方差的 80%，代价是每个查询 15 倍 token，外加为长时运行 Agent 准备的 rainbow deploy。',
};
