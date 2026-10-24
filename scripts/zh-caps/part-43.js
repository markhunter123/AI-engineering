/** zh translations — figures-agents4 (8 figures) */
module.exports = {
  'ae-memory-fusion': 'Mem0 把每条记忆同时写进三个 store，检索时融合。vector 答语义相似，KV 答事实查询，graph 答关系推理。relevance、importance、recency 的加权分数把三者混在一起——单一 add/search 表面不会像任何单一 store 那样，在三选一的 query 类型里错掉两类。',
  'ae-crew-vs-flow': 'CrewAI 提供两种顶层形状。Crew 是自治的、按角色协作，Agent 在松散网状里互相批评——适合探索性工作。Flow 是事件驱动的确定性链，可重放、可审计、可核算成本。文档说得很直白：任何 production-ready 应用，从 Flow 开始。',
  'ae-agent-handoff': '在 OpenAI Agents SDK 里，handoff 只是一个叫 transfer_to_<agent> 的工具。triage Agent 调用它，控制权和进行中的对话上下文就移交目标 Agent、由它继续会话。把委派建模成普通工具，循环保持统一：模型决定交接与决定调用任何 function 用的是同一套机制。',
  'ae-subagent-isolation': 'Claude Agent SDK 产出的 subagent 各自跑在自己的上下文窗口里（虚线边界）。orchestrator 的主上下文保持干净：子代理并行探索、只带回一份紧凑摘要。并行和上下文隔离一箭双雕——一场吵闹的搜索不会灌满父级 transcript。',
  'ae-swebench-gate': 'SWE-bench 靠跑 repo 的测试套件给 patch 打分，不是问模型「看起来对不对」。只有 FAIL_TO_PASS 测试现在通过、且 PASS_TO_PASS 测试仍然通过，任务才算 resolved。基于执行的评分是基准防作弊的原因，也是 SWE-bench Verified 要剔掉测试含糊或损坏的任务的原因。',
  'ae-agent-human-gap': '发布时 WebArena 和 OSWorld 显示差距悬殊：最好的 Agent 只有约 14%，人类在 78% 左右。蓝线逐年爬升，但两种失败模式没变。Agent 依然在 GUI grounding（点哪里）和操作知识（任务实际要什么）上失手——分数涨得比可靠性快。',
  'ae-genai-span-tree': 'OpenTelemetry 的 GenAI 约定给所有厂商一个 schema。invoke_agent span 是根；每个 execute_tool span 挂在它下面；每个 model chat span 挂在调用它的工具下面。名字和父子关系都是标准化的——同一条 trace 在 Datadog、Grafana、Jaeger、Honeycomb 里读法一致。',
  'ae-eval-three-layers': '评估不是最后那个勾选框；它是驱动每个选择的外循环。静态 benchmark 固定模型，自定义离线 eval 度量你的产品形态，在线生产 eval 在真实流量上抓回归。三个圆环绕着 build 不停转——所以 2026 年的实践是把 eval 放在代码旁边、放在 CI 里、把关每个 PR。',
};
