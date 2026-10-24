/** zh translations — figures-capstone-b (8 figures) */
module.exports = {
  'grad-clip-monitor': '每一步都测全局 gradient 范数并裁到上限。一个溢出的 batch（橙）会冲破上限、把损失打成 NaN；GradScaler 检测到 Inf，跳过这一步优化器更新，并把 scale 减半——运行因此存活。',
  'rrf-fusion': '一个 query 同时跑 BM25 和稠密 retriever。各返回一份排序列表；RRF 用 1/(k+rank) 跨列表求和给每份文档打分，任何一路排得好的文档都会上浮。融合靠投票而不是插值——所以它在各种 query 类型上都赢。',
  'rerank-funnel': '便宜的 retriever 返回 N 个候选；cross-encoder 带着完整注意力逐个读（query, document）对并打分。只有 top-K 存活（蓝）。贵的模型只跑 N 次而不是全语料次，精度提升正好落在请求延迟预算之内。',
  'rag-pipeline-flow': '数据包带着一个 query 走过装配好的 pipeline：rewrite 扩写它，hybrid index 检索，cross-encoder 重排，generator 带着 chunk 锚定的引用作答，置信度低就拒绝。六个孤立组件合成一个系统——整体胜过单独测量的任何一级。',
  'sandbox-runner': '生成的代码经 stdin 送进子进程里一个全新解释器，配 import denylist 和输出上限。每条提供的断言跑一遍，点亮 pass（蓝）或 fail（橙）；wall-clock 超时杀掉失控循环。得分是断言通过的比例——崩溃和超时都是一等公民的失败模式。',
  'eval-grid': 'runner 读任务 spec，把任务撒进 worker pool，用 metric 层加校准报告逐一打分。扫过之处格子点亮 pass（蓝）或 fail（橙）。每个模型的 EvalRun 记录直接流进 leaderboard aggregator；demo 在一次干净的运行后自行终止。',
  'injection-gate': 'prompt 流过三层可审计的防线：normalize 解掉 base64/rot13/零宽字符把戏，substring 规则抓手写短语，regex 规则抓整个家族。良性 prompt（蓝）干净通过；攻击（橙）绊响一条规则，aggregator 发出的是置信度和类别，而不是一种感觉。',
  'safety-checkpoints': '一个请求穿过三个 checkpoint。pre-gen 阶段检测器可以直接拦下；during-gen 阶段 token 过滤器见到违禁词就掐断流；post-gen 阶段分类器和规则引擎检查完整输出。闸门汇总各裁决、执行最终动作，并画出一条每个审查者都读得懂的审计轨迹。',
};
