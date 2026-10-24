/** zh translations — figures-capstone-e (10 figures) */
module.exports = {
  'ce-agent-loop': 'harness 跑的是 plan、act、observe、recover 循环，每转一圈烧掉 50 轮预算中的一轮。每次工具调用跨越边界时，PreToolUse 与 PostToolUse hook 相继触发；过大的观察结果——比如 8MB 的 ripgrep 输出——会在重新进入上下文窗口前被截断。',
  'ce-hybrid-retrieval': '查询同时点燃两路检索：embedding 上的稠密最近邻，加上词项上的稀疏 BM25。两者并集交给 cross-encoder 重排，单靠 BM25 会被埋没的那个分块浮上顶部。synthesizer 必须引用存活下来的文件与行号。',
  'ce-voice-latency': '每一段还没跑完就把输出流给下一段：用户还在说话，ASR 就在发 partial 结果；turn detector 一听到收尾线索就触发；TTS 在 800ms 首音频预算内启动。用户在 900ms 时插话，TTS 立刻取消，ASR 重新接管。',
  'ce-late-interaction': '不要 OCR，也不要池化向量：页面保持为图像，嵌入成数千个 patch 向量。每个查询 token 对每个 patch 打分，只保留最大值，各 token 最大值相加就是页面得分。表格、图表、手写体都在 patch 层面直接匹配。',
  'ce-experiment-tree': '每个节点是一次沙箱实验，跑完即打分。搜索扩展得分最高的节点，而不是最新的——0.71 分支赢得子节点，兄弟节点原地踏步。预算表随每次运行往下掉，耗尽时，最好的分支被写成论文。',
  'ce-rootcause-walk': 'OOMKilled 告警落在 pod 上。Agent 沿 ownership 边往上走到 deployment，沿调度边往外走到 node，每一跳拉取一段遥测切片。假设按证据权重排序，胜出者仍要等人工审批放行，任何修复才会执行。',
  'ce-finetune-stages': '同一份工件走完整条 pipeline。干净数据喂给 SFT，把任务行为灌进 base model；DPO 把对齐的色调加深；quantization 把模型压成 4-bit 而不改变它学到的东西；served 端点用 speculative decoding 流式吐 token。',
  'ce-migration-funnel': '大多数文件直接穿过确定性 recipe，落进 CI 全绿的分支。含糊的少数派改道进入 agent loop：构建、给失败分类、应用修复、在硬性上限下重跑。仍然失败的，带着一份示例 diff 归档进分类学。',
  'ce-team-handoff': 'architect 把计划冻结到任务板上，四个 coder 在各自的并行 worktree 里消化，各自按自己的节奏收工。合并是交接失败的地方：两个 coder 改了同一个文件，冲突闪现——直到这时，diff 才通过 reviewer 闸门和隔离的测试运行。',
  'ce-otel-drift': '四套 SDK 家族的 GenAI-semconv span 流经 collector 汇入 ClickHouse，eval 任务在那里给抽样 trace 打分。注入的回归一落地，faithfulness 迷你曲线跌破告警底线，页面告警响起——整个过程仍在五分钟检测预算之内。',
};
