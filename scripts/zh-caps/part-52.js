/** zh translations — figures-llms-systems (8 of 9; mcp-tool-call won by figures-mcp) */
module.exports = {
  'beam-search': '每一步把每个幸存 beam 展开成 B 个候选延续，按累计 log 概率打分，只留最好的 B 个。Greedy decoding 是 B = 1 的特例：一条路、不回头。beam 更宽探索更多，算力也按比例更贵。',
  'speculative-decoding': '小的 draft 模型便宜地提出 γ 个 token；大的目标模型一次并行前向全部验证，接受最长的正确前缀，然后自己再吐一个 token。一致率越高，接受的连续段越长，每轮取代的目标模型调用越多。',
  'moe-routing': 'router 把每个 token 送到 E 个 expert 里它的 top-k 个。每 token 只跑 k/E 的 expert block——尽管所有参数都驻留在内存里，活跃算力只是那个分数。路由不均会让少数 expert 过载，所以 MoE 训练要加 load-balancing 损失。',
  'context-window-slide': '模型只能在固定窗口内注意。序列长过窗口，最老的 token 就滑出橙色框、从注意力里消失。这就是滚动上下文：新 token 留着，老 token 被遗忘——除非被摘要或检索回来。',
  'perplexity-loss': 'Perplexity 是 cross-entropy 损失的指数，读作模型每 token 在多少个等可能选项之间抉择。一个在大小 V 的词表上均匀乱猜的模型，perplexity 恰好是 V（橙线），所以任何有用的模型都必须远远落在这条线以下。',
  'continuous-batching': '同 batch 的请求在不同时刻完成，因为它们生成的 token 数不同。静态 batching 把每个槽位扣到最长的请求结束，留下灰色空转。continuous batching 在槽位空出的那一刻就从队列补位——GPU 保持满载，利用率上升。',
  'image-patch-tokens': 'Vision Transformer 把图像切成固定大小的 patch 网格，每个 patch 当作一个 token——和文本里的词完全一样。token 数是（尺寸÷patch）的平方，所以 patch 尺寸减半，序列长度和注意力成本翻四倍。',
  'multimodal-fusion': '图像 encoder 和文本 encoder 各自把输入映成向量。late fusion 把两者投影进共享 embedding 空间、只在最后比较——CLIP 这类对比模型就是这样对齐图像和 caption 的。early fusion 把两条 token 流拼接、从一开始就联合建模，两种模态全程互相注意。',
};
