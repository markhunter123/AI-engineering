/** zh translations — figures-capstone-a (8 figures) */
module.exports = {
  'cap-bpe-merge': 'Byte-Pair Encoding 从原始字节出发，反复找出最高频的相邻对，合并成一个新符号，词表增加一个 id。扫描框扫过每一对邻居；胜出的一对塌缩成单个 token。',
  'cap-sliding-window': 'tokenizer 把语料摊平成一条长长的 id 流。一个定长 T 的窗口按步长沿流滑动，目标就是同一窗口左移一位——用于 next-token prediction。步长越小，重叠越多，样本数被灌水。',
  'cap-multihead-attention': '单个线性层把输入投影成 Q、K、V，再重塑成 H 个大小 D/H 的并行头。每个头在 causal mask 下独立跑 scaled dot-product attention，输出拼接回 D。头在训练中各自特化。',
  'cap-training-loop': '每一步拉一个 batch，forward 出 logits，算 cross-entropy loss，backward，再按 LR schedule 走一个 AdamW 步。重复几千次，损失曲线下降并走平。周期性的 eval 和样本探针能抓到标量损失藏住的发散。',
  'cap-classifier-head-swap': '预训练模型是冻结的躯干加一个头。要做分类，保留昂贵的躯干，把词表头卸下来，粘上一个输出两个 logit 的小线性层。领域偏移不大时只训头求快；要精度就把躯干解冻。',
  'cap-dpo-preference': 'DPO 直接在「chosen 优于 rejected」的对子上训练 policy，不需要单独的奖励模型。损失是相对冻结 reference 的对数概率差的 sigmoid：chosen 回复被推高、rejected 被压低，KL 锚点让 policy 保持本分。',
  'cap-corpus-downloader': 'shard 从远端流式传来，经 Zstandard 解压、不缓冲整个文件。已校验的字节偏移让断线可以用 HTTP Range 请求续传；每篇文档算一个 MinHash 签名、按 LSH 分桶，近似重复在进入 manifest 之前就相撞而落。',
  'cap-cosine-warmup': 'schedule 先在 warmup 步内把学习率从零线性爬到 lr_max，护住脆弱的最初几步更新，然后沿 cosine 曲线的上半段落回 lr_min。标记骑在优化器每一步真正使用的那精确数值上。',
};
