/** zh translations — figures-capstone-h (10 figures) */
module.exports = {
  'ch-paper-skeleton': '骨架在任何散文存在之前，就把章节、figure 槽位和参考文献键声明为数据。harness 先验证这份契约：每个被引用的 figure 有槽位、每个 citation 有条目、每个章节都在目录里。然后生成器才往槽位里写正文——结构性债务因此永不累积。',
  'ch-critic-converge': '每一轮 critic 返回一个五维分数向量，修订以结构化 diff 应用，轨迹朝着目标区间爬升。收敛是被工程出来的，不是被盼望出来的：循环在平台期、达标或预算耗尽时停止；由于分数是向量而非一段话，任何一维出现倒退都会被抓出来。',
  'ch-ucb-scheduler': '队列里放着按上置信界打分的假设。槽位一空，scheduler 就派出 UCB 最高的分支；完成的实验把结果洒上结果总线、回流更新统计——一个实验的发现会重排它后面的一切。置信界坍缩的分支被剪掉，不再浪费槽位。',
  'ch-research-pipeline': 'demo 把一次运行串过前面 Track D 各课搭建的每个阶段：假设播种、UCB scheduler、实验 runner、critic loop、论文 writer。每次交接都过一个契约闸门，在下一阶段运行前验证形状；循环自动终止成一份 demo 报告。任何契约漏了，被抓住的就是这一课。',
  'ch-patch-tokenizer': '把每个像素都当 token，会得到一条 150528 token 的序列——没有哪个 12 层 transformer 吃得下。把图像切成 16x16 方块得到 196 个 patch；每个展平成 768 个值，经一层线性投影进隐藏维度；一个 2D 正弦信号恢复每个方块原来的位置。transformer 看到的是一条能嚼动的短序列。',
  'ch-cls-funnel': 'patch token 进入时彼此毫无感知。十二个 pre-LN 多头自注意力 block 让每个 patch 与所有 patch 交换信息，CLS token 逐层积累整图特征；它最终的隐藏状态就是全栈其余部分读取的池化摘要。这套 block 配方是 CLIP、SigLIP 和 DINOv2 的脊椎。',
  'ch-projection-bridge': '视觉编码器和文本表住在互不相干的基里。一个两层 MLP（约 130 万参数）把池化后的图像向量运进文本 embedding 空间；对着配对 caption 的 cosine 对齐损失，把投影出的点拉向它的搭档。编码器与文本表全程冻结；桥是唯一学习的部件。',
  'ch-crossattn-fan': '后期融合里，decoder 在纯文本 token 上运行，每一层都伸进图像流：被高亮的 token 向每个记忆 token 发出 query（无掩码），加权答案流回它的残差流。底部一行的自注意力保持因果。图像侧只编码一次、每个 decode 步复用，所以长 caption 依然便宜。',
  'ch-infonce-diagonal': '对 N 个图像—caption 对，相似度矩阵在对角线上有 N 个匹配格，格外有 N²−N 个不匹配格；InfoNCE 对每行每列跑交叉熵，对角线亮起、其余变暗。LM 损失用同一组权重学着给每张图写 caption；50 步的 demo loop 里两条曲线一起下落。',
  'ch-recall-window': '三个界面度量训练损失度量不了的东西。检索按 cosine 对每个图像相对查询 caption 排序，问命中是否落在前 1、5 或 10——这里爬到第 2 名，R@1 未中但 R@5 命中。VQA 按精确答案匹配每样本记一个比特；captioning 对生成的 token 取带 brevity penalty 的 BLEU-4。',
};
