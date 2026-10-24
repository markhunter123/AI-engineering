/** zh translations — figures-llms3 (8 figures) */
module.exports = {
  'expert-routing': 'dense block 给每个 token 都跑同一个 FFN。MoE block 换成许多 expert FFN 加一个 router，每个 token 只送到其中 top-k 个。总参数随 expert 数量增长，每 token 的算力却只随 k 增长——模型变大而不变慢。',
  'encoder-decoder': 'encoder 把整段源文本读一遍，冻结成一个稠密表示。decoder 再逐 token 生成输出，每一层都 cross-attention 进那个表示。翻译、摘要、转写——都是同一种「输入到输出」的形状。',
  'rnn-vs-parallel': 'RNN 拖着一个隐藏状态逐步走过序列，1000 个 token 的输入就是 1000 个串行步骤。transformer 让每个位置在一次并行矩阵乘法里注意所有位置。就是「丢掉循环」这一个决定，让曲线在 2017 年之后继续扩张。',
  'draft-verify-tokens': '便宜的 drafter 用几次小前向快速提出多个 token。大模型随后一次前向验证全部：接受与自己分布一致的最长前缀，拒绝其余。草稿质量好时，一次大步的价钱换来多个 token——输出分布分毫不变。',
  'multi-token-predict': '标准训练只监督每个隐藏状态预测唯一的一个下一 token——信号很弱。MTP 加了几个更深的头，各预测更远处的一个 token，链式连接以保住因果顺序。更丰富的信号磨利了主干，训练好的头在 inference 时还能兼任 speculative drafter。',
  'self-critique-loop': '不必花钱请人打偏好标签，模型给自己评分。它起草一份回答，对照成文的原则 constitution 批评这份草稿，然后改写。修订后的对子成为训练信号——对齐工作的大头就这样在模型内部循环完成。',
  'loss-masking': '一条对话样本包含 system、user、assistant 轮次，但模型只应该被训练去产出 assistant 轮次。loss masking 把所有 prompt token 的梯度清零——模型学会的是回答，而不是背下并复述问题。',
  'activation-recompute': 'backward 需要 forward 的 activation，但全留着会撑爆内存预算。checkpointing 只保存少数几层（实线），其余丢弃。backward 时，每段被丢的计算从最近的 checkpoint 重新计算——多花一点算力，换峰值内存大幅下降。',
};
