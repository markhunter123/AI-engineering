/** zh translations — figures-nlp2 (8 figures) */
module.exports = {
  'bow-tfidf': 'bag of words 数原始词频，「the」于是处处显得重要。TF-IDF 给每个计数乘上 log(N 除以文档频率)：出现在所有文档里的词 idf 为零、权重塌掉；只出现在一篇文档里的词 idf 高、权重上浮。稀少而能区分的词胜出。',
  'rnn-unroll': '循环网络是同一个 cell 应用在每个时间步、权重共享。沿时间展开它成了一条链：每个 cell 用 tanh 把新输入折进上一个隐藏状态，再把结果传给下一步。最终状态见过整个序列——这也正是长序列难把 gradient 传回去的原因。',
  'lstm-gates': 'LSTM 的 cell state 是一段由门编辑的记忆。forget gate 擦掉旧状态的一部分，input gate 写入新候选的一部分，两者相加成为下一个 cell state；output gate 再控制多少状态泄漏成隐藏向量。门取值接近零或一时，cell 能把一个值持有许多步而 gradient 不消失。',
  'seq2seq-alignment': 'attention 给 decoder 一层对源文本的软对齐。每个目标 token 读的是所有 encoder 状态的加权和，权重沿源行 softmax 归一。这里「maison」注意「house」、「rouge」注意「red」，偏离对角线的格子暴露出翻译需要的重排。权重尖锐则锁定一个源词，扁平则混合几个。',
  'edit-distance': 'Levenshtein distance 填一张表，每个格子是「用 insert、delete、substitute 把一个前缀变成另一个」的最省路径。每个格子取三个邻居的最小值，右下角就是整串的距离。回溯选择就还原出实际编辑路径——拼写检查器和翻译指标就是这样对齐文本的。',
  'ngram-backoff': 'n-gram 模型用前 n−1 个词预测下一个词。n 调大，上下文更丰富，但可能的 gram 数量是词表的 n 次方，语料实际观测到的比例塌向零。大多数长上下文从未被见过——所以高阶模型必须对没见过的 gram 做平滑，退回到更短、更稠密的低阶 gram。',
  'ner-bio-tagging': '命名实体识别被表述成逐 token 打标签。BIO 方案给实体的第一个 token 标 B-TYPE，所有后续 token 标 I-TYPE，其余 token 标 O。序列标注器因此能表达多词实体和精确边界：「New York」是 B-LOC 接 I-LOC，区别于两个独立的单词地名。',
  'sentiment-logits': '线性文本分类器给每个词记一个学到的权重，连同偏置求和成一个 logit，再过 sigmoid 压成概率。正权重推向正面情感，负权重反向拉扯。决策在概率一半处翻转——求和的 logit 正好在零上穿过。',
};
