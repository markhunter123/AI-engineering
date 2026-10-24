/** zh translations — figures-nlp3 (9 figures) */
module.exports = {
  'pos-tagger': '每个 token 拿到一个语法类别。tagger 从左到右扫过句子，给每个词贴上标签：determiner、noun、auxiliary、verb、adverb。lemmatizer 和 dependency parser 接下来读的就是这些标签。',
  'dependency-arcs': 'dependency parsing 从每个词向它的 head 画一条带标签的弧。动词「running」是 root；「cats」是它的主语、「were」是它的助动词，介词短语挂在动词上。每条边都是一个（head, dependent, relation）三元组。',
  'qa-span': '抽取式 QA 从不发明答案。两个 head 分别预测给定段落中答案 span 的起止 token；模型扫过文本，点亮两者之间的连续区段。答案是逐字抬出来的——所以它没法幻觉。',
  'summarize-collapse': '抽取式摘要给每句打分、按顺序抬出最核心的几句，长文档塌缩成几行原文。生成式摘要则改写：读起来流畅，但可能捏造源文本从未说过的内容。',
  'topic-drift': '无监督主题建模不需要标签。散乱的词汇按共现或 embedding 相似度聚成几组连贯的簇；每个簇读它的头部词，就念出语料「在谈什么」的潜在主题。',
  'coref-links': '三种表面形式——「He」「the CEO」「Cook」——都指向同一个人。共指消解从每个 mention 画一条链回到同一个簇，下游的 NER、QA 和知识图谱因此把它们当作一个实体。',
  'nli-router': 'NLI 拿一个前提 t 和一个假设 h，把这一对恰好路由到三个标签之一：entailment、neutral、contradiction。同一个分类器支撑着幻觉检查、grounded-QA 验证和 zero-shot 标注。',
  'relation-triples': '关系抽取先把实体 span 锚进文本，再把它们连成带类型的边。「Tim Cook」和「Apple」成为节点，动词短语成为两者之间带标签的边。三元组聚合起来，就是知识图谱。',
  'constrained-decoder': 'logit processor 坐在模型和采样器之间。每一步它问语法哪些 token 合法，把每个非法 logit 设成负无穷。softmax 于是只把概率质量放在合法延续上——输出在构造上就结构正确。',
};
