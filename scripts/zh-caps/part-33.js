/** zh translations — figures-llmeng (8 figures) */
module.exports = {
  'few-shot-curve': '往 prompt 里加有标签示例，准确率先快速上升然后走平。灰点是 zero-shot；头几个 demonstration 就关掉了大部分差距，超过五六个之后，每个新示例几乎不涨分、却照样花 token。功力在于挑出到达平台期所需的最小集合。',
  'cot-decomposition': '硬问题要求直接给答案，等于逼模型把所有步骤压进一次跳跃，它常常滑倒。Chain-of-thought 让它把中间步骤写出来，一次大跳变成一串各自容易的小跳。每个可见的子步骤还是模型自查自纠的机会。',
  'constrained-decoding': '结构化输出在解码时强制一条语法。每个位置上，schema 算出哪些下一 token 合法，其余所有 token 的 logit 在采样前被打到负无穷。模型仍在选择，但只能从允许集合里选——结果永远是可解析的 JSON，而不是碰巧长得像 JSON 的自由文本。',
  'prompt-cache-hit': '跨调用重复的长 system prompt 或文档可以预填充一次、反复复用。缓存前缀（虚线）跳过重算，只处理新的后缀（实线）。省的钱等于命中率乘前缀占比：大而共享、且频繁命中的前缀，才是 caching 最划算的地方。',
  'semantic-cache': 'semantic cache 在新 query 的 embedding 与缓存条目足够接近时，直接用存好的答案回复。每个圆点是一条进来的 query，按它与最近缓存条目的余弦相似度摆放；橙线右侧的阴影区从缓存出答案。阈值调高，命中变少但几乎不出错；调低，缓存更多但可能端出过期或不对题的旧答案（橙点）。',
  'function-call-args': 'function calling 是两份工作合在一起。模型先从可用集合里选出要调用的工具，再从用户请求里抽取值、填进该工具的类型化参数 schema。输出不是散文而是一结构化调用：工具名加一个 JSON 对象，字段类型与声明一致，拿来即执行。',
  'llm-judge-rubric': 'LLM judge 按多个标准给回答打分——这里是有用性、正确性、安全性。总分是加权平均，权重编码了你真正在乎什么。这份回答安全性高分、正确性低分，rubric 若向正确性倾斜，它的分数就被拉下来。决定「好」的含义的是权重，不是模型。',
  'lost-in-the-middle': '把唯一相关的事实埋进长上下文：它靠近开头或结尾，模型记得很好；落在中间，成绩差得多。准确率随位置呈 U 形。所以 context engineering 把最重要的材料放在 prompt 的顶端或底端，绝不依赖一长段无结构的倾倒。',
};
