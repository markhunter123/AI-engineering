/** zh translations — figures-mlprod (3 figures) */
module.exports = {
  'mlprod-serving-cost': 'batch scoring 每轮给每个用户都算一次分，成本随新鲜度上限缩短而增长。online serving 为峰值流量规模的副本付费，成本跟随请求数、与新鲜度无关。低于一小时，一小时一次的 batch 任务根本达不到时限。价格是为了练习取的整数。',
  'mlprod-drift-psi': 'reference 窗口切成十个各占 10% 的桶。挪动当前窗口，各桶占比随之变化。PSI 把这些变化按对数比加权求和。一个桶从 10% 涨到 30%，贡献远大于涨到 12% 的桶。0.1 和 0.25 这两道线是信贷评分的经验法则，不是统计检验。',
  'mlprod-pit-join': '标签日在 day t 的训练行，只能用 day t 当天就存在的特征值。point-in-time join 取标签日当天或之前最新的一行，超过 TTL 就丢弃。按 user id 的朴素 join 取的是表里最新的行——那行常来自标签日之后，把未来抄进了训练集。',
};
