/** zh translations — figures-ml (10 figures) */
module.exports = {
  'linear-regression-fit': '每根橙色杠杆是一个 residual——点到直线的差距。平方求平均就是 mean squared error。最小二乘无非是挑出让这个平均最小的斜率和截距。',
  'logistic-sigmoid': 'logistic regression 把线性分数过一遍 sigmoid 压成概率。权重控制曲线转折多陡；偏置把它左右平移。橙线是决策边界——概率穿过一半的位置。',
  'svm-margin': 'support vector machine 不只是分开两类，它把边界推到离两边都尽可能远。虚线标出 margin；贴着虚线、金色描圈的点是 support vector。移动其他任何点，什么都不会变。',
  'knn-smoothness': 'k = 1 时预测完全照抄最近的点，曲线锯齿丛生、迁就每一个怪癖。k 调大，预测在更多邻居上取平均，边界随之平滑；k 极大时曲线摊平到整体类别比例，彻底无视局部结构。',
  'kmeans-step': 'K-means 在两步间交替：把每个点分给最近的 centroid，再把每个 centroid 挪到自己成员的均值。簇内平方和只会下降，所以一旦没有点再换簇，算法就收敛了。',
  'decision-tree-depth': '深度每加一层，叶子翻倍、树能刻出的区域翻倍再翻倍。前几刀抓住真实结构；再往下叶子变金——树在逐点硬背，而不是学模式。',
  'feature-scaling': '特征尺度悬殊时，损失面是一道狭长峡谷，gradient descent 在两壁间弹跳。把每个特征标准化到零均值单位方差，碗变圆了——同一个算法几乎直线走到最小值。',
  'naive-bayes': '每个类别在特征上有一条自己的钟形曲线。橙线是你观测到的值；圆点是每类产出它的可能性。Bayes 把这些 likelihood 按先验加权，变成后验长条——点属于 B 类的概率。',
  'class-imbalance': '某类很稀有时，永远预测多数类的分类器准确率很高，却抓不到一个你在乎的案例。金色扇区——它永远找不到的正例——对 accuracy 完全隐形。不平衡数据上要看的是 recall 和 F1。',
  'k-fold-cv': '每行是一轮：金色折留作验证，蓝色折用于训练。轮换留出折，每个样本恰好被打分一次；k 个分数取平均，比单次切分稳定得多。',
};
