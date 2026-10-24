/** zh translations — figures-multimodal (7 figures) */
module.exports = {
  'contrastive-matrix': 'CLIP 把 batch 里每张图与每条 caption 两两打分，形成相似度矩阵。对比损失把匹配的对角线对拉近、把非对角推远。除以一个小的温度，softmax 变锐、对角线亮起；温度大了矩阵摊平，模型不再分辨配对。',
  'cross-attention-fusion': 'vision-language 模型里，每个文本 token 是一个 query，对图像 patch 的 key 做注意。网格就是一张注意力图：行是文本 token，列是图像 patch，每行 softmax 归一。「bird」这类实词聚到包含那物体的 patch 上——语言就是这样 grounding 到像素的。',
  'modality-projection': '分开的图像 encoder 和文本 encoder 各落在自己的空间里，匹配的一对起初指向不同方向。一个学到的投影把两者映进同一个 d 维共享空间，训练把匹配的向量转在一起，直到余弦逼近一。对齐之后，一个距离就能跨模态比较。',
  'cfg-guidance-scale': 'Classifier-free guidance 把 diffusion 模型跑两遍——带 prompt 一遍、不带一遍——再沿两者的差外推。scale 为一是普通的条件预测；scale 调高更贴 prompt，拿多样性换服从。推过头样本就饱和崩坏，所以实用 scale 落在中间一段。',
  'vq-codebook': 'VQ-VAE 的 encoder 产出连续向量（蓝点），但模型需要离散 token。每个向量吸附到最近的 codebook 条目（橙方块），图像变成一串整数编码。codebook 更大，量化更细、重建误差更低——但每 token 花更多 bit，还冒着永远用不上的编码的风险。',
  'video-temporal-patches': '视频同时在空间和时间上 token 化。每个 224 像素的帧切成空间 patch 网格，帧沿时间组成 tubelet。token 总数是时间份数乘空间份数，注意力成本又随 token 数平方增长——长片配细 patch 就爆预算。更大的 patch 和 tubelet 是压住预算的主要杠杆。',
  'audio-text-ctc': '音频是一长串帧，转写却很短。CTC 让模型在每帧发一个标签或一个 blank，然后合并重复、删掉 blank 来折叠输出。对齐保持单调、时间只向前走；真正的双写字母不塌成一个，靠的正是 blank token。许多帧级对齐映射到同一段最终文本。',
};
