/** zh translations — figures-cv2 (8 figures) */
module.exports = {
  'object-detection-nms': 'YOLO head 在每个网格单元都预测一个框，一个物体于是催生许多重叠候选（虚线）。Non-maximum suppression 按置信度排序，留下最强的，删掉与它重叠过多的每个框。活下来的是每个物体一个紧框。',
  'segmentation-flood': '分割是逐像素的分类。encoder 压缩图像以提供上下文，decoder 上采样回全分辨率，skip connection 把精细的空间细节带过去、边界保持锋利。看每个像素认领自己的类别、mask 向外漫延。',
  'cv-gan-image': 'generator 把噪声向量变成图像；critic 打分它看起来多真。两者互相对抗训练：generator 把 critic 的 P(real) 推上去，critic 把它压下来。跷跷板趋向平衡，假图像从噪声中磨出结构。',
  'cv-diffusion-image': 'diffusion 模型学的是一次去掉一点噪声。采样从一片纯高斯噪声的网格出发往回走：每步预测噪声并减掉它，结构逐渐浮现。小的去噪步重复足够多次，一张连贯图像就从雪花里凝结出来。',
  'nerf-rays': 'NeRF 把场景存成一个函数：给它一个 3D 点和视角方向，它返回密度和颜色。渲染一个像素时，从相机投出一条光线，沿它采样点，逐点查询 MLP，再从前往后按密度加权积分颜色。每条光线把空间中的一条线压成一个像素。',
  'clip-contrastive': 'CLIP 把图像和 caption 嵌进同一个共享空间。对 N 对样本，它建一个 NxN 相似度矩阵并训练成：对角线（真配对）得分高，每个非对角得分低。看对角线随匹配对被拉近、错配被推远而变亮。',
  'metric-embedding': 'retrieval 按 embedding 空间中的距离给候选排序。metric learning 塑造那个空间：triplet 或 contrastive loss 把同类点拉近、把其他类推开。簇一旦收紧，query 按余弦距离的最近邻就是可靠的正确答案。',
  'depth-sweep': '单目深度模型把一帧 RGB 映成每个像素的距离。冻结的 ViT encoder 读出透视、纹理和学到的场景先验；轻量 decoder 把它们上采样成稠密深度图。近处表面发亮、远处退暗——全凭一张图，不用立体相机或深度传感器。',
};
