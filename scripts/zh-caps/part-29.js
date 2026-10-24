/** zh translations — figures-cv3 (8 figures) */
module.exports = {
  'cv3-roialign-sampling': 'RoIPool 两次把 proposal 框吸附到整数特征网格上，每次取整都让 mask 糊掉零点几个像素。RoIAlign 保持框的小数坐标：在每个输出 bin 里放固定数量的采样点，逐点对四个相邻格子做双线性插值取值。没有取整，mask 就落在物体真实所在的位置。',
  'cv3-latent-compression': '像素空间的 diffusion 每步要反传过 786,432 个数值。Stable Diffusion 训练一个 VAE，把 3×512×512 的图像压成 4×64×64 的 latent（小橙色网格），整条去噪循环都在 latent 里跑，最后再解码回像素。latent 上的注意力便宜 48 倍——开放权重 text-to-image 由此变得可行。',
  'cv3-ctc-collapse': 'CRNN 每个时间步发一个字符（或 blank ε），与输入没有固定对齐。CTC 定义了折叠规则：先合并连续相同标签，再删 blank。两个 l 连续段之间的 blank 正是「hello」不塌缩成「helo」的关键。训练时对所有能折叠成目标的对齐求和。',
  'cv3-pose-heatmap': '姿态模型不直接回归坐标。它为每个关键点输出一张 heatmap，训练目标是让高斯凸包正好盖住真实关节，argmax 像素即坐标。K 个峰全部浮现后，关节按固定的人体图（top-down）或 association field（bottom-up）连成骨架。',
  'cv3-gaussian-splat': '一个场景是数百万个 3D Gaussian，各自携带中心、旋转+缩放的协方差、不透明度和随视角变化的颜色。渲染把每个椭球投影成 2D 椭圆，按深度排序，从前往后 alpha 合成——这正是 NeRF 当年要几百次 MLP 查询才能逼近的混合。所以 splat 能以 100+ fps 渲染、几分钟训完。',
  'cv3-rectified-flow': 'DDPM 学的是从噪声到数据的弯曲轨迹，忠实积分它要几百个小步。rectified flow 训练模型直接走噪声样本与数据点之间的直线。直路需要的积分步数少得多——SD3 和 FLUX 因此 20 步采样（蒸馏后 1–4 步），而不是 1000 步。',
  'cv3-open-vocab': 'Promptable Concept Segmentation 收下一个简短名词短语，一遍跑完就返回所有匹配物体的 mask 和实例编号。灰色的苹果和盒子不选中；三个橙子被描边并编号。早先的系统要把文本 grounding 检测器串上另一个分割器，接缝处误差累积。SAM 3 把这条级联压成一个模型。',
  'cv3-track-assoc': 'tracking-by-detection 每帧跑一次检测器，然后判断哪个新框对应哪条旧轨迹。它按 IoU（常再叠加运动与外观）给每对（轨迹，检测框）打分，用匈牙利算法解指派，把匹配上的 id 带到下一帧。落单的检测框开新轨迹；落单的轨迹逐渐老化退出。产品要的是身份，不是框。',
};
