# 目标检测——从零实现 YOLO

> 检测是分类加回归，在特征图的每个位置运行，然后通过非极大值抑制（NMS）清理结果。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 4 Lesson 04 (Image Classification), Phase 4 Lesson 05 (Transfer Learning)
**Time:** ~75 minutes

## 学习目标

- 解释网格与锚点设计如何把检测变成密集预测问题，并说明输出张量中每个数字的含义
- 计算边界框之间的交并比（IoU），并从零实现非极大值抑制
- 在预训练 backbone 上搭建一个最简 YOLO 风格的检测头，包括分类、目标置信度和边界框回归损失
- 读懂检测指标行（precision@0.5、recall、mAP@0.5、mAP@0.5:0.95）并决定下一步该调哪个旋钮

## 问题

分类说「这张图是狗」。检测说「像素 (112, 40, 280, 210) 处有一只狗，像素 (400, 180, 560, 310) 处有一只猫，画面其余位置没有东西」。这一个结构性变化——从每张图一个标签变为预测可变数量的带标签边界框——正是每个自动驾驶系统、每个监控产品、每个文档布局解析器和每条工厂视觉产线所依赖的能力。

检测也是让视觉领域所有工程 trade-off 同时出现的地方。你需要边界框准确（回归头）、每个框的类别正确（分类头）、模型知道什么时候没有目标可检（目标置信度分数），以及每个真实目标只对应一个预测（非极大值抑制）。漏掉任何一项，管线就会要么漏检、要么产生幻觉框、要么把同一个目标在稍微不同的位置预测十五次。

YOLO（You Only Look Once，Redmon 等，2016）通过单次卷积网络前向传播实现实时检测的设计，让上述所有步骤都能跑起来；而同样的结构性决策至今仍是现代检测器（YOLOv8、YOLOv9、YOLO-NAS、RT-DETR）的骨干。掌握核心原理后，每个变体都只是同一套组件的重新排列。

## 概念

### 检测即密集预测

分类器每张图输出 C 个数。YOLO 风格的检测器每张图输出 `(S x S x (5 + C))` 个数，其中 S 是空间网格尺寸。

```mermaid
flowchart LR
    IMG["Input 416x416 RGB"] --> BB["Backbone<br/>(ResNet, DarkNet, ...)"]
    BB --> FM["Feature map<br/>(C_feat, 13, 13)"]
    FM --> HEAD["Detection head<br/>(1x1 convs)"]
    HEAD --> OUT["Output tensor<br/>(13, 13, B * (5 + C))"]
    OUT --> DEC["Decode<br/>(grid + sigmoid + exp)"]
    DEC --> NMS["Non-max suppression"]
    NMS --> RESULT["Final boxes"]

    style IMG fill:#dbeafe,stroke:#2563eb
    style HEAD fill:#fef3c7,stroke:#d97706
    style NMS fill:#fecaca,stroke:#dc2626
    style RESULT fill:#dcfce7,stroke:#16a34a
```

每个 `S * S` 网格单元预测 B 个框。每个框包含：

- 4 个描述几何的数：`tx, ty, tw, th`。
- 1 个目标置信度分数：「该单元格中心是否有目标？」
- C 个类别概率。

每个单元格总数：`B * (5 + C)`。对于 VOC 数据集，`S=13, B=2, C=20`，即每个单元格 50 个数。

### 为什么需要网格和锚点

直接回归会为每个目标预测绝对坐标的 `(x, y, w, h)`。这对卷积网络来说很困难，因为平移图像不应把所有预测都平移同样多——每个目标都有空间锚定。网格通过把每个真实框分配给其中心落入的单元格来解决这个问题；只有该单元格负责这个目标。

锚点解决第二个问题。一个 3x3 卷积很难从 16 像素感受野的特征单元回归出一个 500 像素宽的框。因此，我们为每个单元格预定义 B 个先验框形状（锚点），并预测每个锚点的小幅偏移。模型学会选择正确的锚点并微调它，而不是从零开始回归。

```
Anchor box priors (example for 416x416 input):

  small:   (30,  60)
  medium:  (75,  170)
  large:   (200, 380)

At each grid cell, every anchor emits (tx, ty, tw, th, obj, c_1, ..., c_C).
```

现代检测器通常使用 FPN，在不同分辨率上使用不同的锚点集合——小锚点放在浅层高分辨率图上，大锚点放在深层低分辨率图上。思想相同，只是尺度更多。

### 解码预测

原始的 `tx, ty, tw, th` 并不是框坐标，而是需要在绘图前进行变换的回归目标：

```math
\begin{aligned}
\text{centre } x &= (\sigma(t_x) + \text{cell}_x) \cdot \text{stride} \\
\text{centre } y &= (\sigma(t_y) + \text{cell}_y) \cdot \text{stride} \\
\text{width} &= \text{anchor}_w \cdot \exp(t_w) \\
\text{height} &= \text{anchor}_h \cdot \exp(t_h)
\end{aligned}
```

`sigmoid` 把中心偏移限制在单元格内部。`exp` 让宽度可以从锚点自由缩放而不改变符号。`stride` 把网格坐标缩放回像素坐标。这个解码步骤自 YOLOv2 以来在每个 YOLO 版本中都相同。

### IoU

检测中两个框之间的通用相似度度量：

```math
\mathrm{IoU}(A, B) = \frac{\mathop{\mathrm{area}}(A \cap B)}{\mathop{\mathrm{area}}(A \cup B)}
```

$\mathrm{IoU} = 1$ 表示完全重合；$\mathrm{IoU} = 0$ 表示没有重叠。预测框与真实框之间的 IoU 决定是否把预测计为真正例（通常 $\mathrm{IoU} \ge 0.5$）。两个预测框之间的 IoU 则供 NMS 用于去重。

### 非极大值抑制

在相邻锚点上训练的卷积网络经常会对同一个目标预测出重叠框。NMS 保留置信度最高的预测，并删除与被选框 IoU 超过阈值的任何其他预测。

```
NMS(boxes, scores, iou_threshold):
    sort boxes by score descending
    keep = []
    while boxes not empty:
        pick the top-scoring box, add to keep
        remove every box with IoU > iou_threshold to the picked box
    return keep
```

典型阈值：目标检测取 0.45。近期检测器会用 soft-NMS、DIoU-NMS 或直接学习抑制（RT-DETR），但结构上的目的相同。

### 损失函数

YOLO 损失是带权重的三项损失之和：

```math
\begin{aligned}
L = {} & \lambda_{\text{coord}} \cdot L_{\text{box}}(\text{pred}, \text{target}, \text{where obj}=1) \\
+ {} & \lambda_{\text{obj}} \cdot L_{\text{obj}}(\text{pred}, 1, \text{where obj}=1) \\
+ {} & \lambda_{\text{noobj}} \cdot L_{\text{obj}}(\text{pred}, 0, \text{where obj}=0) \\
+ {} & \lambda_{\text{cls}} \cdot L_{\text{cls}}(\text{pred}, \text{target}, \text{where obj}=1)
\end{aligned}
```

只有包含目标的单元格才对边界框回归和分类损失有贡献。不含目标的单元格只对目标置信度损失有贡献（教模型保持沉默）。`lambda_noobj` 通常较小（约 0.5），因为绝大多数单元格都是空的，否则会主导总损失。

现代变体把 MSE 框损失换成 CIoU / DIoU（直接优化 IoU），对类别不平衡使用 focal loss，并用 quality focal loss 平衡目标置信度。但三项损失的结构没有改变。

### 检测指标

准确率不适用于检测。四个有效的数字是：

- **Precision@IoU=0.5** —— 在被计为正例的预测中，有多少真正正确。
- **Recall@IoU=0.5** —— 在真实目标中，有多少被找到了。
- **AP@0.5** —— IoU 阈值 0.5 时 precision-recall 曲线下的面积；每类一个数。
- **mAP@0.5:0.95** —— 在 IoU 阈值 0.5、0.55、...、0.95 上 AP 的平均值。COCO 标准指标；最严格也最具信息量。

四个都要报告。mAP@0.5 强但 mAP@0.5:0.95 弱的检测器大致定位到了目标但框不够紧；用更好的边界框回归损失来修复。精确率高但召回率低的检测器过于保守；降低置信度阈值或增加目标置信度权重。

```figure
object-detection-nms
```

## 构建它

### Step 1: IoU

本节课的通用工具函数。处理两个 `(x1, y1, x2, y2)` 格式的框数组。

```python
import numpy as np

def box_iou(boxes_a, boxes_b):
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-8, None)
```

返回 `(N_a, N_b)` 的成对 IoU 矩阵。要针对单个真实框使用，可把其中一个数组做成 `(1, 4)` 形状。

### Step 2: 非极大值抑制

```python
def nms(boxes, scores, iou_threshold=0.45):
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[[i]], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return np.array(keep, dtype=np.int64)
```

确定性算法，复杂度 `O(N log N)` 来自排序，在相同输入下与 `torchvision.ops.nms` 行为一致。

### Step 3: 边界框编码与解码

在像素坐标和网络实际回归的 `(tx, ty, tw, th)` 目标之间转换。

```python
def encode(box_xyxy, cell_x, cell_y, stride, anchor_wh):
    x1, y1, x2, y2 = box_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    tx = cx / stride - cell_x
    ty = cy / stride - cell_y
    tw = np.log(w / anchor_wh[0] + 1e-8)
    th = np.log(h / anchor_wh[1] + 1e-8)
    return np.array([tx, ty, tw, th])


def decode(tx_ty_tw_th, cell_x, cell_y, stride, anchor_wh):
    tx, ty, tw, th = tx_ty_tw_th
    cx = (sigmoid(tx) + cell_x) * stride
    cy = (sigmoid(ty) + cell_y) * stride
    w = anchor_wh[0] * np.exp(tw)
    h = anchor_wh[1] * np.exp(th)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
```

测试：先编码一个框再解码——结果应该非常接近原框（误差来自 sigmoid 反函数在 `tx` 不在 post-sigmoid 范围内时不完全可逆）。

### Step 4: 最简 YOLO 检测头

在特征图上做一次 1x1 卷积，再 reshape 成 `(B, S, S, num_anchors, 5 + C)`。

```python
import torch
import torch.nn as nn

class YOLOHead(nn.Module):
    def __init__(self, in_c, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_c, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        n, _, h, w = x.shape
        y = self.conv(x)
        y = y.view(n, self.num_anchors, 5 + self.num_classes, h, w)
        y = y.permute(0, 3, 4, 1, 2).contiguous()
        return y
```

输出形状：`(N, H, W, num_anchors, 5 + C)`。最后一个维度是 `[tx, ty, tw, th, obj, cls_0, ..., cls_{C-1}]`。

### Step 5: 真实框分配

对每个真实框，决定哪个 `(cell, anchor)` 负责它。

```python
def assign_targets(boxes_xyxy, classes, anchors, stride, grid_size, num_classes):
    num_anchors = len(anchors)
    target = np.zeros((grid_size, grid_size, num_anchors, 5 + num_classes), dtype=np.float32)
    has_obj = np.zeros((grid_size, grid_size, num_anchors), dtype=bool)

    for box, cls in zip(boxes_xyxy, classes):
        x1, y1, x2, y2 = box
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        gx, gy = int(cx / stride), int(cy / stride)
        bw, bh = x2 - x1, y2 - y1

        ious = np.array([
            (min(bw, aw) * min(bh, ah)) / (bw * bh + aw * ah - min(bw, aw) * min(bh, ah))
            for aw, ah in anchors
        ])
        best = int(np.argmax(ious))
        aw, ah = anchors[best]

        target[gy, gx, best, 0] = cx / stride - gx
        target[gy, gx, best, 1] = cy / stride - gy
        target[gy, gx, best, 2] = np.log(bw / aw + 1e-8)
        target[gy, gx, best, 3] = np.log(bh / ah + 1e-8)
        target[gy, gx, best, 4] = 1.0
        target[gy, gx, best, 5 + cls] = 1.0
        has_obj[gy, gx, best] = True
    return target, has_obj
```

锚点选择采用「与真实框形状 IoU 最高」的廉价代理，与 YOLOv2/v3 的分配方式一致。v5 及以后使用更复杂的策略（task-aligned matching、dynamic k）来精炼同一思想。

### Step 6: 三项损失

```python
def yolo_loss(pred, target, has_obj, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=1.0):
    has_obj_t = torch.from_numpy(has_obj).bool()
    target_t = torch.from_numpy(target).float()

    # box-regression loss: only on cells with objects
    box_pred = pred[..., :4][has_obj_t]
    box_true = target_t[..., :4][has_obj_t]
    loss_box = torch.nn.functional.mse_loss(box_pred, box_true, reduction="sum")

    # objectness loss
    obj_pred = pred[..., 4]
    obj_true = target_t[..., 4]
    loss_obj_pos = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[has_obj_t], obj_true[has_obj_t], reduction="sum")
    loss_obj_neg = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[~has_obj_t], obj_true[~has_obj_t], reduction="sum")

    # classification loss on cells with objects
    cls_pred = pred[..., 5:][has_obj_t]
    cls_true = target_t[..., 5:][has_obj_t]
    loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(
        cls_pred, cls_true, reduction="sum")

    total = (lambda_coord * loss_box
             + lambda_obj * loss_obj_pos
             + lambda_noobj * loss_obj_neg
             + lambda_cls * loss_cls)
    return total, {"box": loss_box.item(), "obj_pos": loss_obj_pos.item(),
                   "obj_neg": loss_obj_neg.item(), "cls": loss_cls.item()}
```

五个超参数，每个 YOLO 教程要么硬编码要么 sweep。比例关系更重要：`lambda_coord=5, lambda_noobj=0.5` 复现了原始 YOLOv1 论文，仍是合理的默认设置。

### Step 7: 推理流程

解码原始检测头输出，应用 sigmoid/exp，按目标置信度阈值过滤，再做 NMS。

```python
def postprocess(pred_tensor, anchors, stride, img_size, conf_threshold=0.25, iou_threshold=0.45):
    pred = pred_tensor.detach().cpu().numpy()
    grid_h, grid_w = pred.shape[1], pred.shape[2]
    num_anchors = len(anchors)

    boxes, scores, classes = [], [], []
    for gy in range(grid_h):
        for gx in range(grid_w):
            for a in range(num_anchors):
                tx, ty, tw, th, obj, *cls = pred[0, gy, gx, a]
                score = sigmoid(obj) * sigmoid(np.array(cls)).max()
                if score < conf_threshold:
                    continue
                cls_idx = int(np.argmax(cls))
                cx = (sigmoid(tx) + gx) * stride
                cy = (sigmoid(ty) + gy) * stride
                w = anchors[a][0] * np.exp(tw)
                h = anchors[a][1] * np.exp(th)
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                scores.append(float(score))
                classes.append(cls_idx)

    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    keep = nms(boxes, scores, iou_threshold)
    return boxes[keep], scores[keep], classes[keep]
```

这就是完整的评估路径：检测头 -> 解码 -> 阈值过滤 -> NMS。

## 使用它

`torchvision.models.detection` 提供了生产级检测器，概念结构与你上面实现的一致。加载预训练模型只需三行。

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
model.eval()
with torch.no_grad():
    predictions = model([torch.randn(3, 400, 600)])
print(predictions[0].keys())
print(f"boxes:  {predictions[0]['boxes'].shape}")
print(f"scores: {predictions[0]['scores'].shape}")
print(f"labels: {predictions[0]['labels'].shape}")
```

对于实时推理管线，`ultralytics`（YOLOv8/v9）是标准选择：`from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model(img)`。模型内部处理解码和 NMS，并返回与你上面构建的相同的 `boxes / scores / labels` 三元组。

## 交付它

本节课产出：

- `outputs/prompt-detection-metric-reader.md` —— 一个 prompt，把 `precision, recall, AP, mAP@0.5:0.95` 这一行指标转换成一句话诊断，并指出最有价值的下一步实验。
- `outputs/skill-anchor-designer.md` —— 一个 skill，给定一组真实框数据集后，对 `(w, h)` 运行 k-means，返回每个 FPN 层级的锚点集合，以及选择合适锚点数量所需的覆盖统计。

## 练习

1. **（简单）** 实现 `box_iou`，并在 1,000 组随机边界框上与 `torchvision.ops.box_iou` 对比。验证最大绝对差低于 `1e-6`。
2. **（中等）** 把 `yolo_loss` 改成使用 `CIoU` 框损失的版本。在 100 张合成数据集上证明，在相同 epoch 数下 CIoU 比 MSE 收敛到更高的最终 mAP@0.5:0.95。
3. **（困难）** 实现多尺度推理：把同一张图像以三种分辨率输入模型，合并边界框预测，最后运行一次 NMS。在留出集上测量相比单尺度推理的 mAP 提升。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Anchor | "Box prior" | A pre-defined box shape at each grid cell from which the network predicts deltas instead of absolute coordinates |
| IoU | "Overlap" | Intersection-over-union of two boxes; the universal similarity measure in detection |
| NMS | "Deduplicate" | Greedy algorithm that keeps highest-score predictions and removes overlapping ones above a threshold |
| Objectness | "Is there something here" | Per-anchor, per-cell scalar predicting whether an object is centred in that cell |
| Grid stride | "Downsample factor" | Pixels per grid cell; a 416-px input with a 13-grid head has stride 32 |
| mAP | "Mean average precision" | Average of the area under the precision-recall curve, averaged over classes and (for COCO) IoU thresholds |
| AP@0.5 | "PASCAL VOC AP" | Average precision with IoU threshold 0.5; the lenient version of the metric |
| mAP@0.5:0.95 | "COCO AP" | Average over IoU thresholds 0.5..0.95 step 0.05; the strict version and current community standard |

## 延伸阅读

- [YOLOv1: You Only Look Once (Redmon et al., 2016)](https://arxiv.org/abs/1506.02640) —— 奠基论文；此后每个 YOLO 都是对这一结构的精炼
- [YOLOv3 (Redmon & Farhadi, 2018)](https://arxiv.org/abs/1804.02767) —— 引入多尺度 FPN 风格检测头的论文；图示至今最清晰
- [Ultralytics YOLOv8 docs](https://docs.ultralytics.com) —— 当前生产参考；涵盖数据集格式、增强、训练配方
- [The Illustrated Guide to Object Detection (Jonathan Hui)](https://jonathan-hui.medium.com/object-detection-series-24d03a12f904) —— 对检测器全谱系最通俗的英文导览；对理解 DETR、RetinaNet、FCOS 与 YOLO 的关系极有价值
