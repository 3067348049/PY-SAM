# Python大作业：YOLO + UNet/DeepLabV3+ + SAM 综合分割实验

> 本仓库提供完整的计算机视觉分割实验流程，集成 YOLO 目标检测、UNet/DeepLab 语义分割、SAM 微调与推理。  
> 支持多模型性能对比、统一可视化、全面指标评估。

## 📁 1. 项目目录结构
```text
Python大作业/
├─ dataset_seg/         # UNet/DeepLab/SAM 数据集（语义分割）
│  ├─ images/{train,valid,test}/
│  └─ masks/{train,valid,test}/
│
├─ dataset_yolo/        # YOLO 数据集（YOLO 标注格式）
│  ├─ images/{train,val,test}/
│  └─ labels/{train,val,test}/
│
├─ runs/                # SAM/YOLO 训练输出
│
├─ results/         # 所有模型横向拼图可视化
|
├─ runs/            # YOLOv11预测输出结果
│
├─ tools/               # 本仓库核心代码
│  ├─ ft_sam.py         # SAM 微调
│  ├─ infer_sam.py      # SAM 推理
│  ├─ infer_unet.py     # UNet/DeepLab 推理
│  ├─ train_unet.py     # UNet/AttUNet/UNet++ 训练
│  ├─ train_deeplabv3+.py
│  ├─ train_yolo.py
│  ├─ predict_yolo.py
│  ├─ show_diff.py      # 多模型对比可视化
│  └─ trans.py          # 数据转换工具
│
├─ Unet/                # ★ 需要手动克隆 Awesome-UNet 到此
│  └─ Awesome-U-Net/
│     ├─ models/unet.py
│     ├─ models/attunet.py
│     └─ models/unetpp.py
│
├─ segment-anything/    # ★ 需要手动克隆 SAM
│  ├─ sam_vit_b_01ec64.pth
│  └─ ...
│
└─ ultralytics/         # YOLO 源码，也可 pip 安装

```

---

## ⚙️ 2. 环境安装（必读）

### 2.1 Python 依赖包

建议使用 Python 3.8+ 环境，执行以下命令安装所需依赖：
```bash
# 基础科学计算库
pip install numpy opencv-python tqdm matplotlib pyyaml

# 深度学习框架
pip install torch torchvision torchaudio

# 分割模型库
pip install segmentation-models-pytorch

# YOLO 框架
pip install ultralytics

# COCO 评估工具
pip install pycocotools
```

**提示**：如遇到 PyTorch 安装问题，请访问 [PyTorch 官网](https://pytorch.org/) 根据您的 CUDA 版本选择合适的安装命令。

---

## 📦 3. 手动克隆外部依赖（本仓库未包含）

### 3.1 Segment Anything (SAM)

**克隆仓库：**
```bash
git clone https://github.com/facebookresearch/segment-anything.git
```

**下载预训练权重：**

请从 [SAM 官方仓库](https://github.com/facebookresearch/segment-anything#model-checkpoints) 下载权重文件（推荐 ViT-B）：
```bash
# 将下载的权重放置在此路径
segment-anything/sam_vit_b_01ec64.pth
```

### 3.2 Awesome-UNet

**克隆 Pytorch-UNet 仓库：**
```bash
mkdir Unet
cd Unet
git clone https://github.com/milesial/Pytorch-UNet.git Awesome-U-Net
```

**确认文件结构：**
```bash
Unet/Awesome-U-Net/
├─ models/unet.py       # 标准 UNet
├─ models/attunet.py    # Attention UNet
└─ models/unetpp.py     # UNet++
```

### 3.3 YOLO (Ultralytics)

**最简单方式（推荐）：**
```bash
pip install ultralytics
```

**或手动克隆（可选）：**
```bash
git clone https://github.com/ultralytics/ultralytics.git
```

---

## 📊 4. 数据格式说明

### 4.1 语义分割数据集（UNet/DeepLab/SAM）

**目录结构：**
```bash
dataset_seg/
├─ images/
│  ├─ train/xxx.jpg
│  ├─ valid/xxx.jpg
│  └─ test/xxx.jpg
└─ masks/
   ├─ train/xxx.png    # 单通道灰度图，前景像素值 > 0
   ├─ valid/xxx.png
   └─ test/xxx.png
```

**数据集下载：**

- **推荐数据集**：[flower_segmentationV2 Dataset by MOD05GEN25HUGO](https://universe.roboflow.com/mod05gen25hugo/flower_segmentationv2)
- 下载后解压至 `dataset_seg/` 目录

**数据要求：**

- 图像格式：`.jpg` / `.png`
- 掩码格式：单通道 PNG，背景为 0，前景为 1 或 255
- 图像与掩码文件名必须一一对应

### 4.2 YOLO 数据集

**目录结构：**
```bash
dataset_yolo/
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ labels/
   ├─ train/xxx.txt    # YOLO 格式标注
   ├─ val/xxx.txt
   └─ test/xxx.txt
```

**标注格式（每行一个目标）：**
```text
class_id center_x center_y width height
```

- 所有坐标值归一化到 [0, 1]
- 示例：`0 0.5 0.5 0.3 0.4`

---

## 🚀 5. 快速开始

### 5.1 训练 UNet 系列模型
```bash
python tools/train_unet.py \
    --model unet \              # 可选: unet, attunet, unetpp
    --img_size 512 \
    --batch_size 4 \
    --epochs 50 \
    --lr 0.001 \
    --data_dir dataset_seg
```

### 5.2 训练 DeepLabV3+
```bash
python tools/train_deeplabv3+.py \
    --backbone resnet50 \
    --img_size 512 \
    --batch_size 8 \
    --epochs 50
```

### 5.3 微调 SAM
```bash
python tools/ft_sam.py \
    --model_type vit_b \
    --checkpoint segment-anything/sam_vit_b_01ec64.pth \
    --train_dir dataset_seg/images/train \
    --mask_dir dataset_seg/masks/train \
    --epochs 10 \
    --batch_size 2
```

### 5.4 训练 YOLO
```bash
python tools/train_yolo.py \
    --model yolov8n.pt \
    --data dataset_yolo/data.yaml \
    --epochs 100 \
    --imgsz 640
```

---

## 🔍 6. 模型推理

### 6.1 UNet/DeepLab 推理
```bash
python tools/infer_unet.py \
    --model_path runs/unet/best.pth \
    --img_dir dataset_seg/images/test \
    --output_dir results/unet
```

### 6.2 SAM 推理
```bash
python tools/infer_sam.py \
    --checkpoint runs/sam/finetuned_sam.pth \
    --img_dir dataset_seg/images/test \
    --output_dir results/sam
```

### 6.3 YOLO 推理
```bash
python tools/predict_yolo.py \
    --weights runs/yolo/best.pt \
    --source dataset_yolo/images/test \
    --save_dir results/yolo
```

---

## 📈 7. 模型对比与可视化

**生成多模型对比图：**
```bash
python tools/show_diff.py \
    --original dataset_seg/images/test \
    --gt dataset_seg/masks/test \
    --pred_unet results/unet \
    --pred_sam results/sam \
    --pred_deeplab results/deeplab \
    --output compare_all
```

**输出示例：**
- 横向拼接：原图 | GT | UNet | SAM | DeepLab
- 自动计算各模型 IoU/Dice 指标并标注在图像上

---

## ❓ 8. 常见问题（FAQ）

### Q1: AttUNet 推理效果很差？

**A: 90% 的情况是以下原因导致：**

1. **图像尺寸不一致**  
   训练时使用 `--img_size 512`，推理时却用了其他尺寸 → 务必保持一致

2. **模型路径错误**  
   推理加载的是 `.../unet/best.pth` 而非 `.../attunet/best.pth`

3. **掩码值范围不匹配**  
   训练时 mask 写成 255，推理 resize 后阈值判断错误 → 统一使用 0/1 或 0/255

4. **数据增强不一致**  
   训练时使用随机裁剪，推理时未做相应的 padding 处理

**解决方案**：检查训练配置文件，确保推理时使用完全相同的参数。

---

### Q2: 为什么 SAM 推理速度慢？

**A: 原因分析：**

- SAM 采用 **Vision Transformer (ViT)** 作为图像编码器，计算量巨大
- ViT-B 模型参数量约 90M，单张图像编码耗时较长

**优化建议：**

1. **使用 GPU**：确保 CUDA 可用，避免 CPU 推理
2. **降低批次大小**：ViT-B 建议使用 `batch_size=1`
3. **使用更小模型**：尝试 ViT-Tiny 或 MobileSAM
4. **图像预处理**：适当降低输入分辨率（如 1024 → 512）

---

### Q3: 如何转换数据集格式？

**A: 使用内置转换工具：**
```bash
python tools/trans.py \
    --input_format coco \        # 输入格式: coco/voc/yolo
    --output_format seg \        # 输出格式: seg/yolo
    --input_dir raw_dataset \
    --output_dir dataset_seg
```

支持的格式转换：
- COCO → 语义分割
- VOC → YOLO
- YOLO → 语义分割

---

### Q4: 训练时显存不足？

**A: 优化策略：**

1. **减小批次大小**：`--batch_size 2` 或 `--batch_size 1`
2. **降低图像分辨率**：`--img_size 256` 或 `--img_size 384`
3. **使用混合精度训练**：添加 `--amp` 参数
4. **使用梯度累积**：`--accumulate 4`（每4步更新一次）
5. **冻结部分层**：仅训练解码器部分

---

## 🛠️ 9. 高级功能

### 9.1 数据增强配置

修改 `tools/train_unet.py` 中的 `get_transforms()` 函数：
```python
import albumentations as A

transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.ColorJitter(brightness=0.2, contrast=0.2),
    A.GaussNoise(p=0.2),
    A.Resize(img_size, img_size),
])
```

### 9.2 学习率调度

支持的调度器：
- **StepLR**：每 N 个 epoch 衰减
- **CosineAnnealingLR**：余弦退火
- **ReduceLROnPlateau**：验证集指标停滞时衰减

### 9.3 早停机制
```bash
python tools/train_unet.py \
    --early_stopping \
    --patience 10        # 验证集指标 10 个 epoch 不提升则停止
```

---

## 📄 10. 许可 & 致谢

### 开源许可

本项目采用 **MIT License** 开源协议。

### 致谢

本项目依赖以下优秀开源项目：

- **[Segment Anything](https://github.com/facebookresearch/segment-anything)** - Meta AI Research
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** - Ultralytics
- **[Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)** - Milesial
- **[Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)** - Pavel Yakubovskiy

特别感谢所有为计算机视觉开源社区做出贡献的开发者！
