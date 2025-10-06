# 基于 ViT 的低光照图像增强

> Low-Light Image Enhancement Based on Vision Transformer

## 📝 项目简介

基于 Vision Transformer (ViT) 的低光照图像增强深度学习项目。利用 ViT 的全局注意力机制,结合掩码解码器和增强网络,实现智能图像增强。

### 核心特性

- **ViT 特征提取**: 冻结的预训练 ViT-Base 模型提取全局特征
- **掩码解码器**: 自动定位图像暗区,生成注意力掩码
- **增强网络**: 基于掩码的卷积网络,提升亮度同时保留细节
- **复合损失函数**: L1 损失 + VGG16 感知损失 + 暗区强化损失

## 🏗️ 模型架构

```
输入图像 (224×224)
    ↓
ViT-Base 特征提取 [冻结]
    ↓
掩码解码器 → 注意力掩码
    ↓
RGB + 掩码 → 增强网络
    ↓
增强图像输出
```

## 📦 环境依赖

```bash
# 核心依赖
pip install torch torchvision transformers opencv-python albumentations tensorboard
```

**主要库**:
- PyTorch >= 1.9
- Transformers (Hugging Face)
- OpenCV
- Albumentations (数据增强)
- TensorBoard (训练可视化)

## 🚀 快速开始

### 1. 数据准备

按以下结构组织数据集:

```
data/LOL_dataset/
├── train/
│   ├── low/    # 低光图像
│   └── high/   # 正常光图像
└── val/
    ├── low/
    └── high/
```

### 2. 配置参数

编辑 `config.py` 调整训练参数:

```python
class Config:
    data_root = "./data/LOL_dataset"  # 数据集路径
    input_size = 224                  # 输入尺寸
    batch_size = 8                    # 批次大小
    epochs = 100                      # 训练轮数
    lr = 1e-4                        # 学习率
```

### 3. 训练模型

```bash
python train.py

# 查看训练日志
tensorboard --logdir=./runs
```

### 4. 推理增强

```bash
python infer2.py
```

修改 `infer2.py` 中的路径参数:
```python
input_folder = "./test"      # 输入文件夹
output_folder = "./output"   # 输出文件夹
```

支持格式: `.jpg`, `.jpeg`, `.png`, `.bmp`

## 📁 项目结构

```
.
├── config.py         # 配置文件
├── model.py          # DarkEnhancer 模型定义
├── dataset.py        # CustomLowLightDataset 数据加载器
├── utils.py          # LossCalculator 复合损失函数
├── train.py          # 训练脚本
├── infer2.py         # 推理脚本
├── data/             # 数据集目录
│   └── LOL_dataset/
│       ├── train/
│       └── val/
├── checkpoints/      # 模型检查点
├── runs/             # TensorBoard 日志
├── test/             # 测试图像输入
└── output/           # 增强结果输出
```

## 📄 License

本项目基于 MIT 协议开源,详见 [LICENSE](LICENSE) 文件。

