import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CustomLowLightDataset(Dataset):
    def __init__(self, data_dir, phase='train', transform=None):
        self.phase = phase  # 数据集阶段（train/val）
        self.low_dir = os.path.join(data_dir, phase, "low")  # 低光图像目录
        self.high_dir = os.path.join(data_dir, phase, "high")  # 正常光图像目录
        self.image_names = os.listdir(self.low_dir)  # 获取所有图像文件名
        self.transform = transform  # 数据增强变换

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # 读取图像对
        image_name = self.image_names[idx]
        low_path = os.path.join(self.low_dir, image_name)
        high_path = os.path.join(self.high_dir, image_name)

        # OpenCV读取并转换颜色空间
        low_img = cv2.cvtColor(cv2.imread(low_path), cv2.COLOR_BGR2RGB)
        high_img = cv2.cvtColor(cv2.imread(high_path), cv2.COLOR_BGR2RGB)

        # 调整图像尺寸
        low_img = cv2.resize(low_img, (224, 224))
        high_img = cv2.resize(high_img, (224, 224))

        # 归一化处理
        low_img = low_img.astype(np.float32) / 255.0
        high_img = high_img.astype(np.float32) / 255.0

        # 应用数据增强（保持图像对同步变换）
        if self.transform:
            augmented = self.transform(image=low_img, target=high_img)
            low_img, high_img = augmented['image'], augmented['target']
        else:
            low_img = transforms.ToTensor()(low_img)
            high_img = transforms.ToTensor()(high_img)

        return low_img, high_img


def get_dataloader(data_dir, phase='train', batch_size=8):
    # 定义不同阶段的数据增强策略
    if phase == 'train':
        transform = A.Compose([
            A.Resize(224, 224),
            # A.RandomHorizontalFlip(),  # 可选的数据增强
            A.ColorJitter(brightness=0.2, contrast=0.2),  # 随机调整亮度和对比度
            ToTensorV2()
        ], additional_targets={'target': 'image'})  # 同时对目标图像应用相同变换
    else:
        transform = A.Compose([
            A.Resize(224, 224),
            ToTensorV2()
        ], additional_targets={'target': 'image'})

    # 创建数据集和数据加载器
    dataset = CustomLowLightDataset(data_dir, phase, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(phase == 'train'),  # 仅训练集打乱顺序
        num_workers=4,  # 多进程加载
        drop_last=True  # 丢弃最后不完整的批次
    )