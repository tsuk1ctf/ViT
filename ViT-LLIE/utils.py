import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class LossCalculator:
    """
    复合损失函数：
        L_total = α·L_Pixel + β·L_Perceptual + γ·L_dark

    默认配置（Dark Region Dominant，通过网格搜索确定的最优组合）：
        α=0.20, β=0.20, γ=0.60
    γ 主导暗区增强，α/β 兼顾整体像素保真度和感知自然度。
    """
    def __init__(self, device,
                 alpha: float = 0.20,
                 beta: float = 0.20,
                 gamma: float = 0.60):
        # Dark Region Dominant 配置：γ 主导暗区增强
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # 基础损失函数
        self.l1_loss = nn.L1Loss()  # 像素级 L1 损失

        # VGG16 前 16 层用于感知损失（conv1_1 到 conv3_3）
        # 中层特征(纹理/边缘)比高层语义特征更适合像素级图像恢复任务
        self.vgg = vgg16(pretrained=True).features[:16].to(device).eval()
        # 冻结 VGG 参数
        for param in self.vgg.parameters():
            param.requires_grad = False

        # ImageNet 标准化常量（VGG 的预训练输入要求）
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def perceptual_loss(self, pred, target):
        """
        感知损失：在 VGG16 中层特征上计算 L1 距离
            L_Perceptual = mean(|f̂(pred) - f̂(target)|)
        其中 f̂ = f / ||f||_2 为 L2 归一化后的特征
        """
        # ImageNet 标准化（VGG 预训练域）
        pred_norm = (pred - self._mean) / self._std
        target_norm = (target - self._mean) / self._std

        # 提取 VGG 中层特征（conv1_1 ~ conv3_3）
        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)

        # L2 归一化：在通道维度做归一化，保证不同尺度特征的权重平衡
        pred_features = F.normalize(pred_features, p=2, dim=1)
        target_features = F.normalize(target_features, p=2, dim=1)

        # 特征绝对差
        return self.l1_loss(pred_features, target_features)

    def dark_loss(self, pred, target, mask):
        """
        暗区强化损失：在 mask 标识的暗区上计算 mask-weighted MSE
            L_dark = sum((pred·m - target·m)^2) / N_mask

        分母 N_mask = mask 有效像素权重之和(乘以 RGB 通道数对齐)；
        相比 L1，MSE 的二次惩罚对暗区像素差异更敏感。
        """
        # mask 加权后的平方误差
        diff_squared = (pred * mask - target * mask) ** 2
        # N_mask = Σ m_i（mask 区域的有效像素权重之和）
        # mask 是 [0,1] 连续值；通道方向广播：每个 RGB 通道使用同一个 mask
        n_mask = mask.sum() * pred.shape[1] + 1e-8  # 乘 RGB 通道数对齐求和维度
        return diff_squared.sum() / n_mask

    def total_loss(self, pred, target, mask):
        """
        复合损失：
            L_total = α·L_Pixel + β·L_Perceptual + γ·L_dark
        """
        # 像素级 L1 损失：整体像素保真度
        l_pixel = self.l1_loss(pred, target)

        # 感知损失：纹理与边缘等中层语义一致
        l_percep = self.perceptual_loss(pred, target)

        # 暗区强化损失：聚焦 mask 标识的暗区做 MSE 优化
        l_dark = self.dark_loss(pred, target, mask)

        # 加权组合
        return self.alpha * l_pixel + self.beta * l_percep + self.gamma * l_dark
