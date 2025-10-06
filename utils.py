import torch
import torch.nn as nn
from torchvision.models import vgg16


class LossCalculator:
    def __init__(self, device):
        # 基础损失函数
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        # 初始化VGG16用于感知损失（取前16层）
        self.vgg = vgg16(pretrained=True).features[:16].to(device).eval()
        # 冻结VGG参数
        for param in self.vgg.parameters():
            param.requires_grad = False


    def perceptual_loss(self, pred, target):
        # 输入归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std

        # 特征提取
        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)

        return self.l1_loss(pred_features, target_features)

    def total_loss(self, pred, target, mask):
        # 像素级L1损失
        l_pixel = self.l1_loss(pred, target)

        # 感知损失（特征匹配）
        l_percep = self.perceptual_loss(pred, target)

        # 暗区强化损失（聚焦mask区域）
        masked_pred = pred * mask
        masked_target = target * mask
        l_dark = self.l1_loss(masked_pred, masked_target)

        # 加权组合损失
        return 0.5 * l_pixel + 0.3 * l_percep + 0.2 * l_dark