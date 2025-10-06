import torch
import torch.nn as nn
from transformers import ViTModel


class DarkEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练ViT模型
        local_model_path = 'vit-base-patch16-224-in21k'
        self.vit = ViTModel.from_pretrained(local_model_path)
        self.vit.requires_grad_(False)  # 冻结ViT参数

        # 掩码解码器（将ViT特征转换为注意力掩码）
        self.mask_decoder = nn.Sequential(
            nn.Conv2d(768, 256, 1),  # 1x1卷积降维
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False),  # 上采样到原尺寸
            nn.Conv2d(256, 1, 3, padding=1),  # 输出单通道掩码
            nn.Sigmoid()  # 归一化到[0,1]
        )

        # 图像增强网络（输入：原图+掩码）
        self.enhance_net = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),  # 输入通道4（RGB+mask）
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),  # 输出3通道RGB
            nn.Sigmoid()  # 输出归一化到[0,1]
        )

    def forward(self, x):
        # ViT特征提取
        vit_output = self.vit(x).last_hidden_state  # [B, 197, 768]
        # 分离CLS token和图像patch tokens
        cls_token, patch_tokens = vit_output[:, 0], vit_output[:, 1:]

        # 重塑特征图 [B, 768, 14, 14]（ViT的patch尺寸为16x16，224/16=14）
        batch_size = x.size(0)
        features = patch_tokens.permute(0, 2, 1).view(batch_size, 768, 14, 14)

        # 生成注意力掩码
        mask = self.mask_decoder(features)

        # 拼接原图和掩码
        x_aug = torch.cat([x, mask], dim=1)  # 沿通道维度拼接

        # 生成增强图像
        enhanced = self.enhance_net(x_aug)
        return enhanced, mask