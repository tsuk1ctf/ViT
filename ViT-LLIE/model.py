import torch
import torch.nn as nn
from transformers import ViTModel


class DarkEnhancer(nn.Module):
    """Frozen ViT-Base + Mask Decoder + 3-layer 3x3 Enhancement Network."""

    def __init__(self):
        super().__init__()
        # 冻结 ViT 主干，只训练下方两个模块
        self.vit = ViTModel.from_pretrained('vit-base-patch16-224-in21k')
        self.vit.requires_grad_(False)

        # Mask Decoder: ViT 特征 [B,768,14,14] -> 单通道暗区注意力 [B,1,224,224]
        self.mask_conv1 = nn.Conv2d(768, 256, kernel_size=1, bias=True)      # 通道降维
        self.mask_upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.mask_conv2 = nn.Conv2d(256, 1, kernel_size=3, padding=1, bias=True)
        self.mask_conv3 = nn.Conv2d(1, 1, kernel_size=1, bias=True)          # 独立 1x1
        # Kaiming 初始化适配后续 ReLU
        for m in [self.mask_conv1, self.mask_conv2, self.mask_conv3]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

        # Enhancement Network: RGB+mask (4 通道) -> 增强 RGB，全程 stride=1 保持 224x224
        self.enhance_conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.enhance_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.enhance_conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # 取 ViT patch tokens（跳过 CLS），reshape 回 14x14 空间网格（224/16=14）
        patch_tokens = self.vit(x).last_hidden_state[:, 1:]                  # [B, 196, 768]
        features = patch_tokens.permute(0, 2, 1).view(x.size(0), 768, 14, 14)

        # Mask 解码：14x14 -> 224x224，sigmoid 归一化到 [0,1]
        m = torch.relu(self.mask_conv1(features))
        m = torch.relu(self.mask_conv2(self.mask_upsample(m)))
        mask = torch.sigmoid(self.mask_conv3(m))                             # [B, 1, 224, 224]

        # 沿通道维拼接原图与 mask，送入增强网络
        x_aug = torch.cat([x, mask], dim=1)                                  # [B, 4, 224, 224]
        h1 = torch.relu(self.enhance_conv1(x_aug))
        h2 = torch.relu(self.enhance_conv2(h1))
        enhanced = torch.sigmoid(self.enhance_conv3(h2))                     # [B, 3, 224, 224]

        return enhanced, mask
