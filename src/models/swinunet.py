import torch
import torch.nn as nn
from timm.models.swin_transformer import swin_base_patch4_window7_224

class SwinUNet(nn.Module):
    def __init__(self, img_size=224, num_classes=1, use_pretrained=True):
        super().__init__()
        self.backbone = swin_base_patch4_window7_224(pretrained=use_pretrained)
        self.backbone.head = nn.Identity()  # 분류기 제거

        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 2, stride=2),  # 7 → 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 2, stride=2),   # 14 → 28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2),   # 28 → 56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),    # 56 → 112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),     # 112 → 224
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # [B, 1, 224, 224] → [B, 3, 224, 224]
        feats = self.backbone.forward_features(x)  # 💡 [B, 7, 7, 1024]
        x = feats.permute(0, 3, 1, 2).contiguous()  # [B, 1024, 7, 7] # ✅ permute to [B, C, H, W]
        return self.upconv(x)  # → [B, 1, 224, 224]
