import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import ClassicUNet
from segment_anything import sam_model_registry


class ProjectorBlock(nn.Module):
    def __init__(self, in_channels=256, out_channels=512):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SAM2UNet(nn.Module):
    def __init__(self, checkpoint: str, in_channels: int = 1, out_channels: int = 1, img_size: int = 1024):
        """
        SAM2-UNet: SAM encoder (ViT-H) + 강화된 projector + ClassicUNet decoder
        AMP 없이 구성. 일부 encoder만 fine-tune

        Args:
            checkpoint (str): SAM 모델의 checkpoint 경로 (.pth)
            in_channels (int): 입력 채널 수 (CT 이미지: 1)
            out_channels (int): 출력 채널 수 (보통 1)
            img_size (int): SAM encoder 입력 크기 (기본 640)
        """
        super().__init__()
        self.img_size = img_size

        # 1. SAM encoder 로드 (ViT-H)
        self.sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
        self.encoder = self.sam.image_encoder

        # 2. encoder 일부 fine-tuning
        self.encoder.train()
        for p in self.encoder.parameters():
            p.requires_grad = False
        for name, p in self.encoder.named_parameters():
            if any(k in name for k in ["blocks.9", "blocks.10", "blocks.11", "norm"]):
                p.requires_grad = True

        # 3. channel projection 강화
        self.projector = ProjectorBlock(in_channels=256, out_channels=512)

        # 4. UNet decoder
        self.decoder = ClassicUNet(in_channels=512, out_channels=out_channels)

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, 1, H, W)
        Returns:
            Tensor: (B, 1, H, W) → 세그멘테이션 결과
        """
        # 입력 리사이즈 및 3채널 변환
        x_rgb = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        x_rgb = x_rgb.repeat(1, 3, 1, 1)

        # encoder → projector → decoder
        feat = self.encoder(x_rgb)              # (B, 256, 64, 64)
        feat_proj = self.projector(feat)        # (B, 512, 64, 64)
        out = self.decoder(feat_proj)           # (B, 1, 64, 64)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return out
