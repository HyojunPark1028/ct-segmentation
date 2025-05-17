import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.build_sam import sam_model_registry
from .unet import ClassicUNet

class ProjectorBlock(nn.Module):
    def __init__(self, in_channels=256, out_channels=512):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class MedSAM(nn.Module):
    def __init__(self, checkpoint: str, in_channels: int = 1, out_channels: int = 1, img_size: int = 512):
        super().__init__()

        # 1. SAM ViT-B encoder 로드
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)

        # 2. encoder 일부 fine-tuning 허용
        self.encoder = self.sam.image_encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        for name, p in self.encoder.named_parameters():          
            if any(k in name for k in ["blocks.9", "blocks.10", "blocks.11", "norm"]):
                p.requires_grad = True

        # 3. projector
        self.projector = ProjectorBlock(in_channels=256, out_channels=512)

        # 4. decoder
        self.decoder = ClassicUNet(in_channels=512, out_channels=out_channels)

    def forward(self, x):
        # 입력이 grayscale(1채널)일 경우 → RGB로 확장
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # encoder 출력: B x 256 x H/16 x W/16
        feats = self.encoder(x) 
        # feats = self.sam.image_encoder(x)

        # projector 통해 해상도 유지
        proj = self.projector(feats)

        # decoder 통해 최종 segmentation 출력
        out = self.decoder(proj)  # (B, 1, H/16, W/16)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

