import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import ClassicUNet

# SAM from Meta AI (ViT backbone)
from segment_anything import sam_model_registry


class SAM2UNet(nn.Module):
    def __init__(self, checkpoint: str, in_channels: int = 1, out_channels: int = 1):
        """
        SAM2-UNet 모델 구성
        - SAM encoder는 freeze 상태로 feature extractor 역할만 수행
        - UNet decoder는 기존 ClassicUNet을 재활용

        Args:
            checkpoint (str): SAM 모델의 checkpoint 경로 (.pth)
            in_channels (int): 입력 채널 수 (기본 1 - grayscale CT)
            out_channels (int): 출력 채널 수 (기본 1 - binary segmentation)
        """
        super().__init__()

        # SAM encoder 로드 (ViT-H)
        self.sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
        self.encoder = self.sam.image_encoder
        # self.encoder.eval()
        # self.encoder.requires_grad_(False)  # Freeze

        self.encoder.eval()
        # ❶ 전체를 먼저 freeze
        for p in self.encoder.parameters():
            p.requires_grad = False

        # ❷ 마지막 두 블록과 최종 LayerNorm만 풀어서 미세조정
        for name, p in self.encoder.named_parameters():
            if "blocks.10" in name or "blocks.11" in name or "norm" in name:
                p.requires_grad = True        

        # SAM encoder output: (B, 256, 64, 64) → decoder에 맞게 channel projection
        self.projector = nn.Conv2d(256, 512, kernel_size=1)

        # UNet decoder (입력 채널: 512, 출력 채널: 1)
        self.decoder = ClassicUNet(in_channels=512, out_channels=out_channels)

    def forward(self, x):
        """
        Args:
            x (Tensor): 입력 이미지 (B, 1, H, W) - 예: (B, 1, 512, 512)
        Returns:
            Tensor: 세그멘테이션 마스크 출력 (B, 1, H, W)
        """
        # SAM은 RGB (3채널), 1024x1024 사이즈 입력 필요
        x_rgb = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
        x_rgb = x_rgb.repeat(1, 3, 1, 1)  # Grayscale → 3채널 복제

        with torch.no_grad():
            feat = self.encoder(x_rgb)  # (B, 256, 64, 64)

        feat_proj = self.projector(feat)  # (B, 512, 64, 64)
        out = self.decoder(feat_proj)    # (B, 1, 64, 64) → 내부에서 upsample 진행
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)  # restore to original size

        return out
