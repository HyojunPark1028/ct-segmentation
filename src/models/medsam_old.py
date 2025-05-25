import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.build_sam import sam_model_registry
from .unet import ClassicUNet

class ProjectorBlock(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class MedSAM(nn.Module):
    def __init__(self, checkpoint: str, in_channels: int = 1, out_channels: int = 1, img_size: int = 512):
        super().__init__()

        # 1. SAM ViT-B encoder 로드
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)

        # 2. encoder 일부 fine-tuning 허용 (blocks.6~11 + norm)
        for p in self.sam.image_encoder.parameters():
            p.requires_grad = True
        # for name, p in self.sam.image_encoder.named_parameters():
        #     if any(k in name for k in ["blocks.4", "blocks.5", "blocks.6", "blocks.7", "blocks.8", "blocks.9", "blocks.10", "blocks.11", "norm"]):
        #         p.requires_grad = True

        # 3. projector
        self.projector = ProjectorBlock(in_channels=256, out_channels=256)

        # 4. decoder
        self.decoder = ClassicUNet(in_channels=256, out_channels=out_channels)

        # 5. Decoder 마지막 출력층의 bias 초기값 설정 및 학습 가능하도록 수정
        self.final_conv_ref = None
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels == out_channels:
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    m.bias.requires_grad = True
                    self.final_conv_ref = m


    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 🔵 A. Encoder 출력 확인
        feats = self.sam.image_encoder(x)
        # print(f"\n[ENCODER OUTPUT]")
        # print(f"Shape: {feats.shape}")
        # print(f"Mean: {feats.mean().item():.6f}, Std: {feats.std().item():.6f}, Min: {feats.min().item():.6f}, Max: {feats.max().item():.6f}")

        # 🟢 B. Projector 출력 확인
        proj = self.projector(feats)
        # print(f"[PROJECTOR OUTPUT]")
        # print(f"Shape: {proj.shape}")
        # print(f"Mean: {proj.mean().item():.6f}, Std: {proj.std().item():.6f}, Min: {proj.min().item():.6f}, Max: {proj.max().item():.6f}")

        # 🟣 C. Decoder 출력 확인
        out = self.decoder(proj)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        # print(f"[DECODER OUTPUT]")
        # print(f"Shape: {out.shape}")
        # print(f"Mean: {out.mean().item():.6f}, Std: {out.std().item():.6f}, Min: {out.min().item():.6f}, Max: {out.max().item():.6f}")

        # if self.final_conv_ref is not None:
        #     print(f"[DECODER BIAS MEAN]: {self.final_conv_ref.bias.data.mean().item():.6f}")

        # pred = torch.sigmoid(out)
        # print(f"[SIGMOID MEAN]: {pred.mean().item():.6f}")

        return out  # sigmoid는 loss나 eval 쪽에서 처리
