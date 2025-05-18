import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry


class ResidualProjector(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.identity = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.relu(self.conv(x) + self.identity(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        if skip.shape[-2:] != x.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SAM2UNet(nn.Module):
    def __init__(self, checkpoint: str, in_channels: int = 1, out_channels: int = 1):
        super().__init__()

        # 1. Load SAM ViT-B encoder
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)

        # 2. Freeze all layers except blocks.4~11 and norm
        for p in self.sam.image_encoder.parameters():
            p.requires_grad = False
        for name, p in self.sam.image_encoder.named_parameters():
            if any(f"blocks.{i}" in name for i in range(4, 12)) or "norm" in name:
                p.requires_grad = True

        # 3. Define projector
        self.projector = ResidualProjector(in_channels=256, out_channels=256)

        # 4. Decoder (SAM encoder 중간 feature 사용하여 skip connection 구성)
        self.up4 = UpBlock(256, 256, 128)
        self.up3 = UpBlock(128, 128, 64)
        self.up2 = UpBlock(64, 64, 32)
        self.up1 = UpBlock(32, 32, 16)

        # 5. Final output layer
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # ✅ Step 1: patch embedding
        x = self.sam.image_encoder.patch_embed(x)

        # 🐛 디버깅: patch_embed 출력 shape 확인
        print(f"[DEBUG] patch_embed output shape: {x.shape}")

        # ✅ Step 1.5: reshape to [B, C, H, W] if necessary
        if x.ndim == 3:
            # Case: [B, H*W, C] → [B, C, H, W]
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            print(f"[DEBUG] reshaped from [B, N, C] to: {x.shape}")
        elif x.ndim == 4 and x.shape[-1] in (768, 256):
            # Case: [B, H, W, C] → [B, C, H, W]
            x = x.permute(0, 3, 1, 2).contiguous()
            print(f"[DEBUG] permuted from [B, H, W, C] to: {x.shape}")

        # ✅ Step 2: transformer blocks
        feats = x
        skips = []
        for i, blk in enumerate(self.sam.image_encoder.blocks):
            feats = blk(feats)
            if i in [2, 4, 6, 8]:
                skips.append(feats)

        # ✅ Step 3: projector
        feats = self.projector(feats)

        # ✅ Step 4: decoder with skip connections
        d4 = self.up4(feats, skips[-1])
        d3 = self.up3(d4, skips[-2])
        d2 = self.up2(d3, skips[-3])
        d1 = self.up1(d2, skips[-4])
        out = self.out_conv(d1)

        # ✅ Step 5: resize to match input
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out