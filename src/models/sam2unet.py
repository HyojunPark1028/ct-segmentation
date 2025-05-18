import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry


class ResidualProjector(nn.Module):
    def __init__(self, in_channels=768, out_channels=256):  # ViT-B: 768 output
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

        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)

        for p in self.sam.image_encoder.parameters():
            p.requires_grad = False
        for name, p in self.sam.image_encoder.named_parameters():
            if any(f"blocks.{i}" in name for i in range(4, 12)) or "norm" in name:
                p.requires_grad = True

        self.projector = ResidualProjector(in_channels=768, out_channels=256)

        self.up4 = UpBlock(256, 256, 128)
        self.up3 = UpBlock(128, 128, 64)
        self.up2 = UpBlock(64, 64, 32)
        self.up1 = UpBlock(32, 32, 16)

        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Step 1: Patchify image
        x = self.sam.image_encoder.patch_embed(x)  # [B, C=768, H/16, W/16]
        B, C, H, W = x.shape

        # Step 2: Flatten and add positional encoding
        x = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        num_patches = x.shape[1]

        pos_embed = self.sam.image_encoder.pos_embed  # [1, L, C]
        if pos_embed.shape[1] != num_patches:
            pos_embed = F.interpolate(
                pos_embed.transpose(1, 2),         # [1, C, L]
                size=num_patches,
                mode="nearest",                    # ✅ 수정된 안전 모드
            ).transpose(1, 2)                      # [1, num_patches, C]


        x = x + pos_embed
        x = self.sam.image_encoder.pos_drop(x)

        # Step 3: Pass through transformer blocks
        skips = []
        for i, blk in enumerate(self.sam.image_encoder.blocks):
            x = blk(x)
            if i in [2, 4, 6, 8]:
                skips.append(x)

        # Step 4: Final LayerNorm
        x = self.sam.image_encoder.norm(x)  # [B, HW, C]

        # Step 5: Reshape to 2D feature map
        x = x.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        skips = [s.transpose(1, 2).view(B, C, H, W) for s in skips]

        # Step 6: Projector
        x = self.projector(x)
        skips = [self.projector(s) for s in skips]

        # Step 7: Decoder
        d4 = self.up4(x, skips[-1])
        d3 = self.up3(d4, skips[-2])
        d2 = self.up2(d3, skips[-3])
        d1 = self.up1(d2, skips[-4])

        print(f"[DEBUG] decoder output d1 shape: {d1.shape}")  # 👈 이 줄!

        out = self.out_conv(d1)

        print(f"[DEBUG] after out_conv shape: {out.shape}")     # 👈 이 줄도 추가

        # Step 8: Final resize
        print(f"[DEBUG] decoder out before interpolate: {out.shape}")
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        return out

