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
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
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
            p.requires_grad = True
        # for name, p in self.sam.image_encoder.named_parameters():
        #     if any(f"blocks.{i}" in name for i in range(2, 12)) or "norm" in name:
        #         p.requires_grad = True

        trainable_params = [name for name, p in self.sam.image_encoder.named_parameters() if p.requires_grad]
        print("Trainable SAM params:", trainable_params)


        # Main feature projector
        self.projector = ResidualProjector(in_channels=768, out_channels=128)
        # Skip channel reducers
        # self.skip_proj4 = nn.Conv2d(768, 128, 1)
        self.skip_proj4 = nn.Sequential(
            nn.Conv2d(768, 128, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True)
        )
        self.skip_proj3 = nn.Conv2d(768, 64, 1)
        self.skip_proj2 = nn.Conv2d(768, 32, 1)
        self.skip_proj1 = nn.Conv2d(768, 16, 1)

        self.up4 = UpBlock(128, 128, 64)
        self.up3 = UpBlock(64, 64, 32)
        self.up2 = UpBlock(32, 32, 16)
        self.up1 = UpBlock(16, 16, 8)
        # self.out_conv = nn.Conv2d(8, out_channels, kernel_size=1)
        self.out_conv = nn.Sequential(
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1)
        )

    def forward(self, x):
        input_shape = x.shape  # [B, C, H, W] (여기서 C=1)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.sam.image_encoder.patch_embed(x)  # [B, H, W, C]
        B, H, W, C = x.shape
        raw_embed = self.sam.image_encoder.pos_embed
        if (raw_embed.shape[1], raw_embed.shape[2]) != (H, W):
            raw_embed = raw_embed.permute(0, 3, 1, 2)
            raw_embed = F.interpolate(raw_embed, size=(H, W), mode="bilinear", align_corners=False)
            raw_embed = raw_embed.permute(0, 2, 3, 1)
        pos_embed = raw_embed.expand(B, -1, -1, -1)
        x = x + pos_embed

        skips = []
        for i, blk in enumerate(self.sam.image_encoder.blocks):
            x = blk(x)  # [B, H, W, C]
            if i == 2:
                skips.append(x)
            elif i == 4:
                skips.append(x)
            elif i == 6:
                skips.append(x)
            elif i == 8:
                skips.append(x)

        # Project main feature to 128
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = self.projector(x)       # [B, 128, H, W]
        # Project skip features
        s4 = self.skip_proj4(skips[3].permute(0, 3, 1, 2))  # [B, 128, H, W]
        s3 = self.skip_proj3(skips[2].permute(0, 3, 1, 2))  # [B, 64, H, W]
        s2 = self.skip_proj2(skips[1].permute(0, 3, 1, 2))  # [B, 32, H, W]
        s1 = self.skip_proj1(skips[0].permute(0, 3, 1, 2))  # [B, 16, H, W]

        d4 = self.up4(x, s4)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        out = self.out_conv(d1)
        out = F.interpolate(out, size=input_shape[2:], mode='bilinear', align_corners=False)
        return out
