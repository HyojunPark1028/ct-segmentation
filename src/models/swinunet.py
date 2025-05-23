import torch
import torch.nn as nn
from timm.models.swin_transformer import swin_base_patch4_window7_224


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SwinUNet(nn.Module):
    def __init__(self, img_size=224, num_classes=1, use_pretrained=True):
        super().__init__()
        self.backbone = swin_base_patch4_window7_224(pretrained=use_pretrained)
        self.backbone.head = nn.Identity()

        self.proj4 = nn.Conv2d(1024, 384, kernel_size=1)
        self.proj3 = nn.Conv2d(512, 192, kernel_size=1)
        self.proj2 = nn.Conv2d(256, 96, kernel_size=1)
        self.proj1 = nn.Conv2d(128, 48, kernel_size=1)

        self.decoder3 = UpBlock(384, 192, 192)
        self.decoder2 = UpBlock(192, 96, 96)
        self.decoder1 = UpBlock(96, 48, 48)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(48, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        B = x.size(0)

        x = self.backbone.patch_embed(x)
        skip1 = self.backbone.layers[0](x)  # [B, H1, W1, 128]
        skip2 = self.backbone.layers[1](skip1)  # [B, H2, W2, 256]
        skip3 = self.backbone.layers[2](skip2)  # [B, H3, W3, 512]
        x = self.backbone.layers[3](skip3)      # [B, H4, W4, 1024]

        x = self.backbone.norm(x)
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        skip3 = skip3.permute(0, 3, 1, 2).contiguous()
        skip2 = skip2.permute(0, 3, 1, 2).contiguous()
        skip1 = skip1.permute(0, 3, 1, 2).contiguous()

        x = self.proj4(x)
        skip3 = self.proj3(skip3)
        skip2 = self.proj2(skip2)
        skip1 = self.proj1(skip1)

        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        return self.final(x)