import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import vit_base_patch16_224

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
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

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if skip.shape[2:] != x.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UTransVision(nn.Module):
    def __init__(self, img_size=224, num_classes=1, use_pretrained=True):
        super().__init__()
        self.encoder1 = ConvBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.encoder4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Vision Transformer
        self.transformer = vit_base_patch16_224(pretrained=use_pretrained)
        self.proj = nn.Conv2d(512, 768, kernel_size=1)
        self.norm = nn.LayerNorm(768)

        # Fixed Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, 768))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.decoder4 = UpBlock(768 + 512, 512, 256)
        self.decoder3 = UpBlock(256, 256, 128)
        self.decoder2 = UpBlock(128, 128, 64)
        self.decoder1 = UpBlock(64, 64, 32)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def interpolate_pos_encoding(self, x, H, W):
        N = self.pos_embed.shape[1]
        pos_embed = self.pos_embed.reshape(1, int(N**0.5), int(N**0.5), -1).permute(0, 3, 1, 2)  # [1, C, H, W]
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, H * W, -1)
        return pos_embed

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.pool1(x1))
        x3 = self.encoder3(self.pool2(x2))
        x4 = self.encoder4(self.pool3(x3))
        x_bottom = self.pool4(x4)  # [B, 512, H, W]

        B, C, H, W = x_bottom.shape
        x_t = self.proj(x_bottom).flatten(2).transpose(1, 2)  # [B, N, C]
        x_t = self.norm(x_t)

        pos_embed = self.interpolate_pos_encoding(x_t, H, W)
        x_t = x_t + pos_embed  # Interpolated positional embedding

        x_t = self.transformer.blocks(x_t)
        x_t = self.transformer.norm(x_t)
        x_t = x_t.transpose(1, 2).reshape(B, 768, H, W)  # [B, 768, H, W]

        x4_resized = F.interpolate(x4, size=(H, W), mode='bilinear', align_corners=False)
        x_fused = torch.cat([x_t, x4_resized], dim=1)  # Feature fusion

        d4 = self.decoder4(x_fused, x4_resized)
        d3 = self.decoder3(d4, x3)
        d2 = self.decoder2(d3, x2)
        d1 = self.decoder1(d2, x1)

        out = self.final_conv(d1)
        return out