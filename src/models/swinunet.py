import torch
import torch.nn as nn
from timm.models.swin_transformer import swin_base_patch4_window7_224

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
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

        # print(f"x after up: {x.shape}, skip: {skip.shape}")

        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class SwinUNet(nn.Module):
    def __init__(self, img_size=224, num_classes=1, use_pretrained=True):
        super().__init__()
        self.backbone = swin_base_patch4_window7_224(pretrained=use_pretrained)
        self.backbone.head = nn.Identity()

        # skip connection stage output channels
        self.decoder4 = UpBlock(1024, 512, 512)
        self.decoder3 = UpBlock(512, 256, 256)
        self.decoder2 = UpBlock(256, 128, 128)
        self.decoder1 = UpBlock(128, 96, 64)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 112→224
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        B = x.size(0)

        x = self.backbone.patch_embed(x)  # Patch Embedding

        skip1 = self.backbone.layers[0](x)  # [B, H1, W1, 96]
        skip2 = self.backbone.layers[1](skip1)  # [B, H2, W2, 192]
        skip3 = self.backbone.layers[2](skip2)  # [B, H3, W3, 384]
        x = self.backbone.layers[3](skip3)      # [B, H4, W4, 768]

        x = self.backbone.norm(x)               # [B, H, W, C]
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        skip3 = skip3.permute(0, 3, 1, 2).contiguous()  # [B, 384, H3, W3]
        skip2 = skip2.permute(0, 3, 1, 2).contiguous()  # [B, 192, H2, W2]
        skip1 = skip1.permute(0, 3, 1, 2).contiguous()  # [B, 96, H1, W1]
        orig = x.new_zeros(B, 96, skip1.shape[2] * 2, skip1.shape[3] * 2)  # dummy upsample

        # print(f"skip3 shape: {skip3.shape}")  # ⬅ 꼭 확인
        # print(f"x shape before decoder4: {x.shape}")  # ⬅ 여기서 1024 채널인지

        x = self.decoder4(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder2(x, skip1)
        x = self.decoder1(x, orig)

        return self.final(x)

