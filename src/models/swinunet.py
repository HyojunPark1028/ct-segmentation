import torch
import torch.nn as nn
from timm.models.swin_transformer import swin_base_patch4_window7_224
from einops import rearrange

class PatchExpanding(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2 * input_dim)
        self.norm = nn.LayerNorm(2 * input_dim)  # 반드시 2 * input_dim과 일치
        self.output_dim = input_dim // 2

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x = self.linear(x)                      # (B, H, W, 2C)
        x = self.norm(x)                        # 마지막 차원이 2C와 일치해야 함
        x = rearrange(x, 'b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)',
                      p1=2, p2=2, c_out=self.output_dim)
        return x


class SwinDecoderBlock(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim):
        super().__init__()
        self.up = PatchExpanding(in_dim)
        self.concat_proj = nn.Conv2d((in_dim // 2) + skip_dim, out_dim, kernel_size=1)
        self.block = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.concat_proj(x)
        return self.block(x)

class SwinUNet(nn.Module):
    def __init__(self, img_size=224, num_classes=1, use_pretrained=True):
        super().__init__()
        self.backbone = swin_base_patch4_window7_224(pretrained=use_pretrained)
        self.proj4 = nn.Conv2d(1024, 384, kernel_size=1)
        self.proj3 = nn.Conv2d(512, 192, kernel_size=1)
        self.proj2 = nn.Conv2d(256, 96, kernel_size=1)
        self.proj1 = nn.Conv2d(128, 48, kernel_size=1)

        self.decoder3 = SwinDecoderBlock(384, 192, 192)
        self.decoder2 = SwinDecoderBlock(192, 96, 96)
        self.decoder1 = SwinDecoderBlock(96, 48, 48)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(48, num_classes, kernel_size=1)
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Swin Transformer forward
        B = x.size(0)
        x = self.backbone.patch_embed(x)
        H, W = self.backbone.patch_embed.grid_size
        x = x.view(B, H, W, -1)

        skip1 = self.backbone.layers[0](x)
        H1, W1 = H // 2, W // 2
        skip1 = skip1.view(B, H1, W1, -1).permute(0, 3, 1, 2)

        skip2 = self.backbone.layers[1](skip1.permute(0, 2, 3, 1))
        H2, W2 = H1 // 2, W1 // 2
        skip2 = skip2.view(B, H2, W2, -1).permute(0, 3, 1, 2)

        skip3 = self.backbone.layers[2](skip2.permute(0, 2, 3, 1))
        H3, W3 = H2 // 2, W2 // 2
        skip3 = skip3.view(B, H3, W3, -1).permute(0, 3, 1, 2)

        bottleneck = self.backbone.layers[3](skip3.permute(0, 2, 3, 1))
        H4, W4 = H3 // 2, W3 // 2
        bottleneck = bottleneck.view(B, H4, W4, -1).permute(0, 3, 1, 2)

        x = self.proj4(bottleneck)
        skip3 = self.proj3(skip3)
        skip2 = self.proj2(skip2)
        skip1 = self.proj1(skip1)

        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        return self.final(x)

