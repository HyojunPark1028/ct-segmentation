import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import vit_base_patch16_224


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
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


class TransUNet(nn.Module):
    def __init__(self, img_size=512, num_classes=1):
        super().__init__()
        # ViT backbone (timm pretrained)
        self.vit = vit_base_patch16_224(pretrained=True)
        self.vit.patch_size = 16
        self.vit.img_size = img_size
        self.vit.num_classes = 0

        self.hidden_dim = 768  # ViT output dim
        self.proj = nn.Conv2d(self.hidden_dim, 512, kernel_size=1)

        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 32)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        patch_h = patch_w = 16
        assert H % patch_h == 0 and W % patch_w == 0

        # ViT forward
        n_patches = (H // patch_h) * (W // patch_w)
        x_patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_h, p2=patch_w)
        x_vit = self.vit.forward_features(x_patches)

        # reshape transformer output to 2D feature map
        x_feat = x_vit.permute(0, 2, 1).contiguous()
        x_feat = x_feat.view(B, self.hidden_dim, H // patch_h, W // patch_w)
        x_feat = self.proj(x_feat)  # 512 channels

        # decoder
        d4 = self.decoder4(x_feat)
        d3 = self.decoder3(F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False))
        d2 = self.decoder2(F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False))
        d1 = self.decoder1(F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False))

        out = self.final_conv(F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False))
        return out
