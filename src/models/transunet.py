import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
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
    def __init__(self, img_size=224, num_classes=1):
        super().__init__()
        self.img_size = img_size

        # ResNet50 encoder (low-level feature extractor)
        resnet = resnet50(pretrained=False)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)          # 256
        self.encoder3 = resnet.layer2                                         # 512
        self.encoder4 = resnet.layer3                                         # 1024
        self.encoder5 = resnet.layer4                                         # 2048

        # Project to ViT input dim
        self.vit_proj = nn.Conv2d(2048, 768, kernel_size=1)

        # ViT backbone
        self.vit = vit_base_patch16_224(pretrained=False)
        self.hidden_dim = 768

        # Decoder
        self.decoder4 = DecoderBlock(768, 512)
        self.decoder3 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder1 = DecoderBlock(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Encoder (ResNet)
        e1 = self.encoder1(x)   # [B, 64, H/2, W/2]
        e2 = self.encoder2(e1)  # [B, 256, H/4, W/4]
        e3 = self.encoder3(e2)  # [B, 512, H/8, W/8]
        e4 = self.encoder4(e3)  # [B, 1024, H/16, W/16]
        e5 = self.encoder5(e4)  # [B, 2048, H/32, W/32]

        # ViT input
        x_vit = self.vit_proj(e5)  # [B, 768, H/32, W/32]
        B, C, H_v, W_v = x_vit.shape
        x_patches = rearrange(x_vit, 'b c h w -> b (h w) c')
        x_vit_out = self.vit.blocks(x_patches)
        x_vit_out = self.vit.norm(x_vit_out)
        x_feat = x_vit_out.permute(0, 2, 1).contiguous().view(B, C, H_v, W_v)

        # Decoder
        d4 = self.decoder4(x_feat)                     # [B, 512, H/32, W/32]
        d3 = self.decoder3(F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False))
        d2 = self.decoder2(F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False))
        d1 = self.decoder1(F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False))

        out = self.final_conv(F.interpolate(d1, scale_factor=4, mode='bilinear', align_corners=False))
        return out
