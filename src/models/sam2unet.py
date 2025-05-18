import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("/content/ct-segmentation/sam2")
from sam2.build_sam import build_sam2

class Adapter(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.down = nn.Conv2d(channels, channels // reduction_ratio, 1)
        self.up = nn.Conv2d(channels // reduction_ratio, channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.up(self.relu(self.down(x)))

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class SAM2UNet(nn.Module):
    def __init__(self, checkpoint: str, config: str = "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml", in_channels=1, out_channels=1):
        super().__init__()
        sam2 = build_sam2(config, checkpoint)
        self.encoder = sam2.image_encoder.trunk

        # LayerNorm은 학습 가능하도록 유지
        for name, p in self.encoder.named_parameters():
            if "norm" in name or "ln" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

        # Adapter 삽입 (stage output 이후)
        self.adapters = nn.ModuleList([
            Adapter(112), Adapter(224), Adapter(448), Adapter(896)
        ])

        self.project_bottleneck = nn.Conv2d(896, 256, kernel_size=1)
        self.project_skips = nn.ModuleList([
            nn.Conv2d(448, 256, kernel_size=1),
            nn.Conv2d(224, 256, kernel_size=1),
            nn.Conv2d(112, 256, kernel_size=1)
        ])

        self.up_blocks = nn.ModuleList([
            UpBlock(256, 256, 256),
            UpBlock(256, 256, 256),
            UpBlock(256, 256, 128)
        ])

        self.out_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        feats = self.encoder(x)  # [f0, f1, f2, f3]
        assert len(feats) == 4

        # Adapter 통과
        feats = [adapt(f) for adapt, f in zip(self.adapters, feats)]

        x = self.project_bottleneck(feats[-1])
        d4 = self.up_blocks[0](x, self.project_skips[0](feats[2]))
        d3 = self.up_blocks[1](d4, self.project_skips[1](feats[1]))
        d2 = self.up_blocks[2](d3, self.project_skips[2](feats[0]))

        out = self.out_conv(d2)

        # ⛳ 수정된 부분: 잘못된 size 리스트 반복 대신 명시적 튜플
        h, w = x.shape[2:]
        out = F.interpolate(out, size=(h * 4, w * 4), mode='bilinear', align_corners=False)

        # Deep supervision: d2, d3, d4도 반환
        return out, d4, d3, d2