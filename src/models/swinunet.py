import torch
import torch.nn as nn
import torch.nn.functional as F # F.interpolate 사용을 위해 임포트
from timm.models.swin_transformer import swin_base_patch4_window7_224

# 사용자께서 제공하신 ConvBlock 클래스
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

# 사용자께서 제공하신 UpBlock 클래스
class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # 전치 컨볼루션 (Transpose Convolution)을 통한 업샘플링
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # 업샘플링된 특징과 스킵 연결 특징을 결합하여 처리하는 컨볼루션 블록
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        # x: 이전 디코더 단계의 출력 (B, C, H, W)
        # skip: 인코더에서 오는 스킵 연결 특징 (B, C, H_skip, W_skip)
        
        # 업샘플링 수행
        x = self.up(x)
        
        # 업샘플링된 x와 스킵 연결 특징의 공간 해상도가 일치하지 않을 경우 보간
        # Conv2d는 (B, C, H, W) 형태를 기대하므로, 이 형태를 유지한 채 보간합니다.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        
        # 채널 차원(dim=1)을 따라 x와 skip을 연결
        x = torch.cat([x, skip], dim=1)
        
        # 연결된 특징을 컨볼루션 블록으로 처리
        return self.conv(x)

# SwinUNet 모델 정의
class SwinUNet(nn.Module):
    def __init__(self, img_size=224, num_classes=1, use_pretrained=True):
        super().__init__()
        self.img_size = img_size 
        
        # timm 라이브러리의 사전 학습된 Swin Transformer 백본 사용
        # _out_indices를 설정하여 각 단계의 특징 맵을 반환하도록 구성
        # timm Swin Transformer의 forward_features는 (B, H, W, C) 형태로 반환합니다.
        # 따라서 이후 Conv2d에 전달하기 위해 permute가 필요합니다.
        self.backbone = swin_base_patch4_window7_224(pretrained=use_pretrained, _out_indices=(0, 1, 2, 3))
        # 분류 헤드 제거 (분할 작업에 필요 없음)
        self.backbone.head = nn.Identity()

        # 인코더에서 추출된 특징의 채널을 디코더에 맞게 조정하는 1x1 Conv2d 계층들
        # 이들은 (B, C, H, W) 형태의 입력을 받습니다.
        self.proj4 = nn.Conv2d(1024, 384, kernel_size=1) # 병목 특징 투영 (1024ch -> 384ch)
        self.proj3 = nn.Conv2d(512, 192, kernel_size=1) # 스킵 연결 (Layer 2) 투영 (512ch -> 192ch)
        self.proj2 = nn.Conv2d(256, 96, kernel_size=1)  # 스킵 연결 (Layer 1) 투영 (256ch -> 96ch)
        self.proj1 = nn.Conv2d(128, 48, kernel_size=1)  # 스킵 연결 (Layer 0) 투영 (128ch -> 48ch)

        # 디코더 단계 정의 (사용자께서 제공하신 UpBlock 사용)
        # UpBlock(in_channels, skip_channels, out_channels)
        # 각 UpBlock은 이전 단계의 출력과 해당 스킵 연결을 받습니다.
        self.decoder3 = UpBlock(384, 192, 192) # 384ch (병목) + 192ch (skip3) -> 192ch
        self.decoder2 = UpBlock(192, 96, 96)   # 192ch (decoder3) + 96ch (skip2) -> 96ch
        self.decoder1 = UpBlock(96, 48, 48)    # 96ch (decoder2) + 48ch (skip1) -> 48ch

        # 최종 출력 계층
        # 최종 업샘플링 (4배) 및 1x1 컨볼루션 (클래스 수에 맞게 채널 조정)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(48, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # 입력 이미지 전처리: 단일 채널 이미지를 3채널로 복제 (ImageNet 사전 학습 모델에 맞춤)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1) # (B, 3, H, W)

        # timm 백본의 forward_features를 호출하여 각 단계의 특징 맵 리스트를 얻음
        # features 리스트의 각 요소는 (B, H*W, C) 형태의 텐서입니다.
        features = self.backbone.forward_features(x)
        
        # 각 스킵 연결 특징과 병목 특징 추출
        # Swin Transformer의 각 Stage 출력은 (B, L, C) 형태이므로, (B, H, W, C)로 reshape 후 permute해야 합니다.
        
        # img_size = 224, patch_size = 4
        # Stage 1 (skip1): H/4, W/4 = 56x56
        H_s1, W_s1 = self.img_size // 4, self.img_size // 4
        skip1 = features[0].view(x.size(0), H_s1, W_s1, -1).permute(0, 3, 1, 2).contiguous() # (B, 128, H/4, W/4)

        # Stage 2 (skip2): H/8, W/8 = 28x28
        H_s2, W_s2 = self.img_size // 8, self.img_size // 8
        skip2 = features[1].view(x.size(0), H_s2, W_s2, -1).permute(0, 3, 1, 2).contiguous() # (B, 256, H/8, W/8)

        # Stage 3 (skip3): H/16, W/16 = 14x14
        H_s3, W_s3 = self.img_size // 16, self.img_size // 16
        skip3 = features[2].view(x.size(0), H_s3, W_s3, -1).permute(0, 3, 1, 2).contiguous() # (B, 512, H/16, W/16)

        # Stage 4 (bottleneck): H/32, W/32 = 7x7
        H_s4, W_s4 = self.img_size // 32, self.img_size // 32
        bottleneck = features[3].view(x.size(0), H_s4, W_s4, -1).permute(0, 3, 1, 2).contiguous() # (B, 1024, H/32, W/32)

        # proj 계층 (nn.Conv2d)을 통해 채널 수 조정
        proj_bottleneck = self.proj4(bottleneck)
        proj_skip3 = self.proj3(skip3)
        proj_skip2 = self.proj2(skip2)
        proj_skip1 = self.proj1(skip1)

        # 디코더 경로 (UpBlock 사용)
        # 각 UpBlock은 (B, C, H, W) 형태를 입력받고 출력합니다.
        x = self.decoder3(proj_bottleneck, proj_skip3) # (B, 192, H/16, W/16)
        x = self.decoder2(x, proj_skip2) # (B, 96, H/8, W/8)
        x = self.decoder1(x, proj_skip1) # (B, 48, H/4, W/4)

        # 최종 출력 계층 적용
        x = self.final(x)

        return x
