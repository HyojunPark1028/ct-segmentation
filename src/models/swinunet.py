import torch
import torch.nn as nn
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
        # x: 이전 디코더 단계의 출력
        # skip: 인코더에서 오는 스킵 연결 특징
        
        # 업샘플링 수행
        x = self.up(x)
        
        # 업샘플링된 x와 스킵 연결 특징의 공간 해상도가 일치하지 않을 경우 보간
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        
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
        # 0: Stage 1 (Layer 0) 출력
        # 1: Stage 2 (Layer 1) 출력
        # 2: Stage 3 (Layer 2) 출력
        # 3: Stage 4 (Layer 3) 출력, 즉 병목 특징
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
        # features 리스트의 각 요소는 (B, C, H, W) 형태의 텐서입니다.
        # [0]: Stage 1 (128ch, H/4, W/4), [1]: Stage 2 (256ch, H/8, W/8),
        # [2]: Stage 3 (512ch, H/16, W/16), [3]: Stage 4 (1024ch, H/32, W/32)
        features = self.backbone.forward_features(x)
        
        # 각 스킵 연결 특징과 병목 특징 추출
        skip1 = features[0] # (B, 128, H/4, W/4)
        skip2 = features[1] # (B, 256, H/8, W/8)
        skip3 = features[2] # (B, 512, H/16, W/16)
        bottleneck = features[3] # (B, 1024, H/32, W/32)

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
