import torch
import torch.nn as nn
from timm.models.swin_transformer import swin_base_patch4_window7_224
from einops import rearrange # einops 라이브러리 임포트

# Patch Expanding 모듈 정의
# SwinUNet 논문의 Patch Expanding Layer를 구현합니다.
# 이 계층은 해상도를 2배 높이고 채널 차원을 조절합니다.
class PatchExpanding(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 입력 특징의 채널을 2배로 늘리는 선형 계층
        # 예: C -> 2C
        self.linear = nn.Linear(input_dim, 2 * input_dim)
        # 채널 정규화 계층
        self.norm = nn.LayerNorm(2 * input_dim)
        # rearrange 연산 후의 출력 채널 (원본 채널의 절반)
        # 예: 2C -> C/2
        self.output_dim = input_dim // 2

    def forward(self, x):
        # x의 형태: (Batch_size, Height, Width, Channels)
        B, H, W, C_in = x.shape
        
        # 선형 변환 적용
        x = self.linear(x) # (B, H, W, 2*C_in)
        # 정규화 적용
        x = self.norm(x)

        # rearrange 연산을 통해 공간 해상도를 2배 확장하고 채널을 1/4로 줄임
        # 'p1', 'p2'는 확장 인자 (여기서는 2x2 확장)
        # 'c_out'은 rearrange 후의 출력 채널
        x = rearrange(x, 'b h w (p1 p2 c_out) -> b (h p1) (w p2) c_out', p1=2, p2=2, c_out=self.output_dim)
        return x

# 디코더의 각 단계를 정의하는 모듈
# 기존의 ConvBlock과 UpBlock을 대체하여 트랜스포머 기반의 디코더를 구성합니다.
class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels_after_stage):
        super().__init__()
        # Patch Expanding 계층 초기화
        # 이전 디코더 단계/병목에서 오는 특징을 업샘플링합니다.
        self.patch_expanding = PatchExpanding(in_channels)

        # Patch Expanding 후의 채널과 스킵 연결 채널을 합친 후,
        # 다음 단계의 Swin Transformer 블록에 전달하기 위한 채널로 투영하는 선형 계층
        # (in_channels // 2) + skip_channels -> out_channels_after_stage
        self.concat_proj = nn.Linear((in_channels // 2) + skip_channels, out_channels_after_stage)
        # 정규화 및 활성화 함수
        self.norm = nn.LayerNorm(out_channels_after_stage)
        self.gelu = nn.GELU()

        # 스킵 연결 후 특징 변환을 위한 간단한 MLP-like 구조
        # 원본 논문에서는 이 부분에 Swin Transformer 블록이 사용되지만,
        # timm 내부 클래스를 직접 가져오기 어려워 선형 변환으로 대체합니다.
        self.transform_after_concat = nn.Sequential(
            nn.Linear(out_channels_after_stage, out_channels_after_stage),
            nn.GELU(),
            nn.LayerNorm(out_channels_after_stage)
        )

    def forward(self, x, skip):
        # x: 이전 디코더 단계 또는 병목에서 오는 특징 (B, H, W, C_x)
        # skip: 인코더에서 오는 스킵 연결 특징 (B, H_skip, W_skip, C_skip)
        
        x = self.patch_expanding(x) # 출력: (B, 2H, 2W, in_channels // 2)

        # 스킵 연결 특징과 공간 차원이 일치하는지 확인 (안전 장치)
        if x.shape[-3:-1] != skip.shape[-3:-1]:
            # 공간 차원이 일치하지 않으면 보간 (일반적으로 발생하지 않아야 함)
            x_permuted = x.permute(0, 3, 1, 2) # (B, C, H, W)로 변환하여 보간
            x_interpolated = nn.functional.interpolate(x_permuted, size=skip.shape[-3:-1], mode="bilinear", align_corners=False)
            x = x_interpolated.permute(0, 2, 3, 1) # 다시 (B, H, W, C)로 변환

        # 채널 차원을 따라 특징 연결
        x = torch.cat([x, skip], dim=-1) # 마지막 차원(채널)을 따라 연결

        # 연결된 특징에 선형 투영, 정규화, 활성화 함수 적용
        x = self.concat_proj(x)
        x = self.norm(x)
        x = self.gelu(x)
        # 추가 변환 적용
        x = self.transform_after_concat(x)

        return x

# SwinUNet 모델 정의
class SwinUNet(nn.Module):
    def __init__(self, img_size=224, num_classes=1, use_pretrained=True):
        super().__init__()
        self.img_size = img_size # img_size를 저장하여 H, W 계산에 사용
        # timm 라이브러리의 사전 학습된 Swin Transformer 백본 사용
        self.backbone = swin_base_patch4_window7_224(pretrained=use_pretrained)
        # 분류 헤드 제거 (분할 작업에 필요 없음)
        self.backbone.head = nn.Identity()

        # 인코더에서 추출된 특징의 채널을 디코더에 맞게 조정하는 선형 투영 계층
        # Swin-Base의 각 단계별 출력 채널: 128, 256, 512, 1024
        self.proj4 = nn.Linear(1024, 384) # 병목 특징 투영
        self.proj3 = nn.Linear(512, 192) # 스킵 연결 (Layer 2)
        self.proj2 = nn.Linear(256, 96)  # 스킵 연결 (Layer 1)
        self.proj1 = nn.Linear(128, 48)  # 스킵 연결 (Layer 0)

        # DecoderStage를 사용하여 디코더 단계 정의
        # 각 DecoderStage는 PatchExpanding과 특징 변환을 포함합니다.
        # Decoder 3: 입력 (병목) 384 채널, 스킵 연결 192 채널 -> 출력 192 채널
        self.decoder3 = DecoderStage(in_channels=384, skip_channels=192, out_channels_after_stage=192)
        # Decoder 2: 입력 (이전 디코더 출력) 192 채널, 스킵 연결 96 채널 -> 출력 96 채널
        self.decoder2 = DecoderStage(in_channels=192, skip_channels=96, out_channels_after_stage=96)
        # Decoder 1: 입력 (이전 디코더 출력) 96 채널, 스킵 연결 48 채널 -> 출력 48 채널
        self.decoder1 = DecoderStage(in_channels=96, skip_channels=48, out_channels_after_stage=48)

        # 최종 출력 계층
        # 디코더의 마지막 출력은 (B, H, W, 48) 형태
        # 최종 업샘플링 (예: 56x56 -> 224x224)
        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        # 최종 컨볼루션 계층 (클래스 수에 맞게 채널 조정)
        self.final_conv = nn.Conv2d(48, num_classes, kernel_size=1)

    def forward(self, x):
        # 입력 이미지 전처리: 단일 채널 이미지를 3채널로 복제 (ImageNet 사전 학습 모델에 맞춤)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1) # (B, 3, H, W)

        B_orig = x.size(0)
        
        # Patch embedding 후의 초기 공간 해상도 계산
        # swin_base_patch4_window7_224의 patch_size는 (4, 4)
        H, W = self.img_size // self.backbone.patch_embed.patch_size[0], \
               self.img_size // self.backbone.patch_embed.patch_size[1]
        
        # Patch embedding
        # 출력: (B, num_patches, embed_dim)
        x = self.backbone.patch_embed(x) 

        # 인코더 계층 및 스킵 연결
        # 각 layer(BasicLayer)는 내부적으로 PatchMerging을 수행하여 H와 W를 절반으로 줄입니다.
        # 출력 x는 (B, num_patches_new, C_new) 형태입니다.
        
        # Layer 0 (Stage 1)
        x = self.backbone.layers[0](x)
        # Layer 0 후의 H와 W 계산: H_prev -> (H_prev + 1) // 2
        H0, W0 = (H + 1) // 2, (W + 1) // 2 
        # 스킵 연결을 위해 (B, H, W, C) 형태로 재구성
        skip1 = x.view(B_orig, H0, W0, -1) 

        # Layer 1 (Stage 2)
        x = self.backbone.layers[1](x)
        # Layer 1 후의 H와 W 계산
        H1, W1 = (H0 + 1) // 2, (W0 + 1) // 2 
        skip2 = x.view(B_orig, H1, W1, -1)

        # Layer 2 (Stage 3)
        x = self.backbone.layers[2](x)
        # Layer 2 후의 H와 W 계산
        H2, W2 = (H1 + 1) // 2, (W1 + 1) // 2 
        skip3 = x.view(B_orig, H2, W2, -1)

        # Layer 3 (Stage 4 - 병목)
        x = self.backbone.layers[3](x)
        # Layer 3 후의 H와 W 계산
        H3, W3 = (H2 + 1) // 2, (W2 + 1) // 2 
        x = x.view(B_orig, H3, W3, -1)

        # 병목 특징에 정규화 적용
        x = self.backbone.norm(x)

        # 디코더 입력 및 스킵 연결 특징에 채널 투영 적용
        x_bottleneck = self.proj4(x)
        skip3_proj = self.proj3(skip3)
        skip2_proj = self.proj2(skip2)
        skip1_proj = self.proj1(skip1)

        # 디코더 경로
        x = self.decoder3(x_bottleneck, skip3_proj) # 출력: (B, H2, W2, 192)
        x = self.decoder2(x, skip2_proj) # 출력: (B, H1, W1, 96)
        x = self.decoder1(x, skip1_proj) # 출력: (B, H0, W0, 48)

        # 최종 출력 전 특징 맵 형태 변경 (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()

        # 최종 업샘플링 및 컨볼루션
        x = self.final_upsample(x)
        x = self.final_conv(x)

        return x
