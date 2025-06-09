# src/models/medsam_gan.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.build_sam import sam_model_registry
import segmentation_models_pytorch as smp
from typing import Optional
import os

# --- 1. Discriminator 네트워크 정의 ---
class MaskDiscriminator(nn.Module):
    """
    생성된 마스크의 '진위'를 판별하는 Discriminator 네트워크.
    PatchGAN과 유사한 구조로, 입력 이미지(RGB)와 마스크를 결합하여 받습니다.
    """
    def __init__(self, in_channels: int = 4): # 3 (RGB image) + 1 (mask) = 4 channels
        super().__init__()
        # Discriminator 네트워크 정의 (PatchGAN-like)
        # 각 Conv2d 후 BatchNorm2d와 LeakyReLU를 적용합니다.
        # 마지막 Conv2d는 1채널 출력 (각 패치의 '진위' 점수)
        self.net = nn.Sequential(
            # Input: (B, in_channels, 256, 256)
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1), # Output: (B, 64, 128, 128)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # Output: (B, 128, 64, 64)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # Output: (B, 256, 32, 32)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # Output: (B, 512, 16, 16)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1) # Output: (B, 1, 15, 15)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --- 2. U-Net 모델 정의 (Segmentation_models_pytorch 라이브러리 사용) ---
class CustomUNet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None # 활성화 함수는 나중에 손실 함수에서 적용
        )

    def forward(self, x):
        return self.model(x)

# --- 3. MedSAM 모델 정의 (SAM Backbone + UNet Decoder) ---
class MedSAM(nn.Module):
    """
    MedSAM 모델: SAM의 이미지 인코더에 UNet 디코더를 연결한 구조.
    """
    def __init__(self, checkpoint, model_type, output_channels=1, unet_checkpoint=None):
        super().__init__()
        # SAM 모델 로드
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.image_encoder = self.sam.image_encoder
        self.prompt_encoder = self.sam.prompt_encoder
        self.mask_decoder = self.sam.mask_decoder # SAM의 원래 마스크 디코더 사용

        # UNet 디코더 (GAN 학습에서 Generator의 초기 출력 개선에 사용될 수 있음)
        # 현재 코드에서는 SAM의 마스크 디코더를 직접 사용하고 있어, UNet은 다른 용도 (예: 초기 세그멘테이션)로 활용될 수 있음.
        # 여기서는 MedSAM_GAN 클래스에서 SAM의 마스크 디코더와 연계하여 사용될 예정.
        # in_channels는 1 (CT 이미지 채널)로 변경합니다.
        self.unet_decoder = CustomUNet(in_channels=1, classes=output_channels) 

        # UNet 체크포인트 로드 (선택 사항)
        if unet_checkpoint:
            if not os.path.exists(unet_checkpoint):
                raise FileNotFoundError(f"UNet checkpoint not found: {unet_checkpoint}")
            
            try:
                unet_state_dict = torch.load(unet_checkpoint, map_location='cpu')
                
                # ⭐⭐ 수정: 로드된 state_dict의 키에서 "model." 접두사를 제거합니다. ⭐⭐
                new_state_dict = {k.replace('model.', ''): v for k, v in unet_state_dict.items()}
                
                # CustomUNet은 내부에 `self.model`이라는 smp.Unet 인스턴스를 가지고 있습니다.
                # 따라서 로드된 state_dict를 `self.unet_decoder.model`에 로드해야 합니다.
                self.unet_decoder.model.load_state_dict(new_state_dict, strict=True)
                
                print(f"Loaded UNet checkpoint from {unet_checkpoint}")
            except Exception as e:
                print(f"Error loading UNet checkpoint: {e}")
                # UNet 로드 실패 시, 메시지를 출력하고 학습은 계속 진행 (초기화된 UNet 사용)
                pass # UNet 로드에 실패해도 전체 모델 학습이 가능하도록 예외 처리


    def forward(self, image: torch.Tensor, prompt_boxes: Optional[torch.Tensor] = None):
        # MedSAM의 forward는 이미지와 프롬프트 (여기서는 박스)를 받아 마스크를 생성.
        # 이 함수는 SAM의 인코더-디코더 흐름을 따름.

        # ⭐⭐ 수정: SAM 이미지 인코더는 3채널을 기대하므로 1채널 이미지를 복제합니다. ⭐⭐
        if image.shape[1] == 1:
            image_for_sam_encoder = image.repeat(1, 3, 1, 1) # (B, 3, H, W)
        else:
            image_for_sam_encoder = image

        # 1. 이미지 인코더 (SAM Image Encoder)
        image_embedding = self.image_encoder(image_for_sam_encoder) # 수정된 입력 사용

        # 2. 프롬프트 인코더 (SAM Prompt Encoder)
        # prompt_boxes가 제공되지 않으면, 이미지 전체를 커버하는 박스 생성
        if prompt_boxes is None:
            image_h, image_w = image_for_sam_encoder.shape[-2:] # 3채널 이미지의 크기 사용
            prompt_boxes = torch.tensor([[0, 0, image_w, image_h]], dtype=torch.float, device=image.device).repeat(image.shape[0], 1, 1)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=prompt_boxes,
            masks=None,
        )

        # 3. 마스크 디코더 (SAM Mask Decoder)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(), # 이미지 위치 인코딩
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False # 단일 마스크 출력
        )

        # 최종 마스크를 원본 이미지 크기로 업샘플링
        masks = F.interpolate(
            low_res_masks, size=image.shape[-2:], mode='bilinear', align_corners=False
        )
        
        return masks, iou_predictions, low_res_masks # (생성된 마스크, iou_예측, 저해상도 마스크)

# --- 4. MedSAM_GAN 통합 모델 정의 ---
class MedSAM_GAN(nn.Module):
    def __init__(self, sam_checkpoint, unet_checkpoint, out_channels=1):
        super().__init__()
        # MedSAM (Generator 역할)
        self.sam = MedSAM(
            checkpoint=sam_checkpoint,
            model_type="vit_b",
            output_channels=out_channels,
            unet_checkpoint=unet_checkpoint
        )
        # Discriminator
        self.discriminator = MaskDiscriminator(in_channels=4) # 3 (RGB image) + 1 (mask) = 4 channels

    # ⭐ 수정: real_low_res_mask를 키워드 전용 인자로 만듭니다.
    def forward(self, image: torch.Tensor, *, real_low_res_mask: Optional[torch.Tensor] = None):
        # original_image_size = image.shape[-2:] # (H, W) # 사용되지 않아 주석 처리

        # CT 이미지 전처리 및 SAM 인코더 통과 (MedSAM 내부에서 3채널로 변환됨)

        # 1. Generator (SAM)를 통해 마스크 생성
        # MedSAM.forward는 (생성된 마스크_1024, iou_predictions, 저해상도 마스크_256)를 반환
        # MedSAM은 prompt_boxes를 받으므로, 여기에 이미지 전체를 커버하는 박스 프롬프트 전달
        image_h, image_w = image.shape[-2:] # 원본 이미지 크기 사용
        input_box = torch.tensor([[0, 0, image_w, image_h]], dtype=torch.float, device=image.device).repeat(image.shape[0], 1, 1)

        # self.sam.forward 호출 시 image와 prompt_boxes를 전달
        masks_1024_gen, iou_predictions_gen, low_res_masks_256_gen = self.sam(image, prompt_boxes=input_box)


        # 2. Discriminator 입력용 이미지 준비
        # CT 1채널 이미지를 Discriminator 입력에 맞게 3채널 RGB처럼 복제
        resized_image_rgb_for_D = F.interpolate(
            image.float(), size=(256, 256), mode='bilinear', align_corners=False
        ).repeat(1, 3, 1, 1)

        # 3. 생성된 저해상도 마스크에 대한 Discriminator 출력
        # Discriminator 입력: [리사이즈된 원본 이미지(RGB), 생성된 저해상도 마스크(1채널)]
        discriminator_input_for_generated = torch.cat([resized_image_rgb_for_D, low_res_masks_256_gen], dim=1)
        discriminator_output_for_generated_mask = self.discriminator(discriminator_input_for_generated)

        # 4. 실제 마스크에 대한 Discriminator 출력 (real_low_res_mask가 제공될 경우)
        discriminator_output_for_real_mask = None
        if real_low_res_mask is not None:
            # Discriminator 입력: [리사이즈된 원본 이미지(RGB), 실제 저해상도 마스크(1채널)]
            discriminator_input_for_real = torch.cat([resized_image_rgb_for_D, real_low_res_mask], dim=1)
            discriminator_output_for_real_mask = self.discriminator(discriminator_input_for_real)
        
        # 반환 값 순서: (Generator 생성 마스크, IoU 예측, 생성 마스크에 대한 D 출력, 저해상도 마스크, 실제 마스크에 대한 D 출력)
        return masks_1024_gen, iou_predictions_gen, discriminator_output_for_generated_mask, low_res_masks_256_gen, discriminator_output_for_real_mask