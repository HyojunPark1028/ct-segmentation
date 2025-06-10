# src/models/medsam_gan.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.build_sam import sam_model_registry
import segmentation_models_pytorch as smp
from typing import Optional, Tuple
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

# --- 3. MedSAM 모델 정의 (SAM Backbone + UNet for initial prompt generation) ---
class MedSAM(nn.Module):
    """
    MedSAM 모델: SAM의 이미지 인코더에 UNet을 결합하여 초기 마스크 프롬프트를 생성하고,
    SAM의 마스크 디코더로 최종 마스크를 예측하는 구조.
    """
    def __init__(self, checkpoint: str, model_type: str, out_channels: int = 1, unet_checkpoint: Optional[str] = None):
        super().__init__()
        # SAM 모델 로드
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.image_encoder = self.sam.image_encoder
        self.prompt_encoder = self.sam.prompt_encoder
        self.mask_decoder = self.sam.mask_decoder # SAM의 원래 마스크 디코더 사용

        # UNet 모델 초기화 및 가중치 로드
        # UNet은 SAM의 마스크 디코더를 위한 초기 프롬프트(마스크)를 생성하는 용도로 사용됩니다.
        self.unet_decoder = CustomUNet(in_channels=1, classes=out_channels) 

        # UNet 체크포인트 로드 (선택 사항)
        if unet_checkpoint:
            if not os.path.exists(unet_checkpoint):
                print(f"Warning: UNet checkpoint not found at: {unet_checkpoint}. UNet will be initialized from scratch.")
            else:
                try:
                    unet_state_dict = torch.load(unet_checkpoint, map_location='cpu')
                    # ⭐⭐ 핵심 수정: 로드된 state_dict의 키에서 "model." 접두사를 제거합니다. ⭐⭐
                    new_state_dict = {k.replace('model.', ''): v for k, v in unet_state_dict.items()}
                    
                    self.unet_decoder.model.load_state_dict(new_state_dict, strict=True)
                    
                    print(f"Loaded UNet checkpoint from {unet_checkpoint}")
                except Exception as e:
                    print(f"Error loading UNet checkpoint: {e}. UNet will be initialized from scratch.")
                    # UNet 로드에 실패해도 전체 모델 학습이 가능하도록 예외 처리
        else:
            print(f"UNet checkpoint path not provided. Starting U-Net from scratch (or imagenet if specified).")
            # 만약 UNet이 imagenet weights로 시작해야 한다면, 위 smp.Unet 초기화 시 encoder_weights="imagenet" 설정.

        # ⭐ UNet 가중치 고정 (medsam.py의 의도를 유지) ⭐
        # UNet은 초기 프롬프트 생성용이며, 이 가중치는 학습되지 않도록 고정합니다.
        self.unet_decoder.eval() # 평가 모드로 설정
        for p in self.unet_decoder.parameters():
            p.requires_grad = False # 학습 비활성화

    def forward(self, image: torch.Tensor, prompt_boxes: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # image: NpySegDataset에서 넘어온 텐서. normalize_type="sam"에 의해 [0, 255] 스케일.

        # 1. SAM 이미지 인코더에 입력하기 위한 이미지 준비: 3채널로 복제
        if image.shape[1] == 1:
            image_for_sam_encoder = image.repeat(1, 3, 1, 1) # (B, 3, H, W)
        else:
            image_for_sam_encoder = image

        # ⭐ SAM 이미지 인코더 입력 스케일: Dataset에서 이미 [0, 255]이므로 추가 스케일링 필요 없음 ⭐
        # image_for_sam_encoder는 이미 위에서 정의되었으므로 이 라인은 불필요. (이전 오류 원인 중 하나)
        # image_for_sam_encoder = image_rgb 
        
        # SAM 이미지 인코더를 통해 이미지 임베딩 추출
        image_embedding = self.image_encoder(image_for_sam_encoder)

        # SAM의 위치 인코딩
        # ⭐⭐ 최종 수정: image_pe 배치 차원 일치 및 디바이스 설정 ⭐⭐
        # get_dense_pe()는 일반적으로 (1, C, H, W) 형태를 반환하므로,
        # 현재 배치 크기 (image_embedding.shape[0])에 맞게 반복(repeat) 해 주어야 합니다.
        image_pe = self.sam.prompt_encoder.get_dense_pe()
        batch_size = image_embedding.shape[0]
        # image_pe가 이미 배치 크기와 같거나, 배치 크기가 1이고 현재 배치 크기가 1인 경우 반복하지 않음.
        # 그 외 (image_pe가 1이고 현재 배치 크기가 > 1)일 경우 반복.
        if image_pe.shape[0] == 1 and batch_size > 1: 
            image_pe = image_pe.repeat(batch_size, 1, 1, 1) # Repeat along batch dimension
        # Ensure image_pe is on the same device as image_embedding
        image_pe = image_pe.to(image_embedding.device) 


        # Step 2: UNet을 통한 초기 마스크 예측 (프롬프트 용도)
        # NpySegDataset에서 [0, 255] 범위로 정규화된 이미지를 받으므로,
        # UNet이 학습되었던 [0, 1] 범위로 역스케일링합니다.
        image_for_unet = image / 255.0 # [0, 255] -> [0, 1]

        with torch.no_grad(): # UNet은 고정되었으므로 no_grad 블록에서 실행
            # ⭐⭐ UNet (self.unet_decoder) 사용 ⭐⭐
            initial_mask_logits = self.unet_decoder(image_for_unet) 
            # U-Net의 출력을 이진화하지 않고 확률맵으로 프롬프트에 사용 (권장)


        # Step 3: 초기 마스크 또는 박스를 SAM 프롬프트로 변환
        sparse_prompt_embeddings = None
        dense_prompt_embeddings = None

        if prompt_boxes is not None:
            # 박스 프롬프트가 제공되면 사용
            sparse_prompt_embeddings, dense_prompt_embeddings = self.sam.prompt_encoder(
                points=None,
                boxes=prompt_boxes,
                masks=None,
            )
        else:
            # 박스 프롬프트가 없으면 U-Net으로 생성된 마스크를 프롬프트로 사용
            # SAM 프롬프트는 256x256 크기를 기대합니다.
            resized_prompt_mask = F.interpolate(
                initial_mask_logits, 
                size=(256, 256), mode='bilinear', align_corners=False
            )
            sparse_prompt_embeddings, dense_prompt_embeddings = self.sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=resized_prompt_mask # 저해상도 마스크 프롬프트
            )
        
        # ⭐ AssertionError: [MaskDecoder] Mismatched batch sizes: image_embeddings {image_embeddings.shape[0]} vs dense_prompt_embeddings {dense_prompt_embeddings.shape[0]}
        # 이전에 발생했던 이 오류에 대한 방어 로직 추가: dense_prompt_embeddings 배치 사이즈 일치
        if dense_prompt_embeddings.shape[0] != image_embedding.shape[0]:
            print(f"Warning: dense_prompt_embeddings batch size {dense_prompt_embeddings.shape[0]} mismatch with image_embedding batch size {image_embedding.shape[0]}. Attempting to correct with repeat.")
            dense_prompt_embeddings = dense_prompt_embeddings.repeat(image_embedding.shape[0], 1, 1, 1) # assuming dense_prompt_embeddings is (B, C, H, W)
            # If dense_prompt_embeddings was (B, D) then repeat(image_embedding.shape[0], 1)
            # This repeat must match the exact dimensions after prompt_encoder outputs.
            # However, typically prompt_encoder will already output matching batch size if input masks/boxes had.
            # The more likely scenario is if sparse_prompt_embeddings and dense_prompt_embeddings were from different sources.
            # Given your current setup, it should implicitly align if `masks=resized_prompt_mask` is used and `resized_prompt_mask` has correct batch size.

        # Step 4: Mask Decoder를 통해 최종 마스크 예측
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=image_pe, # ⭐ 최종 수정된 image_pe 사용 ⭐
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False, # 단일 마스크 출력
        )

        # 최종 마스크를 원본 이미지 크기로 업샘플링
        final_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]), # 입력 이미지의 H, W와 동일하게
            mode="bilinear",
            align_corners=False,
        )
        
        # (생성된 마스크_1024, iou_예측, 저해상도 마스크_256) 반환
        return final_masks, iou_predictions, low_res_masks 

# --- 4. MedSAM_GAN 통합 모델 정의 ---
class MedSAM_GAN(nn.Module):
    def __init__(self, sam_checkpoint: str, unet_checkpoint: str, out_channels: int = 1):
        super().__init__()
        # MedSAM (Generator 역할)
        self.sam = MedSAM(
            checkpoint=sam_checkpoint,
            model_type="vit_b",
            out_channels=out_channels, 
            unet_checkpoint=unet_checkpoint
        )
        # Discriminator
        self.discriminator = MaskDiscriminator(in_channels=4) # 3 (RGB image) + 1 (mask) = 4 channels

    def forward(self, image: torch.Tensor, *, real_low_res_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # image: 1채널 CT 이미지 (NpySegDataset에서 [0, 255] 스케일)
        
        # Generator (SAM)를 통해 마스크 생성
        # MedSAM.forward는 (생성된 마스크_1024, iou_predictions, 저해상도 마스크_256)를 반환
        # prompt_boxes는 MedSAM 클래스 내부에서 처리될 것이므로 여기서는 None으로 전달합니다.
        masks_1024_gen, iou_predictions_gen, low_res_masks_256_gen = self.sam(image, prompt_boxes=None)


        # Discriminator 입력용 이미지 준비
        # CT 1채널 이미지를 Discriminator 입력에 맞게 3채널 RGB처럼 복제
        # Discriminator는 256x256 마스크를 기대하므로, 이미지도 256x256으로 보간
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
            # real_low_res_mask는 train_GAN.py에서 256x256으로 리사이즈되어 전달될 것으로 예상됩니다.
            discriminator_input_for_real = torch.cat([resized_image_rgb_for_D, real_low_res_mask], dim=1)
            discriminator_output_for_real_mask = self.discriminator(discriminator_input_for_real)
        
        # 반환 값 순서: (Generator 생성 마스크, IoU 예측, 생성 마스크에 대한 D 출력, 저해상도 마스크, 실제 마스크에 대한 D 출력)
        return masks_1024_gen, iou_predictions_gen, discriminator_output_for_generated_mask, low_res_masks_256_gen, discriminator_output_for_real_mask