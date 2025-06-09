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

# --- 3. MedSAM 모델 정의 (SAM Backbone + UNet as initial prompt generator) ---
class MedSAM(nn.Module):
    """
    MedSAM 모델: SAM의 이미지 인코더에 UNet을 결합하여 초기 마스크 프롬프트를 생성하고,
    SAM의 마스크 디코더로 최종 마스크를 예측하는 구조.
    """
    def __init__(self, checkpoint: str, model_type: str, out_channels: int = 1, unet_checkpoint: Optional[str] = None):
        super().__init__()
        # SAM 모델 로드
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        
        # SAM 파라미터의 학습 가능/불가능 설정 (모두 학습 가능으로 기본 설정)
        for p in self.sam.image_encoder.parameters(): p.requires_grad = True
        for p in self.sam.prompt_encoder.parameters(): p.requires_grad = True
        for p in self.sam.mask_decoder.parameters(): p.requires_grad = True

        # UNet 모델 초기화 및 가중치 로드
        # UNet은 SAM의 마스크 디코더를 위한 초기 프롬프트(마스크)를 생성하는 용도로 사용됩니다.
        self.unet = smp.Unet(
            encoder_name="resnet34", 
            encoder_weights=None, # UNet 체크포인트를 로드할 것이므로 None 유지
            in_channels=1, # CT 이미지는 1채널
            classes=out_channels,
            activation=None # Unet의 최종 활성화 함수를 GAN에서 유연하게 사용하기 위해 None으로 설정
        )
        
        # UNet 체크포인트 로드
        if unet_checkpoint and os.path.exists(unet_checkpoint):
            try:
                unet_state_dict = torch.load(unet_checkpoint, map_location='cpu')
                # ⭐ 핵심 수정: 로드된 state_dict의 키에서 "model." 접두사를 제거합니다. ⭐
                # 이는 train.py의 unet 로딩 방식과 일관성을 유지합니다.
                new_unet_state_dict = {k.replace("model.", ""): v for k, v in unet_state_dict.items()}
                
                # CustomUNet 대신 직접 smp.Unet을 사용했으므로, self.unet에 바로 로드합니다.
                self.unet.load_state_dict(new_unet_state_dict, strict=True)
                
                print(f"Loaded UNet checkpoint from {unet_checkpoint}")
            except Exception as e:
                print(f"Error loading UNet checkpoint: {e}. UNet will be initialized from scratch.")
                # UNet 로드에 실패해도 전체 모델 학습이 가능하도록 예외 처리
        else:
            print(f"UNet checkpoint not found at {unet_checkpoint}. Starting U-Net from scratch (or imagenet if specified).")
            # 만약 UNet이 imagenet weights로 시작해야 한다면, 위 smp.Unet 초기화 시 encoder_weights="imagenet" 설정.

        # ⭐ UNet 가중치 고정 (medsam.py의 의도를 유지) ⭐
        # UNet은 초기 프롬프트 생성용이며, 이 가중치는 학습되지 않도록 고정합니다.
        self.unet.eval() # 평가 모드로 설정
        for p in self.unet.parameters():
            p.requires_grad = False # 학습 비활성화

    def forward(self, image: torch.Tensor, prompt_boxes: Optional[torch.Tensor] = None):
        # image: NpySegDataset에서 넘어온 텐서. normalize_type="sam"에 의해 [0, 255] 스케일.

        # SAM 이미지 인코더는 3채널을 기대하므로 1채널 이미지를 복제합니다.
        if image.shape[1] == 1:
            image_rgb = image.repeat(1, 3, 1, 1) # (B, 3, H, W)
        else:
            image_rgb = image

        # ⭐ SAM 이미지 인코더 입력 스케일: Dataset에서 이미 [0, 255]이므로 추가 스케일링 필요 없음 ⭐
        image_for_sam_encoder = image_rgb

        # Step 1: U-Net을 통한 초기 마스크 예측 (프롬프트 용도)
        # NpySegDataset에서 [0, 255] 범위로 정규화된 이미지를 받으므로,
        # U-Net이 학습되었던 [0, 1] 범위로 역스케일링합니다.
        # U-Net은 ImageNet 사전 학습 가중치를 사용했으므로 [0, 1] 스케일을 기대합니다.
        image_for_unet = image / 255.0 # [0, 255] -> [0, 1]

        with torch.no_grad(): # UNet은 고정되었으므로 no_grad 블록에서 실행
            initial_mask_logits = self.unet(image_for_unet) 
            # U-Net의 출력을 이진화하지 않고 확률맵으로 프롬프트에 사용 (권장)

        # SAM 이미지 인코더를 통해 이미지 임베딩 추출
        image_embedding = self.sam.image_encoder(image_for_sam_encoder)

        # SAM의 위치 인코딩
        image_pe = self.sam.prompt_encoder.get_dense_pe()
        # ⭐⭐ 수정: expand 대신 repeat를 사용하여 배치 차원을 명확히 맞춥니다. ⭐⭐
        batch_size = image_embedding.shape[0]
        if image_pe.shape[0] == 1 and batch_size > 1:
            image_pe = image_pe.repeat(batch_size, 1, 1, 1)
        # Ensure image_pe is on the same device as image_embedding
        image_pe = image_pe.to(image_embedding.device)


        # Step 2: 초기 마스크 또는 박스를 SAM 프롬프트로 변환
        sparse_embeddings = None
        dense_embeddings = None

        if prompt_boxes is not None:
            # 박스 프롬프트가 제공되면 사용
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
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
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=resized_prompt_mask # 저해상도 마스크 프롬프트
            )

        # Step 3: Mask Decoder를 통해 최종 마스크 예측
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=image_pe, 
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False, # 단일 마스크 출력
        )

        # 최종 마스크를 원본 이미지 크기로 업스케일 (SAM 모델의 일반적인 동작)
        final_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]), # 입력 이미지의 H, W와 동일하게
            mode="bilinear",
            align_corners=False,
        )
        
        # (생성된 마스크, iou_예측, 저해상도 마스크) 반환
        return final_masks, iou_predictions, low_res_masks 

# --- 4. MedSAM_GAN 통합 모델 정의 ---
class MedSAM_GAN(nn.Module):
    def __init__(self, sam_checkpoint: str, unet_checkpoint: str, out_channels: int = 1):
        super().__init__()
        # MedSAM (Generator 역할)
        # MedSAM 클래스 내에서 UNet 로딩 및 SAM 설정이 모두 처리됩니다.
        self.sam = MedSAM(
            checkpoint=sam_checkpoint,
            model_type="vit_b",
            output_channels=out_channels,
            unet_checkpoint=unet_checkpoint
        )
        # Discriminator
        self.discriminator = MaskDiscriminator(in_channels=4) # 3 (RGB image) + 1 (mask) = 4 channels

    def forward(self, image: torch.Tensor, *, real_low_res_mask: Optional[torch.Tensor] = None):
        # image: 1채널 CT 이미지 (NpySegDataset에서 [0, 255] 스케일)
        
        # Generator (SAM)를 통해 마스크 생성
        # MedSAM.forward는 (생성된 마스크_1024, iou_predictions, 저해상도 마스크_256)를 반환
        
        # ⭐ MedSAM의 입력은 항상 원본 이미지 (1채널 [0,255])와 선택적인 prompt_boxes
        # MedSAM 클래스 내부에서 1채널 이미지를 SAM 인코더를 위해 3채널로 복제하고,
        # UNet을 위해 [0,1] 스케일로 조정합니다.
        
        # prompt_boxes가 None인 경우, MedSAM 내부에서 UNet으로 초기 마스크를 생성하여 프롬프트로 사용합니다.
        # 따라서 여기서는 명시적으로 prompt_boxes를 전달하지 않아도 됩니다. (train_GAN.py에서 None으로 호출하고 있음)
        masks_1024_gen, iou_predictions_gen, low_res_masks_256_gen = self.sam(image, prompt_boxes=None) # 또는 input_box 사용 가능

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
            # 실제 마스크도 256x256 크기여야 합니다. 
            # NpySegDataset에서 마스크도 이미지와 동일하게 리사이즈됩니다.
            # 하지만 Discriminator 입력은 256x256이므로, real_low_res_mask가 이미 그 크기인지 확인 필요.
            # real_low_res_mask는 NpySegDataset에서 원래 1024x1024 마스크를 로드한 후
            # train_GAN.py에서 F.interpolate를 통해 256x256으로 리사이즈되어 전달될 것으로 예상됩니다.
            
            # Discriminator 입력: [리사이즈된 원본 이미지(RGB), 실제 저해상도 마스크(1채널)]
            discriminator_input_for_real = torch.cat([resized_image_rgb_for_D, real_low_res_mask], dim=1)
            discriminator_output_for_real_mask = self.discriminator(discriminator_input_for_real)
        
        # 반환 값 순서: (Generator 생성 마스크_1024, IoU 예측, 생성 마스크에 대한 D 출력_256, 저해상도 마스크_256, 실제 마스크에 대한 D 출력_256)
        return masks_1024_gen, iou_predictions_gen, discriminator_output_for_generated_mask, low_res_masks_256_gen, discriminator_output_for_real_mask