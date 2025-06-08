import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.build_sam import sam_model_registry
import segmentation_models_pytorch as smp

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

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1), # Output: (B, 512, 31, 31)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1) # Output: (B, 1, 30, 30) - 'realness' map
        )

    def forward(self, x: torch.Tensor):
        """
        Discriminator의 forward pass.
        Args:
            x (torch.Tensor): 이미지와 마스크가 채널 방향으로 합쳐진 텐서.
                              (B, in_channels, H, W)
        Returns:
            torch.Tensor: 각 패치의 '진위' 점수를 나타내는 텐서.
                          (B, 1, H_D, W_D)
        """
        return self.net(x)

# --- 2. MedSAM 모델에 GAN 통합 ---
class MedSAM_GAN(nn.Module):
    def __init__(self, sam_checkpoint: str, unet_checkpoint: str, out_channels: int = 1):
        super().__init__()
        # SAM 모델 초기화 (Generator 역할)
        self.sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)

        # SAM의 image encoder, prompt encoder, mask decoder 파라미터 학습 활성화
        # (이들이 GAN의 Generator 역할을 수행하며, Discriminator의 피드백을 받음)
        for p in self.sam.image_encoder.parameters():
            p.requires_grad = True
        for p in self.sam.prompt_encoder.parameters():
            p.requires_grad = True
        for p in self.sam.mask_decoder.parameters():
            p.requires_grad = True

        # U-Net 모델 초기화 (초기 프롬프트 생성용, 학습 불필요)
        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,  # ImageNet 사전 학습 가중치 사용 안 함
            in_channels=1,
            classes=1,
            activation=None
        )
        # U-Net 체크포인트 로드 및 파라미터 고정
        state_dict = torch.load(unet_checkpoint, map_location="cpu", weights_only=False)
        new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        self.unet.load_state_dict(new_state_dict)
        self.unet.eval() # 평가 모드 유지
        for p in self.unet.parameters():
            p.requires_grad = False # U-Net 파라미터 고정

        # GAN Discriminator 초기화
        # Discriminator는 리사이즈된 원본 이미지(3채널)와 생성된 마스크(1채널)를 입력받으므로 4채널
        self.discriminator = MaskDiscriminator(in_channels=4)

    def forward(self, image: torch.Tensor, real_low_res_mask: torch.Tensor = None, *args, **kwargs):
        """
        MedSAM_GAN 모델의 forward pass.
        Args:
            image (torch.Tensor): 입력 이미지 텐서 (B, 1 or 3, H, W).
            real_low_res_mask (torch.Tensor, optional): 실제 Ground Truth 마스크 텐서 (B, 1, 256, 256).
                                                        Discriminator 학습 시 필요하며, Generator 학습 시에는 None.
        Returns:
            masks (torch.Tensor): 원본 이미지 크기로 업샘플링된 최종 예측 마스크 (B, 1, H, W).
            iou_predictions (torch.Tensor): SAM의 IOU 예측 값 (B, 1).
            discriminator_output_for_generated_mask (torch.Tensor): 생성된 마스크에 대한 Discriminator의 판별 결과.
                                                                    Generator의 adversarial loss 계산에 사용.
                                                                    (B, 1, H_D, W_D)
            discriminator_output_for_real_mask (torch.Tensor, optional): 실제 마스크에 대한 Discriminator의 판별 결과.
                                                                         Discriminator의 loss 계산에 사용.
                                                                         real_low_res_mask가 제공될 때만 반환.
        """
        B = image.shape[0]
        original_image_size = image.shape[2:]

        # Step 1: CT 이미지가 1채널이면 3채널로 복제
        if image.shape[1] == 1:
            image_rgb = image.repeat(1, 3, 1, 1)  # (B, 3, H, W)
        else:
            image_rgb = image

        # Discriminator 입력으로 사용하기 위해 image_rgb를 256x256으로 리사이즈
        # cGAN에서 이미지와 마스크를 함께 Discriminator에 입력하는 일반적인 방법
        resized_image_rgb_for_D = F.interpolate(
            image_rgb, size=(256, 256), mode='bilinear', align_corners=False
        )

        # Step 2: U-Net을 통한 초기 마스크 예측 → SAM prompt 용도
        # U-Net은 no_grad()로 고정되어 있으므로 추론만 수행
        with torch.no_grad():
            initial_mask = self.unet(image)  # (B, 1, H, W), 원본 크기
            initial_mask_bin = (initial_mask > 0.5).float() # 이진 마스크로 변환

        # 프롬프트 마스크를 SAM이 요구하는 256x256 크기로 리사이즈
        resized_prompt_mask = F.interpolate(
            initial_mask_bin, size=(256, 256), mode='bilinear', align_corners=False
        )

        # Step 3: SAM image encoder를 통해 이미지 임베딩 추출
        image_embeddings = self.sam.image_encoder(image_rgb)  # (B, C, H', W')

        # Step 4: SAM prompt encoder를 통해 프롬프트 임베딩 추출
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=None, masks=resized_prompt_mask
        )

        # Step 5: SAM positional encoding
        image_pe = self.sam.prompt_encoder.get_dense_pe()  # (1, C, H', W')
        image_pe = image_pe.expand(B, -1, -1, -1) # 배치 크기에 맞게 확장

        # Step 6: SAM mask decoder (GAN의 Generator 역할)
        # 저해상도 마스크와 IOU 예측 값을 생성
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False # 단일 마스크 출력
        )

        # Step 7: 생성된 저해상도 마스크를 Discriminator에 입력하여 판별 결과 얻기
        # Discriminator 입력: [리사이즈된 원본 이미지(RGB), 생성된 마스크(1채널)]
        discriminator_input_for_generated = torch.cat([resized_image_rgb_for_D, low_res_masks], dim=1)
        discriminator_output_for_generated_mask = self.discriminator(discriminator_input_for_generated)

        # Step 8: 예측 마스크를 원본 이미지 크기로 업샘플링하여 최종 결과 생성
        masks = F.interpolate(
            low_res_masks, size=original_image_size, mode='bilinear', align_corners=False
        )

        # --- Discriminator 학습을 위한 추가 로직 (real_low_res_mask가 제공될 경우) ---
        discriminator_output_for_real_mask = None
        if real_low_res_mask is not None:
            # 실제 마스크를 Discriminator에 입력하여 판별 결과 얻기
            # Discriminator 입력: [리사이즈된 원본 이미지(RGB), 실제 마스크(1채널)]
            discriminator_input_for_real = torch.cat([resized_image_rgb_for_D, real_low_res_mask], dim=1)
            discriminator_output_for_real_mask = self.discriminator(discriminator_input_for_real)
            return masks, iou_predictions, discriminator_output_for_generated_mask, discriminator_output_for_real_mask
        else:
            # Generator 학습 시에는 생성된 마스크에 대한 판별 결과만 반환
            return masks, iou_predictions, discriminator_output_for_generated_mask

