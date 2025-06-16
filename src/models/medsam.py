# medsam.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.build_sam import sam_model_registry
import segmentation_models_pytorch as smp
import time

class MedSAM(nn.Module):
    def __init__(self, sam_checkpoint: str, unet_checkpoint: str, out_channels: int = 1):
        super().__init__()
        # SAM 모델 초기화
        self.sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)

        # SAM의 학습 가능/불가능 설정
        # (현재는 모두 학습 가능으로 설정되어 있지만, 필요에 따라 image_encoder를 고정할 수 있습니다.)
        for p in self.sam.image_encoder.parameters():
            p.requires_grad = True
        for p in self.sam.prompt_encoder.parameters():
            p.requires_grad = True
        for p in self.sam.mask_decoder.parameters():
            p.requires_grad = True

        # UNet 모델 초기화 및 가중치 로드
        self.unet = smp.Unet(
            encoder_name="resnet34", # train.py에서 cfg.model.name이 'unet'일때 사용되는 encoder_name
            encoder_weights=None, # unet_checkpoint를 로드할 것이므로 None 유지
            in_channels=1,
            classes=out_channels,
            activation=None
        )
        state_dict = torch.load(unet_checkpoint, map_location="cpu", weights_only=False)
        new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        self.unet.load_state_dict(new_state_dict)

        # ⭐⭐⭐ 핵심: UNet 가중치 고정 ⭐⭐⭐
        # val_dice 0.73을 달성한 U-Net의 성능을 최대한 활용하고,
        # 불필요한 재학습으로 인한 성능 저하 및 불안정성 방지
        self.unet.eval() # UNet을 평가 모드로 고정 (Batch Normalization, Dropout 등이 추론 모드로 작동)
        for p in self.unet.parameters():
            p.requires_grad = False # UNet 파라미터 학습 비활성화

    def forward(self, image: torch.Tensor, *args, **kwargs):
        # image: NpySegDataset에서 넘어온 텐서, normalize_type="sam"에 의해 [0, 255] 스케일

        # SAM 이미지 인코더는 3채널을 기대하므로 1채널 이미지를 복제합니다.
        if image.shape[1] == 1:
            image_rgb = image.repeat(1, 3, 1, 1) # (B, 3, H, W)
        else:
            image_rgb = image

        # ⭐⭐⭐ SAM 이미지 인코더 입력 스케일: Dataset에서 이미 [0, 255]이므로 추가 스케일링 필요 없음 ⭐⭐⭐
        # NpySegDataset의 normalize_type="sam" 로직에 의해 image_rgb는 이미 [0, 255] 범위입니다.
        image_for_sam_encoder = image_rgb

        # Step 1: U-Net을 통한 초기 마스크 예측 (프롬프트 용도)
        # ⭐⭐⭐ 핵심 수정: U-Net 입력 스케일 조정 ⭐⭐⭐
        # NpySegDataset에서 [0, 255] 범위로 정규화된 이미지를 받으므로,
        # U-Net이 학습되었던 [0, 1] 범위로 역스케일링합니다.
        # U-Net은 ImageNet 사전 학습 가중치를 사용했으므로 [0, 1] 스케일을 기대합니다.
        image_for_unet = image / 255.0 # [0, 255] -> [0, 1]

        with torch.no_grad(): # UNet은 고정되었으므로 no_grad 블록에서 실행 (옵션)
            initial_mask_logits = self.unet(image_for_unet) # ⭐ 스케일 조정된 이미지 사용
            # ⭐ U-Net의 출력을 이진화하지 않고 확률맵으로 프롬프트에 사용 (권장) ⭐
            # initial_mask_bin = (F.sigmoid(initial_mask_logits) > 0.5).float() # 이진화 제거

        # SAM 이미지 인코더를 통해 이미지 임베딩 추출
        image_embedding = self.sam.image_encoder(image_for_sam_encoder)

        # ⭐⭐⭐ 오류 수정: get_image_pe 대신 get_dense_pe() 사용 ⭐⭐⭐
        image_pe = self.sam.prompt_encoder.get_dense_pe() # (1, C, H', W')
        image_pe = image_pe.expand(image_embedding.shape[0], -1, -1, -1) # 배치 차원 확장


        # Step 2: 초기 마스크를 SAM 프롬프트로 변환
        # SAM 프롬프트는 256x256 크기를 기대합니다.
        resized_prompt_mask = F.interpolate(
            initial_mask_logits, # ⭐ 이진화되지 않은 U-Net의 로짓 사용
            size=(256, 256), mode='bilinear', align_corners=False
        )

        # Step 3: Prompt Encoder를 통해 프롬프트 임베딩 생성
        # (points와 boxes는 현재 사용하지 않으므로 None)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=resized_prompt_mask # 저해상도 마스크 프롬프트
        )
        start_decoder_time = time.time()
        # Step 4: Mask Decoder를 통해 최종 마스크 예측
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=image_pe, # 수정된 image_pe 변수 사용
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False, # 단일 마스크 출력
        )

        elapsed_decoder_time = time.time() - start_decoder_time
        print(f"[MedSAM] Mask Decoder Forward Time: {elapsed_decoder_time:.4f} sec")

        # 최종 마스크를 원본 이미지 크기로 업스케일 (SAM 모델의 일반적인 동작)
        final_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]), # 입력 이미지의 H, W와 동일하게
            mode="bilinear",
            align_corners=False,
        )

        # 최종 마스크와 IoU 예측 값을 반환합니다.
        return final_masks, iou_predictions