import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.build_sam import sam_model_registry

class MedSAM(nn.Module):
    def __init__(self, checkpoint: str, out_channels: int = 1):
        super().__init__()

        # 1. SAM ViT-B 모델 전체 로드
        # MedSAM은 SAM의 Image Encoder, Prompt Encoder, Mask Decoder를 모두 활용합니다.
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)

        # 2. Image Encoder Fine-tuning: 모든 파라미터 학습 허용
        # 공식 MedSAM은 Image Encoder의 모든 학습 가능한 파라미터를 업데이트합니다.
        for p in self.sam.image_encoder.parameters():
            p.requires_grad = True

        # 3. Prompt Encoder 고정 (Frozen)
        # 공식 MedSAM은 Prompt Encoder를 고정하여 학습하지 않습니다.
        for p in self.sam.prompt_encoder.parameters():
            p.requires_grad = False
        
        # 4. Mask Decoder Fine-tuning: 모든 파라미터 학습 허용
        # 공식 MedSAM은 Mask Decoder의 모든 학습 가능한 파라미터를 업데이트합니다.
        for p in self.sam.mask_decoder.parameters():
            p.requires_grad = True

        # 참고: SAM의 Mask Decoder는 기본적으로 out_channels=1 (이진 마스크)로 설계되어 있습니다.
        # 다중 클래스 출력이 필요하다면, SAM Mask Decoder의 출력 후 추가적인 처리(예: Conv 레이어)가 필요할 수 있습니다.
        # 여기서는 이진 마스크 출력(의료 영상 세그멘테이션의 일반적인 시나리오)을 가정합니다.
        
    def forward(self, image: torch.Tensor, prompt_masks: torch.Tensor):
        # 0. 입력 이미지 크기 조정 및 채널 처리
        original_image_size = image.shape[2:] # H, W

        # 단일 채널 이미지를 3채널로 복제
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        # 1. Image Encoder (SAM의 비전 트랜스포머)
        image_embeddings = self.sam.image_encoder(image)

        # 2. Prompt Encoder: 마스크 프롬프트만 사용
        # prompt_masks는 (batch_size, 1, H, W) 형태여야 합니다. (이진 마스크)
        # SAM의 PromptEncoder는 마스크 입력을 256x256으로 기대합니다.
        # 따라서 입력 마스크를 적절히 리사이즈해야 합니다.
        # SAM의 내부 Preprocessing 로직을 따르는 것이 가장 좋습니다.
        # 여기서는 간단하게 Bilinear Interpolation을 사용합니다.
        # 실제 SAM 구현에서는 'preprocess' 함수에서 이를 처리합니다.
        
        # SAM Mask Decoder가 기대하는 저해상도 마스크 프롬프트 크기
        # SAM Mask Decoder는 256x256의 마스크 임베딩을 생성하므로,
        # 프롬프트 마스크도 그에 상응하는 크기로 조정될 필요가 있습니다.
        # 보통 SAM의 Image Encoder 출력 해상도와 관련이 있습니다 (예: 64x64 또는 256x256)
        # SAM의 `MaskDecoder`는 `image_embeddings`와 함께 `dense_prompt_embeddings`를 받는데,
        # 이 `dense_prompt_embeddings`는 마스크 프롬프트로부터 생성됩니다.
        # SAM의 `PromptEncoder`는 마스크 프롬프트가 (batch_size, 1, 256, 256) 크기로 들어올 것을 기대합니다.
        
        # 입력 prompt_masks를 256x256으로 리사이즈 (SAM Prompt Encoder의 요구사항)
        # 이 단계는 SAM의 전처리 파이프라인에서 처리될 수 있습니다.
        resized_prompt_masks = F.interpolate(
            prompt_masks.float(), # 마스크는 float 타입이어야 함
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        )

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            # labels=None,
            boxes=None,
            masks=resized_prompt_masks
        )

        num_prompts_per_image = sparse_embeddings.shape[0] // image_embeddings.shape[0]
        
        # 이미지당 여러 프롬프트가 있는 경우 dense_embeddings를 그에 따라 반복합니다.
        if num_prompts_per_image > 1:
            dense_embeddings = torch.repeat_interleave(dense_embeddings, num_prompts_per_image, dim=0)
        # ⭐ 수정된 부분 끝        
        
        # 3. Mask Decoder
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(), # Image PE
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False # MedSAM은 일반적으로 단일 마스크 출력을 사용합니다.
        )

        # 4. 출력 마스크를 원본 이미지 크기로 업샘플링
        masks = F.interpolate(low_res_masks, original_image_size, mode='bilinear', align_corners=False)

        return masks, iou_predictions
