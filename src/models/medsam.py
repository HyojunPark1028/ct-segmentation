import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.build_sam import sam_model_registry

class MedSAM(nn.Module):
    def __init__(self, checkpoint: str, out_channels: int = 1):
        super().__init__()
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)

        # 학습 가능한 모듈만 requires_grad=True
        for p in self.sam.image_encoder.parameters():
            p.requires_grad = True
        for p in self.sam.prompt_encoder.parameters():
            p.requires_grad = False
        for p in self.sam.mask_decoder.parameters():
            p.requires_grad = True

    def forward(self, image: torch.Tensor, prompt_masks: torch.Tensor = None):
        """
        Arguments:
            image: (B, 1 or 3, H, W)
            prompt_masks: (B, 1, H, W) or None
                - 학습 시: GT 마스크 사용
                - 평가 시: prompt 없이 수행
        Returns:
            masks: (B, 1, H, W)
            iou_predictions: (B, 1)
        """
        B = image.shape[0]
        original_image_size = image.shape[2:]

        # Step 1: CT 이미지가 1채널이면 3채널로 복제
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        # Step 2: 이미지 임베딩 생성
        image_embeddings = self.sam.image_encoder(image)  # (B, C, H', W')

        # Step 3: 포지셔널 인코딩
        image_pe = self.sam.prompt_encoder.get_dense_pe()  # (1, C, H', W')
        image_pe = image_pe.expand(B, -1, -1, -1)  # (B, C, H', W')

        # Step 4: 프롬프트 인코딩
        if prompt_masks is not None:
            resized_prompt_masks = F.interpolate(
                prompt_masks.float(), size=(256, 256), mode='bilinear', align_corners=False
            )
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None, boxes=None, masks=resized_prompt_masks
            )
        else:
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            # ⚠️ 프롬프트가 없을 경우, dense embedding은 (1, C, H, W)로 고정되므로 확장 필요
            if dense_embeddings.shape[0] == 1 and B > 1:
                dense_embeddings = dense_embeddings.expand(B, -1, -1, -1)

        # Step 5: 마스크 디코딩
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        # Step 6: 예측 마스크 원래 크기로 복원
        masks = F.interpolate(
            low_res_masks, size=original_image_size, mode='bilinear', align_corners=False
        )

        return masks, iou_predictions
