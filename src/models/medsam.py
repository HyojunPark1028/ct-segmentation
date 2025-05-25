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
                - 학습 시: GT 마스크 사용 (prompt_masks 제공)
                - 평가 시: prompt 없이 예측 (prompt_masks=None)
        Returns:
            masks: (B, 1, H, W) 예측 마스크
            iou_predictions: (B, 1) 품질 추정값
        """
        B = image.shape[0]
        original_image_size = image.shape[2:]

        # 1채널 CT 이미지를 3채널로 확장
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)  # (B, 3, H, W)

        # 이미지 인코딩
        image_embeddings = self.sam.image_encoder(image)  # (B, C, H', W')

        # Positional Encoding
        image_pe = self.sam.prompt_encoder.get_dense_pe()  # (1, C, H', W')
        image_pe = image_pe.expand(B, -1, -1, -1)

        # 프롬프트 처리
        if prompt_masks is not None:
            # 학습 시: GT 마스크를 prompt로 사용
            resized_prompt_masks = F.interpolate(
                prompt_masks.float(), size=(256, 256), mode='bilinear', align_corners=False
            )
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None, boxes=None, masks=resized_prompt_masks
            )
        else:
            # 평가 시: 프롬프트 없이 사용
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None, boxes=None, masks=None
            )

        # 마스크 디코딩
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        # 예측 마스크 원본 크기로 복원
        masks = F.interpolate(
            low_res_masks, size=original_image_size, mode='bilinear', align_corners=False
        )

        return masks, iou_predictions
