import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.build_sam import sam_model_registry

class MedSAM(nn.Module):
    def __init__(self, checkpoint: str, out_channels: int = 1):
        super().__init__()
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)

        # 파인튜닝 범위 지정: image encoder + mask decoder만 학습
        for p in self.sam.image_encoder.parameters():
            p.requires_grad = True
        for p in self.sam.prompt_encoder.parameters():
            p.requires_grad = False
        for p in self.sam.mask_decoder.parameters():
            p.requires_grad = True

    def forward(self, image: torch.Tensor, prompt_masks: torch.Tensor):
        """
        Arguments:
            image (B, 1 or 3, H, W): CT 슬라이스
            prompt_masks (B, 1, H, W): 병변이 있는 ground-truth 마스크
        Returns:
            masks (B, 1, H, W): 예측된 마스크 (0~1 범위)
            iou_predictions (B, 1): 예측 마스크의 품질 추정
        """
        B = image.shape[0]
        original_image_size = image.shape[2:]

        # Step 1: 1채널 CT 이미지를 3채널로 확장
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)  # (B, 3, H, W)

        # Step 2: Image Encoder → (B, C, H', W')
        image_embeddings = self.sam.image_encoder(image)

        # Step 3: Prompt Encoder (GT 마스크 → dense prompt 변환)
        resized_prompt_masks = F.interpolate(
            prompt_masks.float(), size=(256, 256), mode='bilinear', align_corners=False
        )
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=None, masks=resized_prompt_masks
        )

        # Step 4: Positional Encoding → (B, C, H', W')
        image_pe = self.sam.prompt_encoder.get_dense_pe()
        image_pe = image_pe.expand(B, -1, -1, -1)

        # Step 5: Mask Decoder (입력은 모두 B 배치 단위)
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,  # 보통 (B, 0, D)
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        # Step 6: 최종 마스크를 원본 크기로 복원
        masks = F.interpolate(
            low_res_masks, size=original_image_size, mode='bilinear', align_corners=False
        )
        return masks, iou_predictions
