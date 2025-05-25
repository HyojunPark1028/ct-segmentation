import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.build_sam import sam_model_registry

class MedSAM(nn.Module):
    def __init__(self, checkpoint: str, out_channels: int = 1):
        super().__init__()
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)

        for p in self.sam.image_encoder.parameters():
            p.requires_grad = True

        for p in self.sam.prompt_encoder.parameters():
            p.requires_grad = False
        
        for p in self.sam.mask_decoder.parameters():
            p.requires_grad = True
        
    def forward(self, image: torch.Tensor, prompt_masks: torch.Tensor):
        original_image_size = image.shape[2:]

        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        image_embeddings = self.sam.image_encoder(image)

        resized_prompt_masks = F.interpolate(
            prompt_masks.float(),
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        )

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=resized_prompt_masks
        )
        
        # ⭐ 핵심 수정: dense_embeddings의 배치 차원을 MaskDecoder가 기대하는 총 프롬프트 수에 맞춤
        # MaskDecoder는 이미지당 4개의 프롬프트 토큰(implicitly)을 기대하는 것으로 보임.
        # 따라서 dense_embeddings도 (Batch_Size * 4)의 배치 차원을 가져야 함.
        # sparse_embeddings는 (Batch, 0, C) 형태를 유지하여 AttributeError를 피함.
        
        # dense_embeddings를 batch_size * 4 (예: 4 * 4 = 16)로 반복 확장
        _dense_embeddings = torch.repeat_interleave(dense_embeddings, 4, dim=0)
        
        # sparse_embeddings는 (Batch, 0, EmbeddingDim) 형태 그대로 전달하여 MaskDecoder 내부의
        # `expand` 연산을 가능하게 하고, `tokens.shape[0]`가 `Batch * 4`로 계산되도록 합니다.
        _sparse_embeddings = sparse_embeddings 

        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=_sparse_embeddings,
            dense_prompt_embeddings=_dense_embeddings,
            multimask_output=False
        )

        masks = F.interpolate(low_res_masks, original_image_size, mode='bilinear', align_corners=False)

        return masks, iou_predictions