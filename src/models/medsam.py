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
        
        # MaskDecoder의 내부 로직은 마스크 프롬프트에 대해 이미지당 4개의 암시적 토큰을 가정하며
        # image_embeddings와 image_pe를 (Batch_Size * 4)로 확장하는 것으로 보입니다.
        # dense_embeddings 또한 이에 맞춰 배치 차원을 확장해야 합니다.
        
        # sparse_embeddings는 (Batch_Size, 0, Embedding_Dim) 형태를 유지하여
        # 'AttributeError: 'NoneType' object has no attribute 'size''를 피합니다.
        _sparse_embeddings = sparse_embeddings 

        # dense_embeddings를 (Batch_Size * 4)로 반복 확장하여 MaskDecoder의 기대치에 맞춥니다.
        # (예: 배치 4 -> 4 * 4 = 배치 16)
        _dense_embeddings = torch.repeat_interleave(dense_embeddings, 4, dim=0)

        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=_sparse_embeddings,
            dense_prompt_embeddings=_dense_embeddings,
            multimask_output=False
        )

        masks = F.interpolate(low_res_masks, original_image_size, mode='bilinear', align_corners=False)

        return masks, iou_predictions