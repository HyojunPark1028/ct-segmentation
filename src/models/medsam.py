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
        
        # ⭐ 핵심 수정 부분: sparse_embeddings의 두 번째 차원(토큰 수)이 0인 경우 None으로 전달
        # MaskDecoder는 sparse_prompt_embeddings가 None일 때 image_embeddings를 반복 확장하지 않음.
        # dense_embeddings는 이미 image_embeddings의 배치 크기와 일치하므로 바로 사용 가능.
        if sparse_embeddings.shape[1] == 0:
            _sparse_embeddings = None 
            _dense_embeddings = dense_embeddings
        else:
            _sparse_embeddings = sparse_embeddings
            _dense_embeddings = dense_embeddings

        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=_sparse_embeddings, # None 또는 유효한 텐서
            dense_prompt_embeddings=_dense_embeddings,
            multimask_output=False
        )

        masks = F.interpolate(low_res_masks, original_image_size, mode='bilinear', align_corners=False)

        return masks, iou_predictions