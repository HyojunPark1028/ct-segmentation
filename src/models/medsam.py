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
        
        # ⭐ 핵심 수정: sparse_embeddings가 비어있는 경우, MaskDecoder가 예상하는 최소 형태로 전달
        # 이전 시도들이 모두 실패한 것으로 보아, MaskDecoder 내부에서
        # sparse_prompt_embeddings.size(0)가 항상 이미지 배치 크기(B)로 계산되고,
        # 동시에 MaskDecoder가 (아마도 마스크 프롬프트에 대해) 이미지당 4개의 내부 토큰을 기대하여
        # image_embeddings와 image_pe를 (B * 4)로 확장하는 것으로 보입니다.
        # 이때 dense_embeddings와 image_pe는 (B)만 유지하여 불일치 발생.
        
        # 해결책은 dense_embeddings와 image_pe를 (B*4)로 확장하는 것입니다.
        # MaskDecoder 내부에서 image_pe는 pos_src로 반복되므로,
        # dense_embeddings만 (B*4)로 맞춰주면 됩니다.
        
        _sparse_embeddings = sparse_embeddings
        
        # dense_embeddings를 batch_size * 4 (예: 4 * 4 = 16)로 반복 확장
        # (이전에도 시도했으나 다시 동일 오류 발생한 경우, 이 부분이 제대로 적용되지 않았을 가능성도 있음)
        _dense_embeddings = torch.repeat_interleave(dense_embeddings, 4, dim=0)

        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(), # 이 값은 B * H * W 형태
            sparse_prompt_embeddings=_sparse_embeddings,
            dense_prompt_embeddings=_dense_embeddings,
            multimask_output=False
        )

        masks = F.interpolate(low_res_masks, original_image_size, mode='bilinear', align_corners=False)

        return masks, iou_predictions