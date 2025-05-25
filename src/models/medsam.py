import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.build_sam import sam_model_registry

class MedSAM(nn.Module):
    def __init__(self, checkpoint: str, out_channels: int = 1):
        super().__init__()
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)

        # Enable gradients for Image Encoder
        for p in self.sam.image_encoder.parameters():
            p.requires_grad = True

        # Freeze Prompt Encoder
        for p in self.sam.prompt_encoder.parameters():
            p.requires_grad = False
        
        # Enable gradients for Mask Decoder
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
        
        # ⭐ 핵심 수정 부분: sparse_embeddings가 비어있는 경우 처리
        # sparse_embeddings.shape[1]이 0이라는 것은 유효한 토큰이 없다는 의미
        # 그러나 mask_decoder는 sparse_prompt_embeddings.size(0)을 요구하므로 None으로 전달할 수 없음.
        # 따라서 배치 크기 4에 해당하는 빈 토큰 텐서를 생성하여 전달합니다.
        if sparse_embeddings.shape[1] == 0:
            # MaskDecoder가 요구하는 (batch_size, num_tokens, embedding_dim) 형태의
            # 빈 sparse_prompt_embeddings 텐서를 생성합니다.
            # 여기서 num_tokens는 0으로, embedding_dim은 256으로 설정 (SAM의 임베딩 차원)
            # image_embeddings의 배치 크기(sparse_embeddings.shape[0])를 사용
            _sparse_embeddings = torch.empty(
                (sparse_embeddings.shape[0], 0, 256), # (Batch Size, 0, Embedding Dim)
                dtype=sparse_embeddings.dtype,
                device=sparse_embeddings.device
            )
            _dense_embeddings = dense_embeddings
        else:
            # sparse_embeddings에 유효한 토큰이 있는 경우 그대로 사용
            _sparse_embeddings = sparse_embeddings
            _dense_embeddings = dense_embeddings

        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=_sparse_embeddings,
            dense_prompt_embeddings=_dense_embeddings,
            multimask_output=False
        )

        masks = F.interpolate(low_res_masks, original_image_size, mode='bilinear', align_corners=False)

        return masks, iou_predictions