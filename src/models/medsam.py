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
        
        # Adjust sparse_embeddings if no sparse tokens are generated from masks
        if sparse_embeddings.shape[1] == 0:
            _sparse_embeddings = None 
            _dense_embeddings = dense_embeddings
        else:
            # If sparse embeddings are present, they need to align with dense embeddings
            # The previous repeat_interleave logic was based on a misunderstanding of SAM's MaskDecoder.
            # dense_embeddings usually already matches image_embeddings batch size.
            # If sparse_embeddings exist, MaskDecoder should handle repeating image_embeddings.
            _sparse_embeddings = sparse_embeddings
            _dense_embeddings = dense_embeddings # Keep as is, MaskDecoder expects it this way
            
            # --- 이전 시도 코드 (이번 문제와는 관련 없는 것으로 확인되어 삭제) ---
            # num_prompts_per_image = sparse_embeddings.shape[0] // image_embeddings.shape[0]
            # if num_prompts_per_image > 1:
            #     _dense_embeddings = torch.repeat_interleave(dense_embeddings, num_prompts_per_image, dim=0)
            # -------------------------------------------------------------

        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=_sparse_embeddings,
            dense_prompt_embeddings=_dense_embeddings,
            multimask_output=False
        )

        masks = F.interpolate(low_res_masks, original_image_size, mode='bilinear', align_corners=False)

        return masks, iou_predictions