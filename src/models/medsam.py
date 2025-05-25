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
        B = image.shape[0]
        original_image_size = image.shape[2:]

        # Step 1: Convert grayscale to RGB
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        # Step 2: Encode image
        image_embeddings = self.sam.image_encoder(image)  # shape: (B, C, H', W')

        # Step 3: Resize prompt mask and encode
        resized_prompt_masks = F.interpolate(
            prompt_masks.float(), size=(256, 256), mode='bilinear', align_corners=False
        )
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=None, masks=resized_prompt_masks
        )

        # Step 4: Repeat dense prompt to align with internal repeat of image_embeddings
        dense_embeddings = torch.repeat_interleave(dense_embeddings, B, dim=0)  # shape: (B*B, C, H, W)

        # Step 5: Positional Encoding
        image_pe = self.sam.prompt_encoder.get_dense_pe()  # (1, C, H', W')
        image_pe = image_pe.expand(B, -1, -1, -1)  # (B, C, H', W')

        # Step 6: Decode masks
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,         # (B, C, H', W')
            image_pe=image_pe,                         # (B, C, H', W')
            sparse_prompt_embeddings=sparse_embeddings, # (B, 0, D) usually
            dense_prompt_embeddings=dense_embeddings,   # (B*B, C, H', W')
            multimask_output=False
        )

        # Step 7: Resize to original image resolution
        masks = F.interpolate(
            low_res_masks, size=original_image_size, mode='bilinear', align_corners=False
        )

        return masks, iou_predictions
