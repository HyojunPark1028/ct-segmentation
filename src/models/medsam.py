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
        B = image.shape[0]

        # Convert grayscale to RGB
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        # Step 1: image encoding
        image_embeddings = self.sam.image_encoder(image)  # (B, ...)
        image_embeddings = torch.repeat_interleave(image_embeddings, 4, dim=0)  # (B*4, ...)

        # Step 2: prompt encoding
        resized_prompt_masks = F.interpolate(prompt_masks.float(), size=(256, 256), mode='bilinear', align_corners=False)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(points=None, boxes=None, masks=resized_prompt_masks)
        dense_embeddings = torch.repeat_interleave(dense_embeddings, 4, dim=0)  # (B*4, ...)

        # Step 3: positional encoding
        image_pe = self.sam.prompt_encoder.get_dense_pe()  # (1, C, H, W)
        image_pe = image_pe.repeat(B * 4, 1, 1, 1)  # (B*4, ...)

        # Step 4: decode
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        masks = F.interpolate(low_res_masks, original_image_size, mode='bilinear', align_corners=False)
        return masks, iou_predictions
