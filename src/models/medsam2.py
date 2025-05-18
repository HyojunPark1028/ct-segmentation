# medsam2 model implementation
# Place this file under src/models/medsam2.py

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from segment_anything import sam_model_registry

class MedSAM2(nn.Module):
    """
    MedSAM2: differentiable prompt-driven segmentation using SAM2 backbone
    Supports end-to-end training.

    Usage:
        model = MedSAM2(
            checkpoint=cfg.model.checkpoint,  # e.g. "weights/MedSAM2_latest.pt"
            model_type="vit_b",
            image_size=512,
            device=device,
        )
    """
    def __init__(self,
                 checkpoint: str,
                 model_type: str = "vit_b",
                 image_size: int = 512,
                 device: str = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size

        # Initialize SAM2 backbone without loading any weights
        self.sam = sam_model_registry[model_type](checkpoint=None)
        self.sam.to(self.device)

        # Load full MedSAM2 checkpoint (backbone + head)
        ckpt = torch.load(checkpoint, map_location=self.device)
        state = ckpt.get('model', ckpt)
        # load_state_dict with strict=False to allow partial matches
        self.sam.load_state_dict(state, strict=False)

        # Prepare full-image box prompt once
        # shape: (1, 4) -> [x0, y0, x1, y1]
        self.register_buffer(
            'full_box',
            torch.tensor([[0, 0, image_size - 1, image_size - 1]], dtype=torch.float)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass is fully differentiable.
        Args:
            x: (B, C, H, W) input slices (C=1 or 3)
        Returns:
            logits: (B, 1, H, W) mask logits
        """
        B, C, H, W = x.shape
        x = x.to(self.device)
        # replicate grayscale to RGB
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        # resize to SAM input resolution
        x_resized = F.interpolate(
            x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False
        )
        # encode image
        image_embeddings = self.sam.image_encoder(x_resized)

        # prepare box prompt for each sample
        boxes = self.full_box.expand(B, -1)  # (B, 4)
        # prompt encoder: returns sparse_emb, dense_emb
        sparse_emb, dense_emb = self.sam.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None
        )
        # mask decoder: low-res logits
        low_res_logits, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
        )
        # upsample to original size
        logits = F.interpolate(
            low_res_logits, size=(H, W), mode='bilinear', align_corners=False
        )
        return logits
