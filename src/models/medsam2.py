# medsam2 model implementation
# Place this file under src/models/medsam2.py

import torch
import torch.nn as nn
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

class MedSAM2(nn.Module):
    """
    MedSAM2: A prompt-driven segmentation model for medical images
    based on Segment Anything Model 2 (SAM2) with volume/frame memory.

    Usage in your training pipeline:
        model = MedSAM2(
            checkpoint=cfg.model.checkpoint,
            model_type="vit_b",
            image_size=512,
            device=device,
        )
    """
    def __init__(self, checkpoint, image_size=512, device=None):
        super().__init__()
        # Determine device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size

        # Load the SAM2 backbone
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
        self.sam.to(self.device)

        # Predictor for prompt-based segmentation
        self.predictor = SamPredictor(self.sam)

        # Precomputed prompt: full-image bounding box
        # Format: [x_min, y_min, x_max, y_max]
        self.full_box = np.array([0, 0, image_size - 1, image_size - 1])[None, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of 2D slices or RGB images.

        Args:
            x: Tensor of shape (B, C, H, W). C can be 1 (grayscale) or 3 (RGB).
        Returns:
            logits: Tensor of shape (B, 1, H, W) with raw mask logits.
        """
        B, C, H, W = x.shape
        x = x.to(self.device)

        # If single-channel, replicate to 3 channels
        if C == 1:
            x = x.repeat(1, 3, 1, 1)

        # Resize input to the size expected by SAM2
        x_resized = torch.nn.functional.interpolate(
            x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False
        )

        logits_list = []
        # Process each sample individually due to predictor API
        for i in range(B):
            # Convert to HxWx3 numpy array for the predictor
            img_np = x_resized[i].permute(1, 2, 0).cpu().numpy()
            self.predictor.set_image(img_np)

            # Predict mask for the entire image using the full-box prompt
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=self.full_box,
                multimask_output=False,
            )
            # logits is an array of shape (1, H, W)
            logit = torch.from_numpy(logits[0]).to(self.device)  # (H, W)
            logits_list.append(logit)

        # Stack logits into a batch: (B, H, W)
        batch_logits = torch.stack(logits_list, dim=0)
        # Add channel dimension: (B, 1, H, W)
        batch_logits = batch_logits.unsqueeze(1)

        # Resize back to the original resolution
        batch_logits = torch.nn.functional.interpolate(
            batch_logits, size=(H, W), mode='bilinear', align_corners=False
        )
        return batch_logits
