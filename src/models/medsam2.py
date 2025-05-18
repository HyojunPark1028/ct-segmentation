# medsam2 model implementation
# Place this file under src/models/medsam2.py

import torch
import torch.nn as nn
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

class MedSAM2(nn.Module):
    """
    MedSAM2: A prompt-driven segmentation model for medical images
    based on Segment Anything Model 2 (SAM2) with full-image prompt.

    Usage in your training pipeline:
        model = MedSAM2(
            sam_checkpoint=cfg.model.sam_checkpoint,
            med_checkpoint=cfg.model.med_checkpoint,
            model_type="vit_b",
            image_size=512,
            device=device,
        )
    """
    def __init__(self,
                 sam_checkpoint: str,
                 med_checkpoint: str = None,
                 model_type: str = "vit_b",
                 image_size: int = 512,
                 device: str = None):
        super().__init__()
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size

        # Load SAM2 backbone with official weights
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(self.device)

        # If MedSAM2 head checkpoint provided, load its weights (nested under 'model')
        if med_checkpoint:
            ckpt = torch.load(med_checkpoint, map_location=self.device)
            # Some checkpoints nest weights under 'model' key
            state = ckpt.get('model', ckpt)
            # load only matching keys
            self.sam.load_state_dict(state, strict=False)

        # Predictor for prompt-based segmentation
        self.predictor = SamPredictor(self.sam)

        # Precomputed full-image bounding box prompt
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

        # Replicate single-channel to three for SAM
        if C == 1:
            x = x.repeat(1, 3, 1, 1)

        # Resize for SAM input
        x_resized = torch.nn.functional.interpolate(
            x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False
        )

        logits_list = []
        # Process each sample due to predictor API
        for i in range(B):
            img_np = x_resized[i].permute(1, 2, 0).cpu().numpy()
            self.predictor.set_image(img_np)
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=self.full_box,
                multimask_output=False,
            )
            logit = torch.from_numpy(logits[0]).to(self.device)
            logits_list.append(logit)

        batch_logits = torch.stack(logits_list, dim=0).unsqueeze(1)  # (B,1,H,W)
        # Restore original resolution
        batch_logits = torch.nn.functional.interpolate(
            batch_logits, size=(H, W), mode='bilinear', align_corners=False
        )
        return batch_logits
