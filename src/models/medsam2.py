import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from segment_anything import sam_model_registry

class MedSAM2(nn.Module):
    """
    MedSAM2: Differentiable prompt-driven segmentation using SAM2 backbone
    Supports end-to-end training with a single checkpoint (backbone + head).

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

        # Initialize SAM2 backbone without loading weights
        self.sam = sam_model_registry[model_type](checkpoint=None)
        self.sam.to(self.device)

        # Load full MedSAM2 checkpoint (backbone + head)
        ckpt = torch.load(checkpoint, map_location=self.device)
        state = ckpt.get('model', ckpt)
        # Load state_dict with strict=False to allow missing keys
        self.sam.load_state_dict(state, strict=False)

        # Adapt positional embeddings to match resized input
        # pos_embed shape: [1, embed_dim, H_pe, W_pe]
        pe = self.sam.image_encoder.pos_embed  # nn.Parameter
        # Determine patch size used by SAM (kernel_size from patch_embed conv)
        ks = self.sam.image_encoder.patch_embed.kernel_size
        ps = ks[0] if isinstance(ks, tuple) else ks
        H_new = self.image_size // ps
        W_new = self.image_size // ps
        # Interpolate pos_embed
        pe_new = F.interpolate(pe, size=(H_new, W_new), mode='bilinear', align_corners=False)
        # Replace pos_embed parameter
        self.sam.image_encoder.pos_embed = nn.Parameter(pe_new)

        # Prepare full-image box prompt for prompt_encoder
        # shape: (1, 4) -> [x0, y0, x1, y1]
        self.register_buffer(
            'full_box',
            torch.tensor([[0, 0, image_size - 1, image_size - 1]], dtype=torch.float)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fully differentiable forward.
        Args:
            x: (B, C, H, W) -- input slices, C=1 (grayscale) or 3 (RGB)
        Returns:
            logits: (B, 1, H, W) -- raw mask logits
        """
        B, C, H, W = x.shape
        x = x.to(self.device)
        # Replicate grayscale to RGB
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        # Resize to SAM input scale
        x_resized = F.interpolate(
            x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False
        )

        # 1) Image encoding
        image_embeddings = self.sam.image_encoder(x_resized)

        # 2) Prompt encoding: full-image box for each sample
        boxes = self.full_box.expand(B, -1)  # (B,4)
        sparse_emb, dense_emb = self.sam.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None
        )

        # 3) Mask decoding
        low_res_logits, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
        )

        # 4) Upsample to original resolution
        logits = F.interpolate(
            low_res_logits, size=(H, W), mode='bilinear', align_corners=False
        )
        return logits
