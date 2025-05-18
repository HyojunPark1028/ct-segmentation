# src/models/medsam2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from segment_anything import sam_model_registry

class MedSAM2(nn.Module):
    """
    MedSAM2: differentiable prompt-driven segmentation using SAM2 backbone.
    Supports end-to-end training with a single MedSAM2 checkpoint.
    """

    def __init__(
        self,
        checkpoint: str,            # weights/MedSAM2_latest.pt
        model_type: str = "vit_b",
        image_size: int = 512,
        device: str = None,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size

        # 1) SAM2 backbone만 구조 생성 (가중치는 아래에서 로드)
        self.sam = sam_model_registry[model_type](checkpoint=None)
        self.sam.to(self.device)

        # 2) MedSAM2 체크포인트 로드 (backbone + head 모두 포함)
        ckpt = torch.load(checkpoint, map_location=self.device)
        state = ckpt.get("model", ckpt)
        self.sam.load_state_dict(state, strict=False)

        # 3) Positional Embedding 보간해서 입력 크기에 맞춰 줍니다.
        pe = self.sam.image_encoder.pos_embed  # 원래 shape: [1, C, H_pe_old, W_pe_old]

        # patch_embed 내부 conv의 kernel_size로 패치 크기 알아내기
        ks = self.sam.image_encoder.patch_embed.proj.kernel_size
        ps = ks[0] if isinstance(ks, (tuple, list)) else ks

        H_new = self.image_size // ps
        W_new = self.image_size // ps
        pe_new = F.interpolate(pe, size=(H_new, W_new), mode="bilinear", align_corners=False)
        self.sam.image_encoder.pos_embed = nn.Parameter(pe_new)

        # 4) full-image box prompt 한 번만 만들어 둡니다
        self.register_buffer(
            "full_box",
            torch.tensor([[0, 0, image_size - 1, image_size - 1]], dtype=torch.float),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) where C=1 or 3
        returns: (B, 1, H, W) logits
        """
        B, C, H, W = x.shape
        x = x.to(self.device)

        # 흑백이면 3채널로 복제
        if C == 1:
            x = x.repeat(1, 3, 1, 1)

        # SAM 입력 해상도로 리사이즈
        x_rs = F.interpolate(x, size=(self.image_size, self.image_size),
                             mode="bilinear", align_corners=False)

        # 1) image encoding
        img_emb = self.sam.image_encoder(x_rs)

        # 2) prompt encoding (full-image box)
        boxes = self.full_box.expand(B, -1)  # (B,4)
        sparse_emb, dense_emb = self.sam.prompt_encoder(
            points=None, boxes=boxes, masks=None
        )

        # 3) mask decoding
        low_res_logits, _ = self.sam.mask_decoder(
            image_embeddings=img_emb,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
        )

        # 4) 원래 해상도로 업샘플
        out = F.interpolate(low_res_logits, size=(H, W),
                            mode="bilinear", align_corners=False)
        return out
