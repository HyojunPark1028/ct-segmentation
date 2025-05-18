# medsam2 model implementation
# Place this file under src/models/medsam2.py

import torch
import torch.nn as nn
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

class MedSAM2(nn.Module):
    """
    MedSAM2: full-model checkpoint for medical image segmentation
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
        # device 설정
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size

        # SAM2 backbone 초기화 (checkpoint=None 으로 가중치 로드 건너뜀)
        self.sam = sam_model_registry[model_type](checkpoint=None)
        self.sam.to(self.device)

        # MedSAM2 전체 checkpoint 로드 (state 안에 'model' key인 경우 지원)
        ckpt = torch.load(checkpoint, map_location=self.device)
        state = ckpt.get('model', ckpt)
        # backbone + head 모두 strict=False 로 불러오기
        self.sam.load_state_dict(state, strict=False)

        # Prompt 기반 세그멘테이션 예측기
        self.predictor = SamPredictor(self.sam)

        # 전체 이미지 바운딩 박스 프롬프트
        self.full_box = np.array([0, 0, image_size - 1, image_size - 1])[None, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.to(self.device)
        # single-channel replicate to 3-channel
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        # resize to SAM input size
        x_resized = torch.nn.functional.interpolate(
            x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False
        )
        logits_list = []
        for i in range(B):
            img_np = x_resized[i].permute(1, 2, 0).cpu().numpy()
            self.predictor.set_image(img_np)
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=self.full_box,
                multimask_output=False,
            )
            logits_list.append(torch.from_numpy(logits[0]).to(self.device))
        batch_logits = torch.stack(logits_list, dim=0).unsqueeze(1)  # (B, 1, H', W')
        # resize back to original
        batch_logits = torch.nn.functional.interpolate(
            batch_logits, size=(H, W), mode='bilinear', align_corners=False
        )
        return batch_logits
