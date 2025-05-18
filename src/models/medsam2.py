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

        # SAM2 전체 모델 체크포인트 로드
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(self.device)

        # Prompt 기반 세그멘테이션 예측기
        self.predictor = SamPredictor(self.sam)

        # 전체 이미지 바운딩 박스 프롬프트
        self.full_box = np.array([0, 0, image_size - 1, image_size - 1])[None, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W) -- C=1 or 3
        Returns:
            logits: Tensor of shape (B, 1, H, W)
        """
        B, C, H, W = x.shape
        x = x.to(self.device)

        # 흑백 이미지는 3채널로 복제
        if C == 1:
            x = x.repeat(1, 3, 1, 1)

        # SAM 입력 크기로 리사이즈
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
            logit = torch.from_numpy(logits[0]).to(self.device)
            logits_list.append(logit)

        # 배치 텐서로 스택 후 원해상도로 되돌리기
        batch_logits = torch.stack(logits_list, dim=0).unsqueeze(1)
        batch_logits = torch.nn.functional.interpolate(
            batch_logits, size=(H, W), mode='bilinear', align_corners=False
        )
        return batch_logits
