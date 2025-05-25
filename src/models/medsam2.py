# src/models/medsam2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from segment_anything import sam_model_registry

# MedSAM2의 핵심인 MemoryAttentionModule을 가정한 클래스입니다.
# 실제 MedSAM2 구현을 참조하여 정확한 아키텍처와 forward 로직을 구현해야 합니다.
# 이 코드는 단지 MedSAM2의 3D/비디오 처리 능력을 명시적으로 표현하기 위한 플레이스홀더입니다.
class MemoryAttentionModule(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        # MedSAM2 논문/구현에 기반한 실제 트랜스포머 레이어 구성
        # 예를 들어, Self-Attention과 Cross-Attention을 포함할 수 있습니다.
        # 여기서는 단순히 Linear 레이어로 대체하여 개념만 표현합니다.
        self.layers = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(num_layers)
        ])
        # 메모리 뱅크는 일반적으로 런타임에 동적으로 관리되거나,
        # 학습 가능한 파라미터가 아닌 외부 버퍼 형태로 존재할 수 있습니다.
        # 여기서는 설명을 위해 간단한 플레이스홀더를 둡니다.
        # 실제 구현에서는 과거 프레임/슬라이스의 임베딩을 저장하고 활용하는 로직이 필요합니다.
        self.memory_bank = None # 실제 구현에서는 텐서 형태로 관리될 것

    def forward(self, current_features: torch.Tensor, previous_features: torch.Tensor = None):
        # 3D/비디오 데이터의 경우, current_features는 현재 슬라이스/프레임의 이미지 임베딩입니다.
        # previous_features는 이전 슬라이스/프레임의 임베딩 또는 메모리 뱅크의 내용입니다.
        # MedSAM2는 이들을 이용하여 시공간적 컨텍스트를 통합합니다.

        # 이 부분은 MedSAM2의 실제 메모리 모듈 로직을 반영해야 합니다.
        # 예를 들어, previous_features가 있다면 이를 current_features와 결합하여
        # 시공간적 정보를 풍부하게 만들 수 있습니다.
        # 현재는 단순히 current_features를 통과시키는 형태로만 구현했습니다.

        x = current_features
        for layer in self.layers:
            x = F.relu(layer(x)) # 간단한 예시, 실제로는 더 복잡한 트랜스포머 블록이 사용됨

        return x # 메모리 어텐션 모듈을 거친 풍부해진 이미지 임베딩


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
        # ⭐ 추가: 메모리 어텐션 모듈을 위한 파라미터 (필요 시 조정)
        use_memory_module: bool = False, # 3D/비디오 처리를 위해 True로 설정
        memory_module_embedding_dim: int = 256, # SAM 이미지 임베딩 차원과 일치시켜야 함
        memory_module_num_heads: int = 8, # 예시 값
        memory_module_num_layers: int = 4, # 예시 값 (논문에서 4개 레이어 언급)
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.use_memory_module = use_memory_module

        # 1) SAM2 backbone 구조 생성 (가중치는 아래에서 로드)
        self.sam = sam_model_registry[model_type](checkpoint=None)
        self.sam.to(self.device)

        # 2) MedSAM2 체크포인트 로드 (backbone + head 모두 포함)
        ckpt = torch.load(checkpoint, map_location=self.device)
        state = ckpt.get("model", ckpt)
        # strict=False를 사용하여 MedSAM2 체크포인트에 SAM2에 없는 레이어가 있어도 로드 가능하게 함
        self.sam.load_state_dict(state, strict=False)

        # ⭐ 추가: 메모리 어텐션 모듈 초기화
        # 이 모듈은 MedSAM2의 3D/비디오 처리 핵심입니다.
        # sam.image_encoder.embed_dim을 사용하여 임베딩 차원을 일치시킵니다.
        if self.use_memory_module:
            self.memory_module = MemoryAttentionModule(
                embedding_dim=self.sam.image_encoder.embed_dim, # SAM 이미지 임베딩 차원
                num_heads=memory_module_num_heads,
                num_layers=memory_module_num_layers
            )
            self.memory_module.to(self.device)
        else:
            self.memory_module = None

        # 3) Positional Embedding 보간해서 입력 크기에 맞춰 줍니다.
        pe = self.sam.image_encoder.pos_embed  # 원래 shape: [1, C, H_pe_old, W_pe_old]

        # patch_embed 내부 conv의 kernel_size로 패치 크기 알아내기
        ks = self.sam.image_encoder.patch_embed.proj.kernel_size
        ps = ks[0] if isinstance(ks, (tuple, list)) else ks

        H_new = self.image_size // ps
        W_new = self.image_size // ps
        pe_new = F.interpolate(pe, size=(H_new, W_new), mode="bilinear", align_corners=False)
        self.sam.image_encoder.pos_embed = nn.Parameter(pe_new)

        # ⭐ 수정/제안: full_box를 제거하고 forward 함수에서 직접 prompt를 받도록 변경
        # 이렇게 하면 다양한 프롬프트 전략(특정 객체 바운딩 박스, 포인트 등)을 사용할 수 있습니다.
        # self.register_buffer("full_box", ...) 이 라인 삭제

    def forward(
        self,
        x: torch.Tensor,
        boxes: torch.Tensor = None, # (B, 4) 또는 (B, N, 4) 형태로 바운딩 박스 프롬프트 입력 받음
        points: torch.Tensor = None, # (B, N, 2) 또는 (B, N, 3) 형태로 포인트 프롬프트 입력 받음 (labels 포함)
        masks: torch.Tensor = None,  # (B, 1, H, W) 형태로 마스크 프롬프트 입력 받음
        # ⭐ 추가: 3D/비디오 시퀀스 처리를 위한 이전 프레임/슬라이스 임베딩 (필요 시)
        previous_frame_embeddings: torch.Tensor = None, # (B, C_emb, H_emb, W_emb) 형태
    ) -> torch.Tensor:
        """
        x: (B, C, H, W) where C=1 or 3. 입력 이미지.
        boxes: (B, N, 4) (x1, y1, x2, y2) 형태의 바운딩 박스 프롬프트. N은 박스 개수.
        points: (B, N, 2) (x, y) 좌표와 (B, N, 1) label (fore/background)로 구성될 수 있음.
        masks: (B, 1, H, W) 형태의 마스크 프롬프트.
        previous_frame_embeddings: (B, C_emb, H_emb, W_emb) 형태의 이전 프프레임/슬라이스 임베딩 (3D/비디오용).
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
        img_emb = self.sam.image_encoder(x_rs) # (B, C_emb, H_feat, W_feat)

        # ⭐ 추가: 메모리 어텐션 모듈 적용 (3D/비디오 처리 시)
        # 실제 MedSAM2의 메모리 모듈이 이미지 인코딩과 프롬프트 인코딩 사이 또는 이후에 적용될 수 있습니다.
        # 여기서는 이미지 임베딩에 적용하는 것으로 가정합니다.
        if self.use_memory_module and previous_frame_embeddings is not None:
            # 이 부분은 MedSAM2의 실제 메모리 어텐션 로직에 따라 변경되어야 합니다.
            # 예: 현재 임베딩과 이전 임베딩을 결합하여 메모리 모듈에 전달
            # 현재는 플레이스홀더입니다.
            img_emb = self.memory_module(current_features=img_emb,
                                         previous_features=previous_frame_embeddings)
        elif self.use_memory_module and previous_frame_embeddings is None:
            # 메모리 모듈을 사용하도록 설정되었지만, 이전 프레임 임베딩이 제공되지 않은 경우 경고
            print("Warning: use_memory_module is True but no previous_frame_embeddings provided.")
            img_emb = self.memory_module(current_features=img_emb) # 메모리 모듈에 현재 임베딩만 전달

        # 2) prompt encoding
        # 프롬프트가 제공되지 않은 경우, 기본적으로 full_image box를 사용하도록 fallback
        if boxes is None and points is None and masks is None:
            # (H,W) -> (H_feat, W_feat) 스케일에 맞게 프롬프트 좌표 조정 (SAM에서 내부적으로 처리하는 경우가 많지만 명시적으로 보여줌)
            # SAM 모델의 get_preprocess_shape, apply_boxes 메서드를 참고하여 정확한 스케일링 필요
            # 여기서는 입력 이미지 크기를 기준으로 full box를 생성합니다.
            boxes = torch.tensor([[0, 0, W - 1, H - 1]], dtype=torch.float, device=self.device).expand(B, -1)
            # boxes를 sam의 input_size에 맞춰 스케일링
            scale_x = self.image_size / W
            scale_y = self.image_size / H
            boxes_scaled = boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y], device=self.device)
        else:
            # 제공된 박스 프롬프트도 sam의 input_size에 맞춰 스케일링해야 합니다.
            # 사용자가 이미 스케일링된 박스를 제공한다고 가정하거나, 필요시 여기에 스케일링 로직 추가
            boxes_scaled = boxes # 사용자가 이미 image_size 스케일로 제공한다고 가정

        sparse_emb, dense_emb = self.sam.prompt_encoder(
            points=points, # 이제 외부에서 포인트 프롬프트를 받을 수 있습니다.
            boxes=boxes_scaled, # 이제 외부에서 박스 프롬프트를 받을 수 있습니다.
            masks=masks # 이제 외부에서 마스크 프롬프트를 받을 수 있습니다.
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