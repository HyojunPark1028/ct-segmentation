import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.build_sam import sam_model_registry

class MedSAM(nn.Module):
    def __init__(self, checkpoint: str, out_channels: int = 1):
        super().__init__()
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)

        # Enable gradients for Image Encoder
        for p in self.sam.image_encoder.parameters():
            p.requires_grad = True

        # Freeze Prompt Encoder
        for p in self.sam.prompt_encoder.parameters():
            p.requires_grad = False
        
        # Enable gradients for Mask Decoder
        for p in self.sam.mask_decoder.parameters():
            p.requires_grad = True
        
    def forward(self, image: torch.Tensor, prompt_masks: torch.Tensor):
        original_image_size = image.shape[2:]

        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        image_embeddings = self.sam.image_encoder(image)

        resized_prompt_masks = F.interpolate(
            prompt_masks.float(),
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        )

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=resized_prompt_masks
        )
        
        _sparse_embeddings = sparse_embeddings
        _dense_embeddings = dense_embeddings

        # ⭐ 핵심 수정 부분: MaskDecoder의 내부 동작에 맞춰 image_embeddings와 image_pe를 조정
        # MaskDecoder는 sparse_prompt_embeddings의 첫 번째 차원(배치 + 토큰 수)을 기준으로
        # image_embeddings와 image_pe를 확장합니다.
        # 디버깅 출력에서 sparse_embeddings.shape[0]은 4였지만, 실제 MaskDecoder 내부에서는
        # image_embeddings를 4배(16)로 늘립니다. 이는 SAM이 마스크에 대해 4개의 implicit token을
        # 사용한다고 가정한 것으로 보입니다.

        # MaskDecoder의 predict_masks 함수에서
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        # pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        #
        # 이 `tokens.shape[0]`은 `sparse_prompt_embeddings.size(0)`와 같다고 보고,
        # 이 값이 `Batch * Num_Implicit_Tokens_Per_Image` (4 * 4 = 16)으로 계산되는 것이 현재 문제.
        #
        # 따라서, sparse_embeddings.shape[1] == 0 이든 아니든,
        # `image_embeddings`와 `image_pe`가 `16`의 배치 차원에 맞춰져야 합니다.

        # MaskDecoder의 `predict_masks`는 `image_embeddings`와 `image_pe`를 원본 그대로 받아서
        # 내부적으로 `repeat_interleave`를 수행합니다.
        # 문제는 `sparse_prompt_embeddings`의 형태가 `(B, 0, C)`일 때, `tokens.shape[0]`이 `B * 4`가 되어버리는 것입니다.
        # 그리고 `dense_prompt_embeddings`는 `B`만 유지합니다.
        # `image_pe`도 마찬가지로 `B`만 유지하여 `transformer.py`에서 `keys + key_pe` 오류가 발생합니다.

        # 이 문제를 해결하는 가장 직접적인 방법은 `segment_anything` 라이브러리의 `sam.predict` 함수처럼
        # 프롬프트 인코딩 및 마스크 디코딩을 위한 전처리 과정을 수행하는 것입니다.
        # 하지만 `MedSAM`은 모듈을 직접 사용하고 있으므로, `mask_decoder`에 전달되는 인자들을
        # `predict_masks`가 기대하는 형태로 명확하게 맞춰야 합니다.

        # `sam.mask_decoder`는 이미 `image_embeddings`와 `image_pe`를 `repeat_interleave`합니다.
        # 문제는 `sparse_embeddings.shape[1] == 0` 일 때, `tokens.shape[0]`가
        # `sparse_embeddings.shape[0]` (4)가 아니라 `batch_size * num_implicit_tokens_from_mask_prompts` (4 * 4 = 16)으로
        # 해석되는 것입니다.

        # 이 상황은 `segment_anything` 라이브러리 내부의 `PromptEncoder`와 `MaskDecoder` 간의 미묘한 인터페이스 불일치입니다.
        # 가장 강력한 해결책은 `PromptEncoder`가 `masks` 입력을 받을 때,
        # `sparse_embeddings`가 `(Batch, 0, Dim)`이 아니라 `(Batch, 1, Dim)` 또는
        # `(Batch, NumPointsFromMask, Dim)`으로 생성되도록 보장하거나,
        # 아니면 `MaskDecoder`가 `sparse_prompt_embeddings.shape[1] == 0`일 때,
        # `tokens.shape[0]`을 `batch_size` (4)로만 설정하도록 만드는 것입니다.

        # MedSAM 모델의 `forward` 메서드에서 이 문제를 우회하기 위해,
        # `sparse_embeddings`가 실제로 비어있는 경우 (`shape[1] == 0`),
        # MaskDecoder에 `sparse_prompt_embeddings`를 `None`으로 전달하는 이전 시도를 다시 하고,
        # 동시에 `dense_embeddings`도 `None`으로 전달합니다.
        # 그리고 MaskDecoder는 `image_embeddings`와 `image_pe`만을 사용하여 마스크를 생성해야 합니다.
        # 하지만, MaskDecoder의 predict_masks는 `dense_prompt_embeddings`가 필수 인자입니다.

        # 그렇다면 `sparse_embeddings.shape[1] == 0`일 때,
        # `sparse_embeddings`를 `None`이 아닌 `(batch_size, 0, embedding_dim)` 형태로 전달하되,
        # `dense_embeddings`는 `(batch_size * num_implicit_tokens_per_image, C, H, W)` 형태로 맞춰야 합니다.
        # 이 `num_implicit_tokens_per_image`가 4로 보입니다.

        if sparse_embeddings.shape[1] == 0:
            # 이 경우 PromptEncoder는 마스크에서 유효한 포인트를 추출하지 못했지만,
            # MaskDecoder는 여전히 이미지당 4개의 프롬프트 토큰을 기대하는 것으로 보입니다.
            # 따라서 dense_embeddings도 4배로 늘려야 합니다.
            _dense_embeddings = torch.repeat_interleave(dense_embeddings, 4, dim=0)
            # sparse_embeddings는 (B, 0, C) 형태 그대로 유지합니다.
            _sparse_embeddings = sparse_embeddings
        else:
            _sparse_embeddings = sparse_embeddings
            _dense_embeddings = dense_embeddings

        # 이 시점에서 _sparse_embeddings는 (B, X, C) 형태이고 (X는 0이 될 수 있음)
        # _dense_embeddings는 (B*4, C_dense, H_dense, W_dense) 형태로 가정합니다.
        # MaskDecoder의 `predict_masks`는
        # `src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)`
        # `pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)`
        #
        # 여기서 `tokens.shape[0]`은 `_sparse_embeddings.size(0)`이 아니라,
        # `_sparse_embeddings`가 `(B, X, C)`일 때 `B * X`가 됩니다.
        # 만약 X가 0이면 `tokens.shape[0]`이 0이 되어 `repeat_interleave`가 실패합니다.
        #
        # 이전 디버깅 출력 (`'sparse_embeddings' shape: torch.Size([4, 0, 256])`)에서
        # `simulated_tokens_shape_0`이 `4`로 나왔는데, 이는 `sparse_embeddings.shape[0]`를 의미합니다.
        # 그런데 오류는 `tensor a (16)`이므로, MaskDecoder 내부에서 `tokens.shape[0]`은 `16`으로 계산되고 있다는 뜻입니다.
        #
        # 이는 `mask_decoder.py`의 `predict_masks` 함수 내에서 `tokens`를 생성하는 방식이
        # 우리가 예측하는 것과 다르다는 것을 시사합니다.

        #
        # --- 최종 접근: MaskDecoder가 기대하는 특정 프롬프트 형태를 강제 ---
        # `MedSAM`의 구현 목표를 고려할 때, 항상 단일 마스크 프롬프트가 주어진다고 가정합니다.
        # `segment_anything`의 `MaskDecoder`는 `image_embeddings`와 `sparse/dense` 프롬프트의
        # 배치 차원을 `B * N` 형태로 기대합니다. 여기서 `B`는 이미지 배치 크기, `N`은 이미지당 프롬프트 수입니다.
        # `N`이 4로 고정된 것으로 보입니다.

        # MaskDecoder는 항상 4개의 프롬프트 임베딩(IoU 토큰, Mask 토큰)과 `sparse_prompt_embeddings`를 합칩니다.
        # output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        # output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        # tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # 이 `sparse_prompt_embeddings.size(0)`가 바로 `B`입니다. (원래 이미지 배치 크기)
        # 이 줄에서 `sparse_prompt_embeddings`가 `None`이면 `AttributeError`가 발생합니다.
        # `sparse_prompt_embeddings`가 `(B, 0, C)` 형태면 `size(0)`는 `B`가 되어 `expand`는 가능합니다.
        #
        # 그 다음 `tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)`
        # `output_tokens`는 `(B, 2, C)` 형태가 됩니다. (2는 iou_token과 mask_token 개수)
        # `sparse_prompt_embeddings`는 `(B, 0, C)` 형태입니다.
        # `tokens`는 `(B, 2, C)` 형태가 됩니다.

        # `transformer.py`의 `forward` 함수로 넘어갈 때, `queries`는 `tokens`입니다.
        # `queries.shape[0]`은 `B` (4)입니다.
        # `keys`는 `image_embeddings_reshaped` (B * H*W, C) 입니다.
        # `keys.shape[0]`은 `B * H * W`입니다. (예: 4 * 64 * 64) -> 이 값이 16이 아님.
        #
        # `transformer.py`의 CrossAttentionBlock은 쿼리와 키의 첫 번째 차원을 맞추는 것이 아니라,
        # `q=queries + query_pe` 와 `k=keys + key_pe`에서 오류가 났습니다.

        # `queries`는 `tokens` (B, Num_tokens, C)
        # `query_pe`는 `pos_tokens` (B, Num_tokens, C)
        # `keys`는 `src` (B_repeated, C, H, W)를 reshape한 것 (B_repeated, H*W, C)
        # `key_pe`는 `pos_src`를 reshape한 것 (B_repeated, H*W, C)

        # 오류 메시지는 `keys`와 `key_pe`를 더할 때 배치 차원이 다르다는 것입니다.
        # `predict_masks`에서 `pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)`
        # 이 `tokens.shape[0]`은 `sparse_prompt_embeddings.size(0)`입니다. (즉 4)
        # 따라서 `pos_src`의 배치 크기는 `image_pe.shape[0] * sparse_prompt_embeddings.size(0)`
        # = `4 * 4 = 16`이 됩니다.
        # `image_pe`의 배치 크기는 원래 4입니다.

        # `MaskDecoder`는 `predict_masks` 내부에서 `image_embeddings`와 `image_pe`를 직접 반복시킵니다.
        # 즉, `MaskDecoder` 외부에서 이를 조작할 필요가 없습니다.

        # `RuntimeError: The size of tensor a (16) must match the size of tensor b (4) at non-singleton dimension 0`
        # `k = keys + key_pe`에서 오류가 났고, `keys`가 `16`, `key_pe`가 `4`입니다.
        # `keys`는 `image_embeddings`를 반복시킨 결과이고 (batch 16),
        # `key_pe`는 `image_pe` (batch 4) 입니다.

        # `predict_masks` 함수 내에서 `pos_src`가 `image_pe`를 반복시킨 텐서입니다.
        # 이 `pos_src`가 `key_pe`로 전달되어야 합니다.
        #
        # MaskDecoder의 `__init__`을 보면 `self.transformer = LightWeightedMultiHeadAttention()` 이 있습니다.
        # `predict_masks` 함수는 `self.transformer.forward_sam`을 호출하는데,
        # `self.transformer.forward_sam(sparse_prompt_embeddings, dense_prompt_embeddings, image_embeddings_reshaped)`
        # 이 함수가 `keys`와 `key_pe`를 받습니다.

        # `image_embeddings_reshaped`는 `image_embeddings`를 `repeat_interleave`한 `src`를 reshape한 것입니다.
        # `key_pe`는 `pos_src`여야 합니다.

        # `MaskDecoder`의 `predict_masks`의 마지막 줄은
        # `return masks, iou_predictions`
        # `low_res_masks, iou_predictions = self.sam.mask_decoder(...)`

        # 문제는 `sam.mask_decoder(...)`에 전달하는 인자가 잘못되었다는 것입니다.
        # `image_pe` 대신 `pos_src`에 해당하는 텐서를 전달해야 합니다.
        # 하지만 `pos_src`는 `MaskDecoder` 내부에서 생성됩니다.

        # 이 문제는 `segment_anything` 라이브러리의 `predict_masks` 함수 자체의 버그이거나,
        # `sparse_prompt_embeddings`가 `(B, 0, C)` 형태일 때 처리 로직의 결함입니다.

        # 마지막 시도: `sparse_embeddings.shape[1] == 0`일 때,
        # `_sparse_embeddings`를 `None`이 아닌 `(B, 1, C)` (즉, 각 이미지에 대해 단일 더미 토큰) 형태로 전달합니다.
        # 이렇게 하면 `tokens.shape[0]`이 `B` (4)가 되어 `repeat_interleave`가 4 * 1 = 4가 되고,
        # `dense_embeddings` (4)와 일치할 것입니다.

        if sparse_embeddings.shape[1] == 0:
            # MaskDecoder가 예상하는 최소한의 sparse_prompt_embeddings를 제공
            # (batch_size, num_tokens, embedding_dim) -> num_tokens = 1 (더미)
            _sparse_embeddings = torch.zeros(
                (sparse_embeddings.shape[0], 1, 256),
                dtype=sparse_embeddings.dtype,
                device=sparse_embeddings.device
            )
            _dense_embeddings = dense_embeddings
        else:
            _sparse_embeddings = sparse_embeddings
            _dense_embeddings = dense_embeddings

        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=_sparse_embeddings,
            dense_prompt_embeddings=_dense_embeddings,
            multimask_output=False
        )

        masks = F.interpolate(low_res_masks, original_image_size, mode='bilinear', align_corners=False)

        return masks, iou_predictions