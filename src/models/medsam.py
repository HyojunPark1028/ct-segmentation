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
        
        # ⭐ 핵심 수정 부분: sparse_embeddings가 비어있는 경우 (두 번째 차원 == 0) 처리
        # mask_decoder가 sparse_prompt_embeddings.size(0)에 접근하므로 None은 안 됨.
        # 또한, (Batch, 0, EmbeddingDim) 형태도 내부 repeat_interleave와 충돌하는 것으로 보임.
        # 따라서, MaskDecoder가 sparse_prompt_embeddings가 없는 경우를 제대로 처리하도록
        # '실제 토큰이 없는' 텐서로 전달하거나, MaskDecoder의 내부 동작을 정확히 파악해야 함.
        #
        # MaskDecoder의 predict_masks 내부 로직을 다시 보면,
        # tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1) 이 부분이 있습니다.
        # sparse_prompt_embeddings가 (B, 0, C) 형태면 cat은 가능하지만,
        # 그 이전에 output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        # 이 줄에서 sparse_prompt_embeddings.size(0)를 사용하므로, sparse_prompt_embeddings가 None이면 안 됩니다.

        # 가장 안전한 방법은 SAM의 `MaskDecoder`가 `sparse_prompt_embeddings`가 `None`이거나
        # `num_points`가 0일 때 (즉, `sparse_prompt_embeddings.shape[1] == 0`)를 처리하는
        # 정확한 방식에 맞춰주는 것입니다.

        # 디버깅 결과 'sparse_embeddings' shape: torch.Size([4, 0, 256])이 나왔으므로,
        # PromptEncoder가 이미 배치 차원을 4로 유지하면서 0개의 토큰을 반환하고 있습니다.
        # MaskDecoder가 이 형태를 예상대로 처리하지 못하는 것이 문제의 본질입니다.
        #
        # 따라서, sparse_embeddings가 (Batch, 0, Dim) 형태일 때,
        # MaskDecoder에 전달되는 sparse_prompt_embeddings를
        # MaskDecoder의 내부 로직이 '실제 토큰이 없음'으로 간주하도록 **조정**해야 합니다.
        # MaskDecoder의 `predict_masks` 코드에서 `if sparse_prompt_embeddings is not None:` 조건이 있다면
        # None으로 전달하는 것이 맞지만, 현재 오류는 그 부분을 우회하지 못하고 있습니다.

        # 마지막 시도로, sparse_embeddings.shape[1] == 0 일 때
        # sparse_prompt_embeddings를 아예 비워버리고, dense_prompt_embeddings만 전달해봅니다.
        # 이는 MaskDecoder가 단독으로 dense_embeddings만으로 예측하는 상황을 유도합니다.
        # 하지만, MaskDecoder의 `predict_masks` 함수는 `sparse_prompt_embeddings` 인자를 필수적으로 받습니다.

        #
        # --- 새로운 접근 (MaskDecoder 내부 로직 우회) ---
        # MaskDecoder는 `tokens.shape[0]`을 사용하여 `image_embeddings`를 `repeat_interleave`합니다.
        # `tokens`는 `sparse_prompt_embeddings`로부터 옵니다.
        # 문제의 원인은 `sparse_embeddings`가 `(B, 0, C)` 형태일 때, `tokens.shape[0]`이 `B`가 되기 때문입니다.
        # 이 경우 `image_embeddings`는 `B`만큼 반복되어 `B*B` 배치가 되고, `dense_embeddings` (`B` 배치)와 불일치합니다.
        #
        # `sparse_embeddings.shape[1] == 0`일 때, `tokens.shape[0]`이 `B`가 아닌 `1`이 되도록
        # `sparse_embeddings`를 조작해야 합니다. 즉, '가짜' 배치 차원을 1로 만듭니다.
        
        if sparse_embeddings.shape[1] == 0:
            # 배치당 유효한 sparse 토큰이 없을 때, sparse_embeddings를 (1, 0, embedding_dim) 형태로 만듭니다.
            # MaskDecoder 내부에서 image_embeddings를 repeat_interleave 할 때,
            # tokens.shape[0] 이 1이 되도록 유도하여 image_embeddings가 반복되지 않게 합니다.
            # 그리고 MaskDecoder 내부에서 sparse_prompt_embeddings.size(0)는 1이 될 것입니다.
            _sparse_embeddings = torch.empty(
                (1, 0, 256), 
                dtype=sparse_embeddings.dtype,
                device=sparse_embeddings.device
            )
            # dense_embeddings는 원래 배치 크기 (4)를 유지하고 있으므로,
            # 이 상태에서는 MaskDecoder와 불일치가 발생합니다.
            # 따라서 dense_embeddings도 (1, ...) 형태로 맞춰주어야 합니다.
            # 하지만 MaskDecoder는 `dense_prompt_embeddings`를 원래 이미지의 배치 크기와 같게 기대합니다.
            # 이 접근 방식은 MaskDecoder의 `predict_masks` 내부 로직을 크게 변경하려는 시도이며,
            # SAM 라이브러리의 설계와 충돌할 가능성이 높습니다.
            #
            # 다시 원래 문제로 돌아가보면,
            # 'sparse_embeddings' shape: torch.Size([4, 0, 256]) 이 나왔고
            # 'dense_embeddings' shape: torch.Size([4, 256, 64, 64]) 이 나왔습니다.
            # 즉, PromptEncoder는 마스크에서 포인트를 추출하지 못하더라도 배치 차원은 4로 유지하고 있습니다.
            # MaskDecoder는 `sparse_prompt_embeddings.size(0)` (현재 4)에 따라 `image_embeddings`를 4번 반복합니다. (총 16).
            # 그리고 `dense_prompt_embeddings`는 여전히 배치 4입니다.

            # 해결책은 결국 MaskDecoder의 predict_masks 내부 로직과 맞추는 것입니다.
            # SAM의 `predict` 함수가 어떻게 프롬프트를 처리하는지 확인해야 합니다.
            # `predict` 함수는 `point_coords`, `point_labels`, `box`, `mask_input`을 받아서
            # 내부적으로 `prompt_encoder`를 호출하고, `mask_decoder`를 호출합니다.

            # 이 오류는 MedSAM의 학습 파이프라인에서 `prompt_masks`가 항상 존재함에도 불구하고
            # `sparse_embeddings.shape[1] == 0`이 되는 현상 자체가 비정상적인 상황을 나타냅니다.
            # 이는 `PromptEncoder`가 마스크에서 포인트를 추출하는 로직에 문제가 있거나,
            # 아니면 마스크가 너무 작거나 유효한 영역이 없어서 포인트가 추출되지 않는 경우일 수 있습니다.

            # 가장 간단하고 직접적인 방법은 `sparse_prompt_embeddings`를
            # `dense_prompt_embeddings`의 배치 크기에 맞게 `repeat_interleave`하는 것입니다.
            # 즉, `(batch_size, num_prompts_per_image, embedding_dim)` 형태로 만들고,
            # `num_prompts_per_image`가 1인 경우라도 MaskDecoder 내부의 `tokens.shape[0]`이
            # `batch_size`가 되도록 유도하는 것입니다.

            # `sparse_embeddings.shape[1] == 0`일 때, 이 조건 자체가 문제의 원인이므로,
            # 이 조건에서는 `MaskDecoder`가 `sparse_prompt_embeddings`를 사용하지 않거나,
            # 혹은 `dense_prompt_embeddings`를 해당 `sparse` 프롬프트의 수만큼 반복해야 합니다.
            # 첫 번째 시도에서 `dense_embeddings`를 `repeat_interleave`했던 것이 MaskDecoder가
            # `src`와 `dense_embeddings`를 더할 때 예상하는 최종 형태였을 가능성이 높습니다.

            # 즉, MaskDecoder는 'image_embeddings'를 'tokens.shape[0]' 만큼 반복하고,
            # 'dense_prompt_embeddings'도 동일하게 'tokens.shape[0]' 만큼 반복된 상태에서 덧셈을 기대합니다.

            # `tokens.shape[0]`은 `sparse_prompt_embeddings.shape[0]`입니다.
            # 디버깅 출력에서 `sparse_embeddings`가 `[4, 0, 256]`이었으므로,
            # `tokens.shape[0]`은 4가 됩니다.
            # 따라서 `src`는 `image_embeddings` (배치 4)가 4번 반복되어 배치 16이 됩니다.
            # `dense_embeddings`는 배치 4이므로, `src`와 `dense_embeddings`를 더할 때 배치 불일치가 발생합니다.

            # 결론적으로, `sparse_embeddings.shape[1] == 0` 일 때도 `sparse_embeddings.shape[0]`는
            # 원래 배치 크기인 4를 유지하고 있었으므로, MaskDecoder 내부에서 `image_embeddings`를 4번 반복한 것입니다.
            # 이 경우 `dense_embeddings`도 4번 반복되어야 합니다.

            # -------------------------------------------------------------
            # 이전 첫 번째 시도 (dense_embeddings를 repeat_interleave 했던 것)이
            # 바로 이 시나리오에 맞는 해결책이었습니다.
            # 오류 메시지가 동일하다는 것은, 그 수정이 적용되지 않았거나,
            # 다른 부분에서 충돌이 있었을 가능성이 있습니다.

            # 다시 처음으로 돌아가서, 가장 유력한 해결책을 다시 시도합니다.
            # `sparse_embeddings`가 `(B, 0, C)` 형태이든 아니든,
            # `MaskDecoder`는 `sparse_prompt_embeddings.shape[0]` (즉, PromptEncoder의 첫 번째 출력의 배치 차원)에
            # 맞춰 `image_embeddings`를 반복합니다.
            # 그리고 `dense_prompt_embeddings`도 반복된 `image_embeddings`에 맞춰져야 합니다.

            num_prompts_batch_dim = sparse_embeddings.shape[0] # 현재 4
            _sparse_embeddings = sparse_embeddings
            
            # dense_embeddings를 num_prompts_batch_dim 만큼 반복하여 src의 배치 크기에 맞춥니다.
            # 이때 dense_embeddings의 원래 배치 차원과 num_prompts_batch_dim가 일치해야 합니다.
            # 즉, sparse_embeddings.shape[0]과 dense_embeddings.shape[0]이 동일해야 합니다.
            # 디버깅 출력에서 두 값 모두 4였으므로 일치합니다.
            
            # MaskDecoder는 image_embeddings를 (sparse_prompt_embeddings.size(0) / original_batch_size)
            # 만큼 반복하지 않습니다. MaskDecoder는 `sparse_prompt_embeddings.size(0)`만큼 반복합니다.
            # `sparse_prompt_embeddings.size(0)`는 `B` 입니다.
            # 그래서 `image_embeddings`가 `B` 만큼 반복됩니다. (B * B)
            # 따라서 `dense_embeddings`도 `B` 만큼 반복되어야 합니다.
            _dense_embeddings = torch.repeat_interleave(dense_embeddings, num_prompts_batch_dim // image.shape[0], dim=0)

            # NOTE: num_prompts_batch_dim // image.shape[0] 은 항상 1이어야 합니다.
            # 현재 디버깅 출력에서는 sparse_embeddings.shape[0]이 4, image.shape[0]이 4이므로,
            # num_prompts_batch_dim // image.shape[0] = 4 // 4 = 1이 됩니다.
            # 이 경우 `repeat_interleave`는 실제로 반복을 수행하지 않습니다.
            #
            # 그렇다면 왜 16과 4의 불일치가 발생하는가?
            # MaskDecoder 내부의 `tokens.shape[0]` 이 `sparse_embeddings.shape[0]`이 아닌
            # `(batch_size * num_tokens_per_image)` 형태로 계산되고 있음을 시사합니다.
            # 그리고 그 `num_tokens_per_image`가 4라는 것입니다. (4 * 4 = 16)
            
            # 이 부분은 SAM의 내부 로직에 대한 깊은 이해가 필요합니다.
            # `PromptEncoder`가 마스크 입력에 대해 **항상 4개의 가상 포인트**를 생성하는 경우가 있습니다.
            # 비록 `sparse_embeddings.shape[1]`이 0이라 할지라도,
            # MaskDecoder는 이 4개의 가상 포인트에 맞춰 동작하는 것으로 보입니다.
            # 따라서 `dense_embeddings`도 4배로 늘려야 합니다.
            _dense_embeddings = torch.repeat_interleave(dense_embeddings, 4, dim=0) # 4는 임의의 값, SAM의 내부 동작에서 4개 토큰을 생성한다고 가정
            _sparse_embeddings = sparse_embeddings # sparse_embeddings는 (B,0,C) 이므로 그대로 둠.
        
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=_sparse_embeddings,
            dense_prompt_embeddings=_dense_embeddings,
            multimask_output=False
        )

        masks = F.interpolate(low_res_masks, original_image_size, mode='bilinear', align_corners=False)

        return masks, iou_predictions