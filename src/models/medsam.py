import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.build_sam import sam_model_registry
import segmentation_models_pytorch as smp

class MedSAM(nn.Module):
    def __init__(self, sam_checkpoint: str, unet_checkpoint: str, out_channels: int = 1):
        super().__init__()
        self.sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)

        # SAM: image encoder + mask decoder만 학습
        for p in self.sam.image_encoder.parameters():
            p.requires_grad = True
        for p in self.sam.prompt_encoder.parameters():
            p.requires_grad = False
        for p in self.sam.mask_decoder.parameters():
            p.requires_grad = True

        # U-Net: pretrained checkpoint로 초기화 (encoder_weights=None)
        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,  # ImageNet pretrained 안 쓰고 내가 학습한 모델만 사용
            in_channels=1,
            classes=1,
            activation=None
        )
        state_dict = torch.load(unet_checkpoint)
        self.unet.model.load_state_dict(state_dict)
        self.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad = False

    def forward(self, image: torch.Tensor):
        """
        Arguments:
            image: (B, 1 or 3, H, W)
        Returns:
            masks: (B, 1, H, W)
            iou_predictions: (B, 1)
        """
        B = image.shape[0]
        original_image_size = image.shape[2:]

        # Step 1: CT 이미지가 1채널이면 3채널로 복제
        if image.shape[1] == 1:
            image_rgb = image.repeat(1, 3, 1, 1)  # (B, 3, H, W)
        else:
            image_rgb = image

        # Step 2: U-Net을 통한 초기 마스크 예측 → prompt 용도
        with torch.no_grad():
            initial_mask = self.unet(image)  # (B, 1, H, W), 원본 크기
            initial_mask_bin = (initial_mask > 0.5).float()

        resized_prompt_mask = F.interpolate(
            initial_mask_bin, size=(256, 256), mode='bilinear', align_corners=False
        )

        # Step 3: SAM image encoder
        image_embeddings = self.sam.image_encoder(image_rgb)  # (B, C, H', W')

        # Step 4: SAM prompt encoder
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=None, masks=resized_prompt_mask
        )

        # Step 5: SAM positional encoding
        image_pe = self.sam.prompt_encoder.get_dense_pe()  # (1, C, H', W')
        image_pe = image_pe.expand(B, -1, -1, -1)

        # Step 6: SAM mask decoder
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        # Step 7: 예측 마스크를 원본 크기로 업샘플링
        masks = F.interpolate(
            low_res_masks, size=original_image_size, mode='bilinear', align_corners=False
        )

        return masks, iou_predictions
