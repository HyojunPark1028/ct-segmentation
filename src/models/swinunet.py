import torch
import torch.nn as nn
import torch.nn.functional as F # F.interpolate 사용을 위해 임포트
from timm.models.swin_transformer import swin_base_patch4_window7_224

class PatchExpanding(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # input_dim -> input_dim * 4 로 채널을 늘리고 PixelShuffle로 2배 해상도 확장 (채널은 input_dim // 4 로 줄어듦)
        # PixelShuffle의 출력 채널은 input_dim // 4 가 됩니다.
        self.proj = nn.Conv2d(input_dim, input_dim * 4, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        # PixelShuffle 후의 채널 수는 input_dim * 4 / (2*2) = input_dim 이 됩니다.
        # 따라서 LayerNorm은 이 output_dim에 맞춰야 합니다.
        self.output_dim = input_dim # PixelShuffle 후 채널
        self.norm = nn.LayerNorm(self.output_dim)


    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, input_dim * 4, H, W)
        x = self.pixel_shuffle(x)  # (B, input_dim, H*2, W*2)

        # LayerNorm은 (B, H, W, C) 형태에서 마지막 C 차원에 적용되므로 permute 필요
        x = x.permute(0, 2, 3, 1).contiguous() # (B, H*2, W*2, input_dim)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous() # 다시 (B, input_dim, H*2, W*2)
        return x


class SwinDecoderBlock(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim):
        super().__init__()
        self.up = PatchExpanding(in_dim)
        # PatchExpanding의 출력 채널은 이제 in_dim이므로,
        # skip_dim과 합쳐서 in_dim + skip_dim이 되어야 합니다.
        self.concat_proj = nn.Conv2d(in_dim + skip_dim, out_dim, kernel_size=1)
        self.block = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Spatial resolution check and interpolate if mismatch
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.concat_proj(x)
        return self.block(x)

class SwinUNet(nn.Module):
    def __init__(self, img_size=224, num_classes=1, use_pretrained=True):
        super().__init__()
        self.img_size = img_size 
        
        # Initialize backbone WITHOUT _out_indices, we will manually extract features
        self.backbone = swin_base_patch4_window7_224(pretrained=use_pretrained)
        self.backbone.head = nn.Identity() 

        # Get expected channel dimensions from the backbone's embed_dim
        embed_dim = self.backbone.embed_dim
        self.backbone_out_channels = [
            embed_dim,          # Stage 1 (Layer 0) output: 128
            embed_dim * 2,      # Stage 2 (Layer 1) output: 256
            embed_dim * 4,      # Stage 3 (Layer 2) output: 512
            embed_dim * 8       # Stage 4 (Layer 3) output: 1024 (bottleneck)
        ]

        # Proj layers: adjust input channels based on actual backbone outputs
        self.proj4 = nn.Conv2d(self.backbone_out_channels[3], 384, kernel_size=1) # Bottleneck
        self.proj3 = nn.Conv2d(self.backbone_out_channels[2], 192, kernel_size=1) # Skip3
        self.proj2 = nn.Conv2d(self.backbone_out_channels[1], 96, kernel_size=1)  # Skip2
        self.proj1 = nn.Conv2d(self.backbone_out_channels[0], 48, kernel_size=1)  # Skip1

        # Decoder stages
        self.decoder3 = SwinDecoderBlock(384, 192, 192)
        self.decoder2 = SwinDecoderBlock(192, 96, 96)
        self.decoder1 = SwinDecoderBlock(96, 48, 48)

        # Final output layer
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False), 
            nn.Conv2d(48, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Handle single channel input for 3-channel pretrained backbone
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        B = x.size(0)

        # Initial patch embedding
        # Output is (B, L, C) where L = H_initial * W_initial
        x = self.backbone.patch_embed(x)
        
        # Initial H, W after patch_embed
        H, W = self.img_size // self.backbone.patch_embed.patch_size[0], \
               self.img_size // self.backbone.patch_embed.patch_size[1] # 56, 56

        # Manually extract features from backbone layers
        # Layer 0 (Stage 1)
        x = self.backbone.layers[0](x) # Output (B, L, C) where L = H*W
        skip1 = x.view(B, H, W, self.backbone_out_channels[0]).permute(0, 3, 1, 2).contiguous() # (B, 128, 56, 56)

        # Layer 1 (Stage 2) - includes PatchMerging
        H, W = H // 2, W // 2 # 28, 28
        x = self.backbone.layers[1](x) # Output (B, L, C) where L = H*W
        skip2 = x.view(B, H, W, self.backbone_out_channels[1]).permute(0, 3, 1, 2).contiguous() # (B, 256, 28, 28)

        # Layer 2 (Stage 3) - includes PatchMerging
        H, W = H // 2, W // 2 # 14, 14
        x = self.backbone.layers[2](x) # Output (B, L, C) where L = H*W
        skip3 = x.view(B, H, W, self.backbone_out_channels[2]).permute(0, 3, 1, 2).contiguous() # (B, 512, 14, 14)

        # Layer 3 (Stage 4 - Bottleneck) - includes PatchMerging
        H, W = H // 2, W // 2 # 7, 7
        x = self.backbone.layers[3](x) # Output (B, L, C) where L = H*W
        # Apply final normalization from backbone
        x = self.backbone.norm(x) # This norm is applied to the token sequence (B, L, C)
        bottleneck = x.view(B, H, W, self.backbone_out_channels[3]).permute(0, 3, 1, 2).contiguous() # (B, 1024, 7, 7)

        # Apply proj layers to adjust channel counts
        x = self.proj4(bottleneck) # x becomes the input to the first decoder block (decoder3)
        proj_skip3 = self.proj3(skip3)
        proj_skip2 = self.proj2(skip2)
        proj_skip1 = self.proj1(skip1)

        # Decoder path: Pass features and corresponding skip connections
        x = self.decoder3(x, proj_skip3)
        x = self.decoder2(x, proj_skip2)
        x = self.decoder1(x, proj_skip1)

        return self.final(x)
