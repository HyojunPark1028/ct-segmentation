import torch
import torch.nn as nn
import torch.nn.functional as F # F.interpolate 사용을 위해 임포트
from timm.models.swin_transformer import swin_base_patch4_window7_224
from einops import rearrange 

class PatchExpanding(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2 * input_dim)
        self.norm = nn.LayerNorm(2 * input_dim)
        self.output_dim = input_dim // 2 

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous() # (B, H, W, C)
        x = self.linear(x)                     # (B, H, W, 2C)
        x = self.norm(x)
        x = rearrange(x, 'b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)',
                      p1=2, p2=2, c_out=self.output_dim)
        return x

class SwinDecoderBlock(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim):
        super().__init__()
        self.up = PatchExpanding(in_dim) 
        self.concat_proj = nn.Conv2d((in_dim // 2) + skip_dim, out_dim, kernel_size=1)
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
        
        # Initialize backbone with _out_indices to get intermediate features for skip connections
        # This will populate self.backbone.feature_info
        self.backbone = swin_base_patch4_window7_224(pretrained=use_pretrained, _out_indices=(0, 1, 2, 3))
        self.backbone.head = nn.Identity() 

        # Get actual output channel dimensions from the backbone's feature_info
        # This is the most robust way to get the exact channel counts for each stage
        self.backbone_out_channels = [info['chs'] for info in self.backbone.feature_info]

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

        # Get intermediate features from the backbone using forward_features with _out_indices
        # `features` will be a list of (B, L, C) tensors, where L is H*W
        features = self.backbone.forward_features(x)

        # Calculate spatial dimensions for each stage's output
        # Based on original img_size=224 and patch_size=4
        H_s1, W_s1 = self.img_size // 4, self.img_size // 4 # 56x56 for features[0]
        H_s2, W_s2 = self.img_size // 8, self.img_size // 8 # 28x28 for features[1]
        H_s3, W_s3 = self.img_size // 16, self.img_size // 16 # 14x14 for features[2]
        H_s4, W_s4 = self.img_size // 32, self.img_size // 32 # 7x7 for features[3]

        # Reshape (B, L, C) to (B, H, W, C) and then permute to (B, C, H, W)
        # Use .contiguous() after permute for proper memory layout
        skip1 = features[0].view(B, H_s1, W_s1, self.backbone_out_channels[0]).permute(0, 3, 1, 2).contiguous()
        skip2 = features[1].view(B, H_s2, W_s2, self.backbone_out_channels[1]).permute(0, 3, 1, 2).contiguous()
        skip3 = features[2].view(B, H_s3, W_s3, self.backbone_out_channels[2]).permute(0, 3, 1, 2).contiguous()
        bottleneck = features[3].view(B, H_s4, W_s4, self.backbone_out_channels[3]).permute(0, 3, 1, 2).contiguous()

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
