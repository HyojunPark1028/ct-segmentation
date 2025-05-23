import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import swin_base_patch4_window7_224
from einops import rearrange
import numpy as np

class PatchExpanding(nn.Module):
    def __init__(self, input_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim // 2
        self.linear = nn.Linear(input_dim, 2 * input_dim, bias=False)
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        x = self.linear(x)
        # Rearrange from (B, H*W, 2C) to (B, 2H*2W, C/2)
        B, L, C = x.shape
        H = W = int(L**0.5)
        
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B, -1, C//4)
        x = self.norm(x)
        
        return x

class SwinDecoderBlock(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim):
        super().__init__()
        self.up = PatchExpanding(in_dim)
        
        # Concatenation projection
        concat_dim = (in_dim // 2) + skip_dim
        self.concat_proj = nn.Linear(concat_dim, out_dim)
        
        # Swin-style blocks with LayerNorm
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        
        # Simple MLP block
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim * 4, out_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x, skip):
        # Upsample
        x = self.up(x)  # (B, 4*L, C/2)
        
        # Match spatial dimensions if needed
        if x.shape[1] != skip.shape[1]:
            # Interpolate to match skip connection size
            B, L_x, C_x = x.shape
            B, L_skip, C_skip = skip.shape
            H_x = W_x = int(L_x**0.5)
            H_skip = W_skip = int(L_skip**0.5)
            
            x = x.view(B, H_x, W_x, C_x).permute(0, 3, 1, 2)
            x = F.interpolate(x, size=(H_skip, W_skip), mode='bilinear', align_corners=False)
            x = x.permute(0, 2, 3, 1).view(B, H_skip*W_skip, C_x)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=-1)
        x = self.concat_proj(x)
        
        # Apply transformer-style blocks
        residual = x
        x = self.norm1(x)
        x = residual + x  # Simple residual connection
        
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        
        return x

class SwinUNet(nn.Module):
    def __init__(self, img_size=224, num_classes=1, use_pretrained=True):
        super().__init__()
        
        # Load pretrained Swin Transformer backbone
        self.backbone = swin_base_patch4_window7_224(pretrained=use_pretrained)
        
        # Get the feature dimensions from backbone
        # Swin-B: [128, 256, 512, 1024]
        self.feature_dims = [128, 256, 512, 1024]
        
        # Projection layers to standardize dimensions
        self.proj4 = nn.Linear(1024, 512)  # Bottleneck
        self.proj3 = nn.Linear(512, 256)   # Skip connection 3
        self.proj2 = nn.Linear(256, 128)   # Skip connection 2
        self.proj1 = nn.Linear(128, 64)    # Skip connection 1
        
        # Decoder blocks
        self.decoder3 = SwinDecoderBlock(512, 256, 256)  # 512->256, skip:256, out:256
        self.decoder2 = SwinDecoderBlock(256, 128, 128)  # 256->128, skip:128, out:128
        self.decoder1 = SwinDecoderBlock(128, 64, 64)    # 128->64,  skip:64,  out:64
        
        # Final upsampling and classification
        self.final_expand = PatchExpanding(64)  # 64 -> 32
        self.final_norm = nn.LayerNorm(32)
        self.classifier = nn.Linear(32, num_classes)
        
        # Additional upsampling to original size
        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        B = x.size(0)
        
        # Extract features using backbone
        x = self.backbone.patch_embed(x)
        if self.backbone.patch_embed.norm is not None:
            x = self.backbone.patch_embed.norm(x)
        
        # Get multi-scale features
        features = []
        for layer in self.backbone.layers:
            features.append(x)
            x = layer(x)
        
        # Features: [skip1, skip2, skip3, bottleneck]
        skip1, skip2, skip3, bottleneck = features
        
        # Apply projections to standardize dimensions
        bottleneck = self.proj4(bottleneck)  # 1024 -> 512
        skip3 = self.proj3(skip3)            # 512 -> 256
        skip2 = self.proj2(skip2)            # 256 -> 128
        skip1 = self.proj1(skip1)            # 128 -> 64
        
        # Decoder path
        x = self.decoder3(bottleneck, skip3)  # 512 -> 256
        x = self.decoder2(x, skip2)           # 256 -> 128
        x = self.decoder1(x, skip1)           # 128 -> 64
        
        # Final processing
        x = self.final_expand(x)              # 64 -> 32
        x = self.final_norm(x)
        
        # Convert to spatial format for final processing
        B, L, C = x.shape
        H = W = int(L**0.5)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Final upsampling to original resolution
        x = self.final_upsample(x)  # Upsample by 4x to get original size
        
        # Classification
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.classifier(x)      # (B, H, W, num_classes)
        x = x.permute(0, 3, 1, 2)  # (B, num_classes, H, W)
        
        return x

# Simplified version for testing
class SwinUNetSimple(nn.Module):
    def __init__(self, img_size=224, num_classes=1, use_pretrained=True):
        super().__init__()
        
        # Load pretrained backbone
        self.backbone = swin_base_patch4_window7_224(pretrained=use_pretrained)
        
        # Simple decoder with conv layers (for stability)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 2, 2),  # Upsample
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 256, 2, 2),   # Upsample
            nn.GroupNorm(16, 256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 2, 2),   # Upsample
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 2, 2),    # Upsample
            nn.GroupNorm(4, 64),
            nn.GELU(),
            nn.Conv2d(64, num_classes, 1)         # Final classification
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        B = x.size(0)
        
        # Encoder
        x = self.backbone.patch_embed(x)
        if self.backbone.patch_embed.norm is not None:
            x = self.backbone.patch_embed.norm(x)
        
        # Pass through all backbone layers
        for layer in self.backbone.layers:
            x = layer(x)
        
        # Convert to spatial format
        # x shape: (B, L, C) where L = H*W, C = 1024
        B, L, C = x.shape
        H = W = int(L**0.5)  # Assuming square feature maps
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Decode
        x = self.decoder(x)
        
        return x