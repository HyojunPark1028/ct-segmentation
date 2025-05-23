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
        
        # Remove the classification head
        self.backbone.head = nn.Identity()
        
        # Swin-B feature dimensions: [128, 256, 512, 1024] at different stages
        # But we need to check actual dimensions by running forward pass
        
        # Create a simple conv-based decoder for now to fix dimension issues
        self.decoder = nn.ModuleDict({
            'up4': nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            'conv4': nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            'up3': nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            'conv3': nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            'up2': nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            'conv2': nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            'up1': nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            'conv1': nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            'final': nn.Conv2d(64, num_classes, kernel_size=1)
        })
        
        # Additional upsampling if needed
        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Handle grayscale images - convert to RGB if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        B, _, H, W = x.shape
        
        # Get features from Swin backbone
        # Use the forward_features method to get intermediate features
        features = self.extract_features(x)
        
        # Use the deepest feature for decoding
        x = features  # This should be (B, 1024, H/32, W/32)
        
        # Decoder path
        x = self.decoder['up4'](x)      # -> (B, 512, H/16, W/16)
        x = self.decoder['conv4'](x)
        
        x = self.decoder['up3'](x)      # -> (B, 256, H/8, W/8)
        x = self.decoder['conv3'](x)
        
        x = self.decoder['up2'](x)      # -> (B, 128, H/4, W/4)
        x = self.decoder['conv2'](x)
        
        x = self.decoder['up1'](x)      # -> (B, 64, H/2, W/2)
        x = self.decoder['conv1'](x)
        
        # Final classification
        x = self.decoder['final'](x)    # -> (B, num_classes, H/2, W/2)
        
        # Upsample to original size
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x
    
    def extract_features(self, x):
        """Extract features from Swin backbone"""
        # Debug: Print input shape
        print(f"Input shape: {x.shape}")
        
        # Patch embedding
        x = self.backbone.patch_embed(x)
        print(f"After patch_embed: {x.shape}")
        
        if self.backbone.patch_embed.norm is not None:
            x = self.backbone.patch_embed.norm(x)
            print(f"After patch_embed norm: {x.shape}")
        
        # Pass through all layers
        for i, layer in enumerate(self.backbone.layers):
            x = layer(x)
            print(f"After layer {i}: {x.shape}")
        
        # Check if we have the norm layer
        if hasattr(self.backbone, 'norm') and self.backbone.norm is not None:
            x = self.backbone.norm(x)
            print(f"After final norm: {x.shape}")
        
        # Handle different output shapes
        if len(x.shape) == 3:  # (B, L, C)
            B, L, C = x.shape
            H = W = int(L**0.5)
            x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        elif len(x.shape) == 2:  # (B, C) - pooled output
            # This happens when the model has been pooled/averaged
            # We need to handle this case differently
            B, C = x.shape
            # Create a spatial feature map by reshaping
            # Assuming we want a small spatial size like 7x7
            spatial_size = 7
            x = x.view(B, C, 1, 1).expand(B, C, spatial_size, spatial_size)
        elif len(x.shape) == 4:  # (B, C, H, W) - already in correct format
            pass
        else:
            raise ValueError(f"Unexpected output shape from backbone: {x.shape}")
        
        print(f"Final output shape: {x.shape}")
        return x