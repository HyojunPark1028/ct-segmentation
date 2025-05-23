import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import swin_base_patch4_window7_224
from einops import rearrange
import numpy as np

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Identity()  # Simplified for this implementation
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchExpanding(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B, -1, C//4)
        x = self.norm(x)

        return x

class SwinDecoderLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch expanding layer
        if upsample is not None:
            self.upsample = PatchExpanding(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

class SwinUNet(nn.Module):
    def __init__(self, img_size=224, num_classes=1, use_pretrained=True, 
                 depths=[2, 2, 2], num_heads=[6, 12, 24], window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, 
                 patch_norm=True, final_upsample="expand_first"):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = 128
        self.patch_norm = patch_norm
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # Load pretrained Swin Transformer backbone
        self.backbone = swin_base_patch4_window7_224(pretrained=use_pretrained)
        
        # Remove the classifier head from backbone
        self.backbone.head = nn.Identity()
        
        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(self.embed_dim*2**(self.num_layers-1-i_layer)),
                                    int(self.embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            
            if i_layer == 0:
                layer_up = PatchExpanding(input_resolution=(img_size//(4*2**(self.num_layers-1-i_layer)), 
                                                          img_size//(4*2**(self.num_layers-1-i_layer))), 
                                        dim=int(self.embed_dim * 2 ** (self.num_layers-1-i_layer)), 
                                        dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = SwinDecoderLayer(dim=int(self.embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                          input_resolution=(img_size//(4*2**(self.num_layers-1-i_layer)), 
                                                          img_size//(4*2**(self.num_layers-1-i_layer))),
                                          depth=depths[(self.num_layers-1-i_layer)],
                                          num_heads=num_heads[(self.num_layers-1-i_layer)],
                                          window_size=window_size,
                                          mlp_ratio=self.mlp_ratio,
                                          qkv_bias=qkv_bias,
                                          drop=drop_rate, attn_drop=attn_drop_rate,
                                          drop_path=drop_path_rate,
                                          norm_layer=norm_layer,
                                          upsample=PatchExpanding if (i_layer < self.num_layers - 1) else None)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = PatchExpanding(input_resolution=(img_size//4, img_size//4), 
                                   dim=self.embed_dim, dim_scale=4, norm_layer=norm_layer)
            self.output = nn.Conv2d(in_channels=self.embed_dim//4, out_channels=self.num_classes, 
                                  kernel_size=1, bias=False)

        # Deep supervision heads (auxiliary outputs)
        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2**(i+1), mode='bilinear', align_corners=False),
                nn.Conv2d(int(self.embed_dim * 2 ** (self.num_layers-1-i)), 
                         num_classes, kernel_size=1)
            ) for i in range(self.num_layers)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # Encoder
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        x = self.backbone.patch_embed(x)
        if self.backbone.patch_embed.norm is not None:
            x = self.backbone.patch_embed.norm(x)
        
        # Extract features at different scales
        x_downsample = []
        
        for layer in self.backbone.layers:
            x_downsample.append(x)
            x = layer(x)
        
        x = self.norm(x)  # B L C
        return x, x_downsample

    def forward_up_features(self, x, x_downsample):
        aux_outputs = []
        
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3-inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
                
                # Auxiliary output for deep supervision
                B, L, C = x.shape
                H = W = int(L**0.5)
                aux_feat = x.view(B, H, W, C).permute(0, 3, 1, 2)
                aux_outputs.append(self.aux_heads[inx](aux_feat))

        x = self.norm_up(x)  # B L C
        return x, aux_outputs

    def up_x4(self, x):
        H, W = self.backbone.patch_embed.grid_size
        B, L, C = x.shape
        assert L == H*W, "input features has wrong size"

        if self.final_upsample=="expand_first":
            x = self.up(x)
            x = x.view(B, 4*H, 4*W, -1)
            x = x.permute(0, 3, 1, 2) #B,C,H,W
            x = self.output(x)
            
        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x, aux_outputs = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)

        # Return main output and auxiliary outputs for deep supervision
        if self.training:
            return x, aux_outputs
        else:
            return x