import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """Exact UNet copied from original notebook"""
    def __init__(self, in_channels=1, out_channels=1, features=(64,128,256,512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        # encoder
        for f in features:
            self.downs.append(self.double_conv(in_channels, f))
            in_channels = f
        # bottleneck
        self.bottleneck = self.double_conv(features[-1], features[-1]*2)
        # decoder
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))
            self.ups.append(self.double_conv(f*2, f))
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
    def forward(self,x):
        skips=[]
        for down in self.downs:
            x=down(x); skips.append(x); x=F.max_pool2d(x,2)
        x=self.bottleneck(x); skips=skips[::-1]
        for idx in range(0,len(self.ups),2):
            x=self.ups[idx](x)
            skip=skips[idx//2]
            if x.shape!=skip.shape:
                x=F.interpolate(x,size=skip.shape[2:])
            x=torch.cat((skip,x),dim=1)
            x=self.ups[idx+1](x)
        return self.final_conv(x)
    @staticmethod
    def double_conv(in_c,out_c):
        return nn.Sequential(
            nn.Conv2d(in_c,out_c,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_c,out_c,3,padding=1), nn.ReLU(inplace=True))