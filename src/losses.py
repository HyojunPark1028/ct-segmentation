import torch, torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__(); self.smooth=smooth
    def forward(self, p, t):
        p=torch.sigmoid(p).reshape(-1); t=t.reshape(-1)
        inter=(p*t).sum(); return 1- (2*inter+self.smooth)/(p.sum()+t.sum()+self.smooth)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.):
        super().__init__(); self.a,self.g=alpha,gamma; self.bce=nn.BCEWithLogitsLoss(reduction='none')
    def forward(self,p,t):
        b=self.bce(p,t); pt=torch.exp(-b); return (self.a*(1-pt)**self.g*b).mean()

class DiceFocalLoss(nn.Module):
    def __init__(self, w_dice=0.5, w_focal=0.5):
        super().__init__(); self.d=DiceLoss(); self.f=FocalLoss(); self.wd, self.wf = w_dice, w_focal
    def forward(self,p,t):
        return self.wd*self.d(p,t)+self.wf*self.f(p,t)

def get_loss():
    return DiceFocalLoss()


# class FocalTverskyLoss(nn.Module):
#     def __init__(self, alpha=0.7, beta=0.3, gamma=4/3, smooth=1.):
#         super().__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.smooth = smooth
    
#     def forward(self, p, t):
#         p = torch.sigmoid(p)
#         p, t = p.view(-1), t.view(-1)
#         TP = (p * t).sum()
#         FP = ((1-t) * p).sum()
#         FN = (t * (1-p)).sum()
#         tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
#         return (1 - tversky) ** self.gamma
    
# def get_loss():
#     return FocalTverskyLoss()
