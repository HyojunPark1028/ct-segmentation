import torch, torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__(); self.smooth=smooth
    def forward(self, p, t):
        p=torch.sigmoid(p).view(-1); t=t.view(-1)
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

# def get_loss():
    # return DiceFocalLoss()

def get_loss(pos_weight=2.0)    :
    class DiceBCE(nn.Module):
        def __init__(self, w_dice=0.4, w_bce=0.6):
            super().__init__()
            self.dice = DiceLoss()
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
            self.wd, self.wb = w_dice, w_bce
        def forward(self, p, t):
            return self.wd*self.dice(p,t) + self.wb*self.bce(p,t)
    return DiceBCE()