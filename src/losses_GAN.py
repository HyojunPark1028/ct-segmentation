import torch, torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__(); self.smooth=smooth
    def forward(self, p, t):
        p=torch.sigmoid(p).reshape(-1); t=t.reshape(-1)
        inter=(p*t).sum(); return 1- (2*inter+self.smooth)/(p.sum()+t.sum()+self.smooth)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__(); self.a,self.g=alpha,gamma; self.bce=nn.BCEWithLogitsLoss(reduction='none')
    def forward(self,p,t):
        b=self.bce(p,t); pt=torch.exp(-b); return (self.a*(1-pt)**self.g*b).mean()

class DiceFocalLoss(nn.Module):
    def __init__(self, w_dice=0.5, w_focal=0.5):
        super().__init__(); self.d=DiceLoss(); self.f=FocalLoss(); self.wd, self.wf = w_dice, w_focal
    def forward(self,p,t):
        return self.wd*self.d(p,t)+self.wf*self.f(p,t)

# --- GAN Loss 추가 ---
class DiscriminatorLoss(nn.Module):
    """
    Discriminator의 손실 함수. 일반적으로 BCEWithLogitsLoss를 사용.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_real, pred_fake):
        # Discriminator는 진짜(real)는 1로, 가짜(fake)는 0으로 예측하도록 학습
        loss_real = self.bce(pred_real, torch.ones_like(pred_real))
        loss_fake = self.bce(pred_fake, torch.zeros_like(pred_fake))
        return loss_real + loss_fake

class GeneratorAdversarialLoss(nn.Module):
    """
    Generator의 Adversarial Loss.
    Generator는 Discriminator를 속여 생성된 이미지가 진짜로 판별되도록 학습.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_fake):
        # Generator는 가짜(fake)가 진짜(real)처럼 보이도록 학습
        return self.bce(pred_fake, torch.ones_like(pred_fake))

def get_segmentation_loss():
    return DiceFocalLoss()

def get_discriminator_loss():
    return DiscriminatorLoss()

def get_generator_adversarial_loss():
    return GeneratorAdversarialLoss()