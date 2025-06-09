import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Segmentation Loss (기존 유지) ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__(); self.smooth=smooth
    def forward(self, p, t):
        p=torch.sigmoid(p).reshape(-1); t=t.reshape(-1)
        inter=(p*t).sum(); return 1- (2*inter+self.smooth)/(p.sum()+t.sum()+self.smooth)

class FocalLoss(nn.Module):
    # ⭐ 변경 사항: alpha, gamma를 __init__에서 받도록 수정
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__(); self.a,self.g=alpha,gamma; self.bce=nn.BCEWithLogitsLoss(reduction='none')
    def forward(self,p,t):
        b=self.bce(p,t); pt=torch.exp(-b); return (self.a*(1-pt)**self.g*b).mean()

class DiceFocalLoss(nn.Module):
    # ⭐ 변경 사항: w_dice, w_focal, 그리고 FocalLoss에 전달할 alpha_focal, gamma_focal을 받도록 수정
    def __init__(self, w_dice=0.5, w_focal=0.5, alpha_focal=0.8, gamma_focal=2.0):
        super().__init__(); 
        self.d=DiceLoss(); 
        # ⭐ 변경 사항: FocalLoss 인스턴스화 시 파라미터 전달
        self.f=FocalLoss(alpha=alpha_focal, gamma=gamma_focal); 
        self.wd, self.wf = w_dice, w_focal
    def forward(self,p,t):
        return self.wd*self.d(p,t)+self.wf*self.f(p,t)

# ⭐ 변경 사항: get_segmentation_loss 함수가 파라미터를 받도록 수정
def get_segmentation_loss(w_dice=0.5, w_focal=0.5, alpha_focal=0.8, gamma_focal=2.0):
    return DiceFocalLoss(w_dice=w_dice, w_focal=w_focal, alpha_focal=alpha_focal, gamma_focal=gamma_focal)

# --- WGAN-GP Loss (기존 유지) ---
class DiscriminatorLossWGAN_GP(nn.Module):
    """
    WGAN-GP Discriminator 손실 함수.
    Args:
        lambda_gp (float): Gradient Penalty의 가중치.
    """
    def __init__(self, lambda_gp=10.0):
        super().__init__()
        self.lambda_gp = lambda_gp

    def forward(self, pred_real, pred_fake, real_samples, fake_samples, discriminator_model):
        loss_real = -torch.mean(pred_real)
        loss_fake = torch.mean(pred_fake)
        wasserstein_loss = loss_real + loss_fake

        alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        
        d_interpolates = discriminator_model(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp

        total_d_loss = wasserstein_loss + gradient_penalty
        return total_d_loss

class GeneratorAdversarialLossWGAN_GP(nn.Module):
    """
    WGAN-GP Generator 손실 함수.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_fake):
        return -torch.mean(pred_fake)

def get_discriminator_loss_wgan_gp(lambda_gp=10.0):
    return DiscriminatorLossWGAN_GP(lambda_gp)

def get_generator_adversarial_loss_wgan_gp():
    return GeneratorAdversarialLossWGAN_GP()

# 편의를 위해 기존 함수명으로 WGAN-GP 버전을 호출하도록 매핑 (호환성 유지)
get_discriminator_loss = get_discriminator_loss_wgan_gp
get_generator_adversarial_loss = get_generator_adversarial_loss_wgan_gp

