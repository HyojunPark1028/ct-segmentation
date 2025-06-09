import torch, pandas as pd, matplotlib.pyplot as plt
# .models.medsam은 일반 MedSAM 모델일 수 있으므로, 여기서는 특정 모델 클래스를 임포트하지 않습니다.
# 대신, 모델의 동작 방식을 가정하고 코드를 작성합니다.

def _metric(pred, tgt, thr):
    """
    예측 마스크와 실제 마스크를 기반으로 Dice Score와 IoU Score를 계산합니다.
    Args:
        pred (torch.Tensor): 예측 마스크 텐서 (sigmoid가 적용되지 않은 logits 또는 확률).
        tgt (torch.Tensor): 실제 마스크 텐서 (바이너리).
        thr (float): 예측 마스크를 이진화할 임계값.
    Returns:
        tuple: (dice_score, iou_score)
    """
    p = (pred > thr).float() # 예측을 임계값으로 이진화
    inter = (p * tgt).sum() # 교차 영역
    union = p.sum() + tgt.sum() - inter # 합집합 영역
    
    # 분모가 0이 되는 것을 방지하기 위해 작은 값 (1e-6) 추가
    dice = (2 * inter + 1e-6) / (p.sum() + tgt.sum() + 1e-6)
    iou = (inter + 1e-6) / (union + 1e-6)
    
    return dice.item(), iou.item()

def _vis(img, m, p, thr):
    """
    이미지, 실제 마스크, 예측 마스크를 시각화합니다.
    Args:
        img (torch.Tensor): 원본 이미지 텐서.
        m (torch.Tensor): 실제 마스크 텐서.
        p (torch.Tensor): 예측 마스크 텐서 (sigmoid가 적용되지 않은 logits 또는 확률).
        thr (float): 예측 마스크를 이진화할 임계값.
    """
    import numpy as np
    img = img.squeeze().cpu().numpy() # 채널 차원 제거 및 CPU로 이동, NumPy 변환
    m = m.squeeze().cpu().numpy()
    p = (p.squeeze().cpu().numpy() > thr) # 이진화 및 NumPy 변환
    
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, d, t in zip(ax, [img, m, p], ["Img", "GT", "Pred"]):
        a.imshow(d, cmap='gray'); a.axis('off'); a.set_title(t)
    plt.show()

def evaluate(model, loader, device, thr=0.5, vis=False):
    """
    GAN 모델의 성능 (Dice, IoU)를 평가합니다.
    Args:
        model (nn.Module): 평가할 모델 (MedSAM_GAN 인스턴스).
        loader (DataLoader): 평가에 사용할 데이터 로더.
        device (torch.device): 모델이 위치한 장치 (CPU 또는 GPU).
        thr (float): 예측 마스크 이진화 임계값.
        vis (bool): 시각화 여부.
    Returns:
        dict: 'dice_score'와 'iou_score'를 포함하는 딕셔너리.
    """
    model.eval() # 모델을 평가 모드로 설정
    d_total = 0 # 총 Dice Score
    i_total = 0 # 총 IoU Score
    n_batches = len(loader) # 배치 수

    with torch.no_grad(): # 그래디언트 계산 비활성화
        for k, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            # MedSAM_GAN 모델의 forward pass는 (masks, iou_predictions, discriminator_output_for_generated_mask) 반환
            # validate_one_epoch 및 evaluate 함수는 G의 마스크 출력만 사용
            predicted_masks, _, _ = model(x, None, real_low_res_mask=None) 
            
            # 예측 마스크에 sigmoid 적용 (모델의 출력이 logits인 경우)
            p = torch.sigmoid(predicted_masks) # ⭐ 변경: predicted_masks에 sigmoid 적용
            
            # Dice와 IoU 계산
            dice_batch, iou_batch = _metric(p, y, thr)
            d_total += dice_batch
            i_total += iou_batch
            
            # 시각화 (선택 사항)
            if vis and k == 0: # 첫 번째 배치만 시각화
                vis_n = min(10, x.shape[0]) # 최대 10개 샘플
                for j in range(vis_n):
                    _vis(x[j], y[j], p[j], thr)

    # 평균 Dice 및 IoU 계산
    avg_dice = d_total / n_batches if n_batches > 0 else 0.0
    avg_iou = i_total / n_batches if n_batches > 0 else 0.0
    
    return {'dice_score': avg_dice, 'iou_score': avg_iou}


def compute_mask_coverage(model, loader, device, thr=0.5):
    """
    마스크의 커버리지 통계를 계산합니다 (GT 픽셀, 예측 픽셀, 교차, 커버리지 비율).
    Args:
        model (nn.Module): 평가할 모델 (MedSAM_GAN 인스턴스).
        loader (DataLoader): 평가에 사용할 데이터 로더.
        device (torch.device): 모델이 위치한 장치 (CPU 또는 GPU).
        thr (float): 예측 마스크 이진화 임계값.
    Returns:
        dict: 마스크 커버리지 관련 통계 (gt_pixels, pred_pixels, intersection, coverage, overpredict).
    """
    model.eval() # 모델을 평가 모드로 설정
    gt_total = 0 # 실제 마스크 픽셀 수 총합
    pred_total = 0 # 예측 마스크 픽셀 수 총합
    inter_total = 0 # 교차 픽셀 수 총합

    with torch.no_grad(): # 그래디언트 계산 비활성화
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y = (y > 0.5).float() # 실제 마스크를 명확히 이진화

            # MedSAM_GAN 모델의 forward pass는 (masks, iou_predictions, discriminator_output_for_generated_mask) 반환
            predicted_masks, _, _ = model(x, None, real_low_res_mask=None) 
            
            pred = torch.sigmoid(predicted_masks) > thr # ⭐ 변경: predicted_masks에 sigmoid 적용 후 이진화
            
            inter = (pred * y).sum() # 교차 픽셀 계산
            gt_total += y.sum() # 실제 픽셀 수 누적
            pred_total += pred.sum() # 예측 픽셀 수 누적
            inter_total += inter # 교차 픽셀 수 누적

    # 커버리지 및 과대 예측 비율 계산
    coverage = (inter_total / gt_total).item() if gt_total > 0 else 0.0
    overpredict = (pred_total / gt_total).item() if gt_total > 0 else 0.0
    
    return {
        "gt_pixels": int(gt_total),
        "pred_pixels": int(pred_total),
        "intersection": int(inter_total),
        "coverage": coverage,
        "overpredict": overpredict
    }

