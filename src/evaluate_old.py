import torch, pandas as pd, matplotlib.pyplot as plt
# from .models.medsam import MedSAM  # MedSAM 분기 처리를 위해 import

def _metric(pred, tgt, thr):
    # Ensure pred and tgt are binary (0 or 1)
    p = (pred > thr).float() # Predicted binary mask
    tgt = (tgt > 0.5).float() # Ground truth binary mask (explicitly binarize again to be safe)

    inter = (p * tgt).sum()
    union = p.sum() + tgt.sum() - inter

    # Handle cases where both masks are empty
    if p.sum() == 0 and tgt.sum() == 0:
        dice_val = 1.0
        iou_val = 1.0
    else:
        # Avoid division by zero for dice
        dice_denominator = p.sum() + tgt.sum()
        # If dice_denominator is 0 here (shouldn't be if not both empty), add epsilon
        dice_tensor = (2 * inter + 1e-6) / (dice_denominator + 1e-6)

        # Avoid division by zero for iou
        iou_denominator = union
        # If iou_denominator is 0 here (shouldn't be if not both empty), add epsilon
        iou_tensor = (inter + 1e-6) / (iou_denominator + 1e-6)

        # Ensure dice and iou are within [0, 1] range in case of numerical instability
        dice_val = torch.clamp(dice_tensor, 0.0, 1.0).item()
        iou_val = torch.clamp(iou_tensor, 0.0, 1.0).item()
    
    return dice_val, iou_val

def _vis(img, m, p, thr):
    import numpy as np
    img=img.squeeze().cpu().numpy(); m=m.squeeze().cpu().numpy(); p=(p.squeeze().cpu().numpy()>thr)
    fig,ax=plt.subplots(1,3,figsize=(12,4))
    for a,d,t in zip(ax,[img,m,p],["Img","GT","Pred"]):
        a.imshow(d,cmap='gray'); a.axis('off'); a.set_title(t)
    plt.show()

def evaluate(model, loader, device, thr=0.5, vis=False):
    model.eval(); d=i=0; n=len(loader)
    with torch.no_grad():
        for k,(x,y) in enumerate(loader):
            x,y=x.to(device),y.to(device); 
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]
            p = torch.sigmoid(output)
            di,io=_metric(p,y,thr); d+=di; i+=io
            if vis and k==0: 
                vis_n = min(10, x.shape[0])  # 안전하게 제한
                for j in range(vis_n):
                    _vis(x[j],y[j],p[j], thr)
                    plt.pause(0.1)
    return d/n,i/n

def compute_mask_coverage(model, loader, device, thr=0.5):
    model.eval()
    gt_total = 0
    pred_total = 0
    inter_total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y = (y > 0.5).float()  # ⭐ 강제 이진화
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]  # SAM2UNet 등 대응

            pred = torch.sigmoid(output) > thr            
            inter = (pred * y).sum()
            gt_total += y.sum()
            pred_total += pred.sum()
            inter_total += inter
    coverage = (inter_total / gt_total).item() if gt_total > 0 else 0.0
    overpredict = (pred_total / gt_total).item() if gt_total > 0 else 0.0
    return {
        "gt_pixels": int(gt_total),
        "pred_pixels": int(pred_total),
        "intersection": int(inter_total),
        "coverage": coverage,
        "overpredict": overpredict
    }