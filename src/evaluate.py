import torch, pandas as pd, matplotlib.pyplot as plt

def _metric(pred, tgt, thr):
    p=(pred>thr).float(); inter=(p*tgt).sum(); union=p.sum()+tgt.sum()-inter
    dice=(2*inter+1e-6)/(p.sum()+tgt.sum()+1e-6); iou=(inter+1e-6)/(union+1e-6)
    return dice.item(), iou.item()

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
            x,y=x.to(device),y.to(device); p=torch.sigmoid(model(x))
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
            pred = torch.sigmoid(model(x)) > thr
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
