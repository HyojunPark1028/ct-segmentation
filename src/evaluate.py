import torch, pandas as pd, matplotlib.pyplot as plt

def _metric(pred, tgt, thr):
    p=(pred>thr).float(); inter=(p*tgt).sum(); union=p.sum()+tgt.sum()-inter
    dice=(2*inter+1e-6)/(p.sum()+tgt.sum()+1e-6); iou=(inter+1e-6)/(union+1e-6)
    return dice.item(), iou.item()

def _vis(img, m, p):
    import numpy as np
    img=img.squeeze().cpu().numpy(); m=m.squeeze().cpu().numpy(); p=(p.squeeze().cpu().numpy()>0.3)
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
            if vis and k==0: _vis(x[0],y[0],p[0])
    return d/n,i/n
