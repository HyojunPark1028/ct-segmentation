import os, torch, pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from .models.unet import UNet
from .dataset import NpySegDataset
from .losses import get_loss
from .evaluate import evaluate

def train(cfg_path):
    cfg=OmegaConf.load(cfg_path); os.makedirs(cfg.train.save_dir, exist_ok=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset (pre‑split)
    tr_ds=NpySegDataset(os.path.join(cfg.data.root_dir,'train'), augment=True)
    vl_ds=NpySegDataset(os.path.join(cfg.data.root_dir,'val'))
    ts_ds=NpySegDataset(os.path.join(cfg.data.root_dir,'test'))
    tr_dl=DataLoader(tr_ds,batch_size=cfg.train.batch_size,shuffle=True ,num_workers=cfg.train.num_workers)
    vl_dl=DataLoader(vl_ds,batch_size=cfg.train.batch_size,shuffle=False,num_workers=cfg.train.num_workers)
    ts_dl=DataLoader(ts_ds,batch_size=cfg.train.batch_size,shuffle=False,num_workers=cfg.train.num_workers)

    model=UNet().to(device)
    opt=torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion=get_loss()

    history=[]
    for ep in range(cfg.train.epochs):
        model.train(); run=0
        loop = tqdm(tr_dl, desc=f"Epoch {ep+1}/{cfg.train.epochs}", leave=False)
        for x,y in loop:
            x,y=x.to(device),y.to(device)
            pred = model(x)
            loss=criterion(pred,y);
            opt.zero_grad(); loss.backward(); opt.step(); run+=loss.item()
            loop.set_postfix(loss=loss.item(), mean_pred=torch.sigmoid(pred).mean().item())
        vd,vi=evaluate(model,vl_dl,device,cfg.data.threshold)
        row=dict(epoch=ep+1,train_loss=run/len(tr_dl),val_dice=vd,val_iou=vi)
        history.append(row); print(row)
        print(f"[Epoch {ep+1}] loss: {row['train_loss']:.4f}, val_dice: {vd:.f4}, val_iou: {vi:.4f}")

    # save metrics
    pd.DataFrame(history).to_csv(os.path.join(cfg.train.save_dir,'metrics.csv'),index=False)

    # final test evaluation + visualize
    td,ti=evaluate(model,ts_dl,device,cfg.data.threshold,vis=cfg.eval.visualize)
    print(f"TEST ➜ Dice:{td:.4f} IoU:{ti:.4f}")