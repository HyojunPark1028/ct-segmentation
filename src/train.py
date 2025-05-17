import time
import gc
import os, torch, pandas as pd
import random, numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from .models.unet import UNet
from .models.transunet import TransUNet
from .models.swinunet import SwinUNet
from .models.utransvision import UTransVision
from .models.sam2unet import SAM2UNet
from .models.medsam import MedSAM
from .dataset import NpySegDataset
from .losses import get_loss
from .evaluate import evaluate, compute_mask_coverage

def seed_everything(seed=42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

# Seed-safe DataLoader 세팅
def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(cfg_path):
    start_time=time.time()
    cfg=OmegaConf.load(cfg_path); seed_everything(cfg.experiment.seed); os.makedirs(cfg.train.save_dir, exist_ok=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'

    # Save config snapshot
    OmegaConf.save(cfg, os.path.join(cfg.train.save_dir, "used_config.yaml"))

    g = torch.Generator()
    g.manual_seed(cfg.experiment.seed)

    # Dataset (pre‑split)
    # ✅ 모델에 따라 img_size 결정
    model_name = cfg.model.name.lower()
    img_size = cfg.model.img_size if model_name in ["transunet", "swinunet", "medsam"] else None
    use_pretrained = cfg.model.get("use_pretrained", False)
    normalize_type = "sam" if cfg.model.name.lower() in ["medsam", "sam2unet"] else "default"

    # Dataset (pre‑split)
    tr_ds = NpySegDataset(os.path.join(cfg.data.root_dir, 'train'), augment=True, img_size=img_size, normalize_type=normalize_type)
    vl_ds = NpySegDataset(os.path.join(cfg.data.root_dir, 'val'), img_size=img_size, normalize_type=normalize_type)
    ts_ds = NpySegDataset(os.path.join(cfg.data.root_dir, 'test'), img_size=img_size, normalize_type=normalize_type)
    tr_dl=DataLoader(tr_ds,batch_size=cfg.train.batch_size,shuffle=True ,num_workers=cfg.train.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    vl_dl=DataLoader(vl_ds,batch_size=cfg.train.batch_size,shuffle=False,num_workers=cfg.train.num_workers)
    ts_dl=DataLoader(ts_ds,batch_size=cfg.train.batch_size,shuffle=False,num_workers=cfg.train.num_workers)

    # Model Selection
    if cfg.model.name.lower() == "unet":
        model=UNet(use_pretrained=use_pretrained).to(device)
    elif cfg.model.name.lower() == "transunet":
        model = TransUNet(img_size=cfg.model.img_size, num_classes=1, use_pretrained=use_pretrained).to(device)
    elif cfg.model.name.lower() == "swinunet":
        model = SwinUNet(img_size=cfg.model.img_size, num_classes=1, use_pretrained=use_pretrained).to(device)
    elif cfg.model.name.lower() == 'utransvision':
        model = UTransVision(img_size=cfg.model.img_size, num_classes=1, use_pretrained=use_pretrained).to(device)        
    elif cfg.model.name.lower() == "sam2unet":
        model = SAM2UNet(checkpoint=cfg.model.checkpoint).to(device)
    elif cfg.model.name.lower() == "medsam":
        model = MedSAM(checkpoint=cfg.model.checkpoint).to(device)
    else:
        raise ValueError(f"Unsupported model name: {cfg.model.name}")

    opt=torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion=get_loss()

    history=[]
    best_dice = 0
    patience = cfg.train.get("patience", 10)
    counter = 0 # early stopping trigger
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
        torch.cuda.empty_cache()
        gc.collect()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row=dict(epoch=ep+1,train_loss=run/len(tr_dl),val_dice=vd,val_iou=vi,timestamp=timestamp)
        history.append(row); print(row)
        print(f"[Epoch {ep+1}] loss: {row['train_loss']:.4f}, val_dice: {vd:.4f}, val_iou: {vi:.4f}")
        print(f"[SIGMOID DEBUG] pred mean: {pred.mean().item():.4f}, max: {pred.max().item():.4f}, min: {pred.min().item():.4f}")

        # Save best model
        if vd > best_dice:
            best_dice = vd
            torch.save(model.state_dict(), os.path.join(cfg.train.save_dir, "model_best.pth"))
            counter = 0 # reset patience counter
        else:
            counter += 1
            print(f"No improvement in val_dice. EarlyStopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # save metrics
    pd.DataFrame(history).to_csv(os.path.join(cfg.train.save_dir,'metrics.csv'),index=False)

    # final test evaluation using best model + visualize
    model.load_state_dict(torch.load(os.path.join(cfg.train.save_dir, "model_best.pth")))
    td,ti=evaluate(model,ts_dl,device,cfg.data.threshold,vis=cfg.eval.visualize)
    print(f"TEST ➜ Dice:{td:.4f} IoU:{ti:.4f}")

    coverage_stats = compute_mask_coverage(model, ts_dl, device, cfg.data.threshold)
    print("\n Mask Prediction Coverage:")
    for k, v in coverage_stats.items():
        print(f"{k}: {v}")

    # save test coverage
    pd.DataFrame([coverage_stats]).to_csv(os.path.join(cfg.train.save_dir, "test_coverage.csv"), index=False)

    # print total time
    elapsed = time.time() - start_time
    print(f"\n Total training time: {elapsed/60:.2f} minutes")

    # Save test result
    test_result = {"test_dice":td,"test_iou":ti,"elapsed_minutes": round(elapsed / 60, 2)}
    pd.DataFrame([test_result]).to_csv(os.path.join(cfg.train.save_dir, "test_result.csv"), index=False)
