import time
import gc
import os, torch, pandas as pd
import random, numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from .models.unet import UNet
from .models.transunet import TransUNet
from .models.swinunet import SwinUNet
from .models.utransvision import UTransVision
from .models.sam2unet import SAM2UNet
from .models.medsam import MedSAM
from .models.medsam2 import MedSAM2
from .dataset import NpySegDataset
from .losses import get_loss
from .evaluate import evaluate, compute_mask_coverage

print("hello4")

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
    torch.use_deterministic_algorithms(False)

def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(cfg_path):
    start_time = time.time()
    cfg = OmegaConf.load(cfg_path)
    seed_everything(cfg.experiment.seed)
    os.makedirs(cfg.train.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    OmegaConf.save(cfg, os.path.join(cfg.train.save_dir, "used_config.yaml"))

    g = torch.Generator()
    g.manual_seed(cfg.experiment.seed)

    model_name = cfg.model.name.lower()
    img_size = cfg.model.img_size if model_name in ["transunet", "swinunet", "medsam", "medsam2", "sam2unet"] else None
    use_pretrained = cfg.model.get("use_pretrained", False)
    normalize_type = "sam" if model_name in ["medsam", "medsam2", "sam2unet"] else "default"

    tr_ds = NpySegDataset(os.path.join(cfg.data.root_dir, 'train'), augment=True, img_size=img_size, normalize_type=normalize_type)
    vl_ds = NpySegDataset(os.path.join(cfg.data.root_dir, 'val'), img_size=img_size, normalize_type=normalize_type)
    ts_ds = NpySegDataset(os.path.join(cfg.data.root_dir, 'test'), img_size=img_size, normalize_type=normalize_type)
    tr_dl = DataLoader(tr_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    vl_dl = DataLoader(vl_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    ts_dl = DataLoader(ts_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    print(f"[DEBUG] config img_size: {img_size}")
    for x_dbg, y_dbg in tr_dl:
        print(f"[DEBUG] one batch input shape: {x_dbg.shape}")
        break

    if model_name == "unet":
        model = UNet(use_pretrained=use_pretrained).to(device)
    elif model_name == "transunet":
        model = TransUNet(img_size=cfg.model.img_size, num_classes=1, use_pretrained=use_pretrained).to(device)
    elif model_name == "swinunet":
        model = SwinUNet(img_size=cfg.model.img_size, num_classes=1, use_pretrained=use_pretrained).to(device)
    elif model_name == "utransvision":
        model = UTransVision(img_size=cfg.model.img_size, num_classes=1, use_pretrained=use_pretrained).to(device)
    elif model_name == "sam2unet_old":
        model = SAM2UNet(checkpoint=cfg.model.checkpoint).to(device)
    elif model_name == "sam2unet":
        model = SAM2UNet(checkpoint=cfg.model.checkpoint, config=cfg.model.config).to(device)
    elif model_name == "medsam":
        model = MedSAM(checkpoint=cfg.model.checkpoint).to(device)
    elif model_name == "medsam2":
        model = MedSAM2(checkpoint=cfg.model.checkpoint, image_size=cfg.model.img_size).to(device)
    else:
        raise ValueError(f"Unsupported model name: {cfg.model.name}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = get_loss()

    history = []
    best_dice = 0
    patience = cfg.train.get("patience", 10)
    counter = 0

    for ep in range(cfg.train.epochs):
        model.train(); run = 0
        loop = tqdm(tr_dl, desc=f"Epoch {ep+1}/{cfg.train.epochs}", leave=False)
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            if isinstance(preds, tuple):
                pred_main, *pred_deep = preds
                loss = 0.7 * criterion(pred_main, y)
                for i, p in enumerate(pred_deep):
                    try:
                        print(f"[DEBUG] pred_deep[{i}] shape: {p.shape}")
                        h, w = y.shape[2], y.shape[3]

                        # 1️⃣ 채널 수 1로 줄이기
                        p = nn.Conv2d(p.shape[1], 1, kernel_size=1).to(p.device)(p)

                        # 2️⃣ 업샘플링
                        p_up = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)

                        print(f"[DEBUG] p_up shape: {p_up.shape}")
                        loss += 0.1 * criterion(p_up, y)
                    except Exception as e:
                        print(f"[ERROR] in deep supervision loss for p[{i}]: {e}")
                        raise
            else:
                loss = criterion(preds, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step(); run += loss.item()
            pred_cpu = (preds[0] if isinstance(preds, tuple) else preds).detach().cpu()
            loop.set_postfix(loss=loss.item(), mean_pred=torch.sigmoid(pred_cpu).mean().item())
            del preds, loss
            torch.cuda.empty_cache(); gc.collect()

        model.eval(); vloss = 0
        with torch.no_grad():
            for x_val, y_val in vl_dl:
                x_val, y_val = x_val.to(device), y_val.to(device)
                v_pred = model(x_val)
                if isinstance(v_pred, tuple): v_pred = v_pred[0]
                v_loss = criterion(v_pred, y_val)
                vloss += v_loss.item()

        vd, vi = evaluate(model, vl_dl, device, cfg.data.threshold)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = dict(epoch=ep+1, train_loss=run/len(tr_dl), val_loss=vloss/len(vl_dl), val_dice=vd, val_iou=vi, timestamp=timestamp)
        history.append(row); print(row)

        if vd > best_dice:
            best_dice = vd
            torch.save(model.state_dict(), os.path.join(cfg.train.save_dir, "model_best.pth"))
            counter = 0
        else:
            counter += 1
            print(f"No improvement in val_dice. EarlyStopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    pd.DataFrame(history).to_csv(os.path.join(cfg.train.save_dir, 'metrics.csv'), index=False)

    model.load_state_dict(torch.load(os.path.join(cfg.train.save_dir, "model_best.pth"), weights_only=False))
    model.eval(); test_loss = 0
    test_start = time.time()
    with torch.no_grad():
        for x_test, y_test in ts_dl:
            x_test, y_test = x_test.to(device), y_test.to(device)
            pred = model(x_test)
            if isinstance(pred, tuple): pred = pred[0]
            loss = criterion(pred, y_test)
            test_loss += loss.item()
    test_end = time.time(); test_elapsed = test_end - test_start

    td, ti = evaluate(model, ts_dl, device, cfg.data.threshold, vis=cfg.eval.visualize)
    print(f"TEST ➜ Dice:{td:.4f} IoU:{ti:.4f} Test Loss:{test_loss/len(ts_dl):.4f} Time:{test_elapsed:.2f}s")

    coverage_stats = compute_mask_coverage(model, ts_dl, device, cfg.data.threshold)
    for k, v in coverage_stats.items():
        print(f"{k}: {v}")

    pd.DataFrame([coverage_stats]).to_csv(os.path.join(cfg.train.save_dir, "test_coverage.csv"), index=False)

    elapsed = time.time() - start_time
    print(f"\n Total training time: {elapsed/60:.2f} minutes")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    test_result = {
        "test_dice": td,
        "test_iou": ti,
        "test_loss": round(test_loss / len(ts_dl), 4),
        "test_inference_time_sec": round(test_elapsed, 2),
        "elapsed_minutes": round(elapsed / 60, 2),
        "param_count": param_count
    }
    pd.DataFrame([test_result]).to_csv(os.path.join(cfg.train.save_dir, "test_result.csv"), index=False)
