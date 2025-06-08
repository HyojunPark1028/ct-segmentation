# src/train_GAN.py

import time
import gc
import os, torch, pandas as pd
import random, numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import KFold # KFold 임포트 확인
import shutil
import glob

# 모델 임포트
from .models.medsam_gan import MedSAM_GAN

from .dataset import NpySegDataset
import cv2

# 기존 import를 유지하되 GAN Loss 함수 추가
from .losses_GAN import get_segmentation_loss, get_discriminator_loss, get_generator_adversarial_loss
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

def train_one_epoch(
    model,
    dataloader,
    optimizer_G,
    optimizer_D,
    seg_criterion,
    adv_criterion_D,
    adv_criterion_G,
    device,
    epoch,
    log_interval,
    gan_lambda_adv=0.1
):
    model.train()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    
    total_seg_loss = 0
    total_g_adv_loss = 0
    total_g_loss = 0
    total_d_loss = 0

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        real_low_res_masks = F.interpolate(
            masks, size=(256, 256), mode='nearest'
        ).float()


        # --- 1. Discriminator (D) 학습 ---
        optimizer_D.zero_grad()

        with torch.no_grad():
            gen_masks, _, disc_output_gen_for_G_no_grad = model(images, None)
        
        _, _, disc_output_gen_for_D, disc_output_real = model(images, real_low_res_masks)

        d_loss = adv_criterion_D(disc_output_real, disc_output_gen_for_D)
        d_loss.backward()
        optimizer_D.step()

        total_d_loss += d_loss.item()


        # --- 2. Generator (G) 학습 ---
        optimizer_G.zero_grad()

        gen_masks, iou_predictions, disc_output_gen_for_G = model(images, None)

        seg_loss = seg_criterion(gen_masks, masks)
        total_seg_loss += seg_loss.item()

        g_adv_loss = adv_criterion_G(disc_output_gen_for_G)
        total_g_adv_loss += g_adv_loss.item()

        g_loss = seg_loss + gan_lambda_adv * g_adv_loss
        g_loss.backward()
        optimizer_G.step()
        
        total_g_loss += g_loss.item()

        pbar.set_postfix({
            "D_Loss": f"{d_loss.item():.4f}",
            "G_Seg_Loss": f"{seg_loss.item():.4f}",
            "G_Adv_Loss": f"{g_adv_loss.item():.4f}",
            "G_Total_Loss": f"{g_loss.item():.4f}"
        })

    avg_seg_loss = total_seg_loss / len(dataloader)
    avg_g_adv_loss = total_g_adv_loss / len(dataloader)
    avg_g_loss = total_g_loss / len(dataloader)
    avg_d_loss = total_d_loss / len(dataloader)

    return avg_g_loss, avg_d_loss, avg_seg_loss, avg_g_adv_loss

def validate_one_epoch(model, dataloader, seg_criterion, device):
    model.eval()
    pbar = tqdm(dataloader, desc="Validation")
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)

            predicted_masks, _, _ = model(images, None)
            
            loss = seg_criterion(predicted_masks, masks)
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def run_training_pipeline(cfg):
    seed_everything(cfg.seed)
    
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() and cfg.gpu is not None else "cpu")
    print(f"Using device: {device}")

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.output_dir, f"{cfg.model.name}_{current_time}")
    os.makedirs(output_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))

    if not os.path.exists(cfg.model.unet_checkpoint):
        raise FileNotFoundError(f"U-Net checkpoint not found at: {cfg.model.unet_checkpoint}")
    
    # ⭐⭐ 이 부분이 가장 중요하게 수정됩니다. ⭐⭐
    # K-Fold를 위해 모든 데이터셋 (train + val)의 경로를 먼저 찾습니다.
    # cfg.data.data_dir 아래에 train, val, test 폴더가 있다는 것을 가정합니다.
    base_data_dir = cfg.data.data_dir # /content/drive/MyDrive/Paper/ct_segmentation/output_dataset

    all_image_paths = []
    all_mask_paths = []

    # 'train' 폴더의 데이터 로드
    train_images = sorted(glob.glob(os.path.join(base_data_dir, "train", "images", "*.npy")))
    train_masks = sorted(glob.glob(os.path.join(base_data_dir, "train", "masks", "*.png")))
    all_image_paths.extend(train_images)
    all_mask_paths.extend(train_masks)

    # 'val' 폴더의 데이터 로드 (K-Fold 시 train/val split을 위해 합쳐야 함)
    val_images = sorted(glob.glob(os.path.join(base_data_dir, "val", "images", "*.npy")))
    val_masks = sorted(glob.glob(os.path.join(base_data_dir, "val", "masks", "*.png")))
    all_image_paths.extend(val_images)
    all_mask_paths.extend(val_masks)

    # 디버깅 출력 추가
    print(f"DEBUG: Base data directory: {base_data_dir}")
    print(f"DEBUG: Found {len(train_images)} train images and {len(train_masks)} train masks.")
    print(f"DEBUG: Found {len(val_images)} val images and {len(val_masks)} val masks.")
    print(f"DEBUG: Total images for K-Fold: {len(all_image_paths)}")
    print(f"DEBUG: Total masks for K-Fold: {len(all_mask_paths)}")
    if len(all_image_paths) > 0:
        print(f"DEBUG: First combined image path: {all_image_paths[0]}")
    if len(all_mask_paths) > 0:
        print(f"DEBUG: First combined mask path: {all_mask_paths[0]}")


    if len(all_image_paths) == 0 or len(all_image_paths) != len(all_mask_paths):
        raise ValueError(f"No image/mask files found or mismatch in combined train/val data from {base_data_dir}. Check paths and file types (.npy for images, .png for masks) within train/val subdirectories.")

    # Dataset 초기화 (KFold를 위해 전체 데이터셋을 한 번에 로드)
    dataset = NpySegDataset(
        image_paths=all_image_paths, # ⭐ 수정: 전체 이미지 경로 리스트 전달
        mask_paths=all_mask_paths,   # ⭐ 수정: 전체 마스크 경로 리스트 전달
        augment=cfg.data.augment,
        img_size=cfg.data.image_size,
        normalize_type=cfg.data.normalize_type
    )

    all_indices = list(range(len(dataset)))
    
    # ⭐ KFold 객체 생성 (이전 답변에서 누락되었던 부분)
    kf = KFold(n_splits=cfg.kfold.n_splits, shuffle=True, random_state=cfg.seed)
    
    fold_results = []
    
    start_time = time.time()

    for fold, (train_indices, val_indices) in enumerate(kf.split(all_indices)):
        print(f"\n--- Fold {fold+1}/{cfg.kfold.n_splits} ---")
        
        fold_train_subset = torch.utils.data.Subset(dataset, train_indices)
        fold_val_subset = torch.utils.data.Subset(dataset, val_indices)
        
        train_loader = DataLoader(fold_train_subset, batch_size=cfg.dataloader.batch_size, shuffle=True, num_workers=cfg.dataloader.num_workers, pin_memory=True)
        val_loader = DataLoader(fold_val_subset, batch_size=cfg.dataloader.batch_size, shuffle=False, num_workers=cfg.dataloader.num_workers, pin_memory=True)

        fold_output_dir = os.path.join(output_dir, f"fold_{fold+1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0

        # 각 폴드마다 모델 상태를 새로 초기화 (옵티마이저도 새로 생성)
        model = MedSAM_GAN(
            sam_checkpoint=cfg.model.sam_checkpoint,
            unet_checkpoint=cfg.model.unet_checkpoint,
            out_channels=cfg.model.out_channels
        ).to(device)

        # Initialize loss criteria ( moved here to be initialized per fold )
        seg_criterion = get_segmentation_loss().to(device)
        adv_criterion_D = get_discriminator_loss().to(device)
        adv_criterion_G = get_generator_adversarial_loss().to(device)

        # 옵티마이저도 각 폴드마다 새로 초기화
        optimizer_G = torch.optim.AdamW(
            params=[
                {'params': model.sam.image_encoder.parameters(), 'lr': cfg.optimizer.g_lr},
                {'params': model.sam.prompt_encoder.parameters(), 'lr': cfg.optimizer.g_lr},
                {'params': model.sam.mask_decoder.parameters(), 'lr': cfg.optimizer.g_lr}
            ],
            lr=cfg.optimizer.g_lr,
            weight_decay=cfg.optimizer.weight_decay
        )
        optimizer_D = torch.optim.AdamW(
            model.discriminator.parameters(),
            lr=cfg.optimizer.d_lr,
            weight_decay=cfg.optimizer.weight_decay
        )


        for epoch in range(cfg.epochs):
            train_g_loss, train_d_loss, train_seg_loss, train_g_adv_loss = train_one_epoch(
                model, train_loader, optimizer_G, optimizer_D,
                seg_criterion, adv_criterion_D, adv_criterion_G,
                device, epoch + 1, cfg.log_interval, cfg.gan_lambda_adv
            )
            
            val_loss = validate_one_epoch(model, val_loader, seg_criterion, device)
            
            print(f"Epoch {epoch+1} / {cfg.epochs}:")
            print(f"Train G Loss: {train_g_loss:.4f} (Seg: {train_seg_loss:.4f}, Adv: {train_g_adv_loss:.4f}), Train D Loss: {train_d_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")

            # Early Stopping 및 Best Model 저장 로직
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_G_state_dict': model.sam.state_dict(),
                    'model_D_state_dict': model.discriminator.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                    'best_val_loss': best_val_loss,
                    'cfg': cfg
                }, os.path.join(fold_output_dir, "best_model.pth"))
                print(f"Best model saved for fold {fold+1} at epoch {epoch+1} with Val Loss: {best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch+1} for fold {fold+1}.")
                    break
        
        # 각 폴드의 최종 성능 평가
        checkpoint = torch.load(os.path.join(fold_output_dir, "best_model.pth"), map_location=device)
        model.sam.load_state_dict(checkpoint['model_G_state_dict'])
        model.discriminator.load_state_dict(checkpoint['model_D_state_dict'])
        
        model.eval()
        
        metrics = evaluate(model, val_loader, device)
        fold_results.append(metrics)
        print(f"Fold {fold+1} Test Metrics (on validation set): {metrics}")
        
        del train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()


    # 최종 결과 집계 및 저장 (K-Fold 완료 후)
    final_metrics_df = pd.DataFrame(fold_results)
    final_metrics_df.loc['mean'] = final_metrics_df.mean()
    final_metrics_df.loc['std'] = final_metrics_df.std()
    final_metrics_df.to_csv(os.path.join(output_dir, "kfold_results.csv"))
    print(f"\nK-Fold Cross Validation Results saved to: {os.path.join(output_dir, 'kfold_results.csv')}")

    # 최종 테스트 (cfg.test_img_dir가 설정되어 있다면)
    # ⭐ 테스트셋 로딩도 구조에 맞게 수정
    if cfg.data.test_img_dir and os.path.exists(cfg.data.test_img_dir):
        print("\n--- Running final test on independent test set ---")
        test_base_dir = cfg.data.test_img_dir # 이것도 configs/medsam_gan.yaml에서 실제 test 폴더를 지정
        
        test_image_paths = sorted(glob.glob(os.path.join(test_base_dir, "images", "*.npy")))
        test_mask_paths = sorted(glob.glob(os.path.join(test_base_dir, "masks", "*.png")))

        if len(test_image_paths) == 0 or len(test_image_paths) != len(test_mask_paths):
            print(f"Warning: No image/mask files found or mismatch in test_img_dir {test_base_dir}. Skipping final test evaluation.")
        else:
            test_dataset = NpySegDataset(
                image_paths=test_image_paths,
                mask_paths=test_mask_paths,
                augment=False,
                img_size=cfg.data.image_size,
                normalize_type=cfg.data.normalize_type
            )
            test_loader = DataLoader(test_dataset, batch_size=cfg.dataloader.batch_size, shuffle=False, num_workers=cfg.dataloader.num_workers, pin_memory=True)
            
            # 여기서 사용할 모델은 K-Fold 후의 최종 모델이거나, 별도로 로드한 모델이어야 함
            # 현재 코드에서는 마지막 폴드의 모델이 그대로 사용됨
            test_metrics = evaluate(model, test_loader, device)
            coverage_stats = compute_mask_coverage(model, test_loader, device)
            
            print(f"Final Test Metrics: {test_metrics}")
            test_result = {
                **test_metrics,
                "total_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "gt_total": coverage_stats['gt_pixels'].item() if isinstance(coverage_stats['gt_pixels'], torch.Tensor) else coverage_stats['gt_pixels'],
                "pred_total": coverage_stats['pred_pixels'].item() if isinstance(coverage_stats['pred_pixels'], torch.Tensor) else coverage_stats['pred_pixels'],
                "inter_total": coverage_stats['intersection'].item() if isinstance(coverage_stats['intersection'], torch.Tensor) else coverage_stats['intersection'],
                "mask_coverage_ratio": coverage_stats['coverage']
            }
            pd.DataFrame([test_result]).to_csv(os.path.join(output_dir, "final_test_result.csv"), index=False)
            print(f"Final test results saved to: {os.path.join(output_dir, 'final_test_result.csv')}")

    else:
        print(f"No independent test set found or invalid path specified. Skipping final test evaluation.")

    total_elapsed = time.time() - start_time
    print(f"\nTotal cross-validation and test process time: {total_elapsed/60:.2f} minutes")

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) > 1:
#         config_path = sys.argv[1]
#     else:
#         print("Please provide a config file path (e.g., python train.py configs/gan_medsam.yaml)")
#         sys.exit(1)
#
#     cfg = OmegaConf.load(config_path)
#     run_training_pipeline(cfg)

