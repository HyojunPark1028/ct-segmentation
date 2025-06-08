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
from sklearn.model_selection import KFold
import shutil
import glob # ⭐ 추가: 파일 경로를 glob으로 찾기 위해 필요

# 모델 임포트
from .models.medsam_gan import MedSAM_GAN # Assuming MedSAM_GAN is in models/medsam_gan.py

from .dataset import NpySegDataset # NpySegDataset은 그대로 사용
import cv2 # cv2는 dataset.py에서 사용됨

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
    optimizer_G, # Generator 옵티마이저
    optimizer_D, # Discriminator 옵티마이저
    seg_criterion, # Segmentation Loss (DiceFocalLoss)
    adv_criterion_D, # Discriminator Loss
    adv_criterion_G, # Generator Adversarial Loss
    device,
    epoch,
    log_interval,
    gan_lambda_adv=0.1 # Generator Loss에서 adversarial loss의 가중치
):
    model.train() # 모델을 학습 모드로 설정
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    
    total_seg_loss = 0
    total_g_adv_loss = 0
    total_g_loss = 0
    total_d_loss = 0

    for batch_idx, (images, masks) in enumerate(pbar): # ⭐ 수정: 세 번째 반환 값 (idx) 제거
        images = images.to(device)
        masks = masks.to(device) # Original GT masks (H, W)

        # Discriminator에 입력할 저해상도 GT 마스크 준비 (256x256)
        real_low_res_masks = F.interpolate(
            masks, size=(256, 256), mode='nearest' # 'nearest' for binary masks
        ).float()


        # --- 1. Discriminator (D) 학습 ---
        optimizer_D.zero_grad()

        # Generator의 출력 (가짜 마스크) 얻기
        # 이 단계에서는 Generator의 파라미터는 업데이트하지 않으므로 torch.no_grad() 사용
        with torch.no_grad():
            # forward 호출 시 real_low_res_mask = None -> masks, iou_predictions, discriminator_output_for_generated_mask 반환
            gen_masks, _, disc_output_gen_for_G_no_grad = model(images, None)
        
        # Discriminator에 실제 마스크와 가짜 마스크 입력하여 손실 계산
        # model.forward()에서 real_low_res_mask가 있으면 4번째 리턴 값으로 real에 대한 D의 출력이 나옴
        # (gen_masks는 no_grad 블록에서 계산된 것이므로 여기서 다시 계산할 필요는 없음)
        _, _, disc_output_gen_for_D, disc_output_real = model(images, real_low_res_masks)

        # Discriminator 손실 계산
        d_loss = adv_criterion_D(disc_output_real, disc_output_gen_for_D)
        d_loss.backward()
        optimizer_D.step()

        total_d_loss += d_loss.item()


        # --- 2. Generator (G) 학습 ---
        optimizer_G.zero_grad()

        # Generator를 통해 마스크 다시 생성 (이번엔 G 파라미터 업데이트 위함)
        # Discriminator는 G 학습 시에는 고정 (eval 모드)하거나 requires_grad=False 로 설정
        # MedSAM_GAN forward는 항상 G 파라미터를 계산하므로, D 파라미터 업데이트를 막는 것이 중요
        # `discriminator_output_for_generated_mask`를 G의 adversarial loss 계산에 사용
        gen_masks, iou_predictions, disc_output_gen_for_G = model(images, None) # ⭐ 변수명 변경: D 학습 시와 구분

        # Segmentation Loss
        seg_loss = seg_criterion(gen_masks, masks)
        total_seg_loss += seg_loss.item()

        # Generator Adversarial Loss: Discriminator를 속여 진짜처럼 보이게 하는 손실
        # D_output_gen_for_G는 D가 G의 출력을 '진짜'라고 예측하도록 유도
        g_adv_loss = adv_criterion_G(disc_output_gen_for_G) # ⭐ 변수명 변경
        total_g_adv_loss += g_adv_loss.item()

        # 총 Generator 손실
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

    return avg_g_loss, avg_d_loss, avg_seg_loss, avg_g_adv_loss # 반환 값 변경

def validate_one_epoch(model, dataloader, seg_criterion, device):
    model.eval() # 모델을 평가 모드로 설정
    pbar = tqdm(dataloader, desc="Validation")
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar): # ⭐ 수정: 세 번째 반환 값 (idx) 제거
            images = images.to(device)
            masks = masks.to(device)

            # 검증 시에는 GAN Loss 계산이 주 목적이 아님 (주로 Segmentation 성능 확인)
            # real_low_res_mask = None 으로 호출하여 Generator 결과만 얻음
            # Masks: (B, 1, H, W)
            # iou_predictions: (B, 1)
            # discriminator_output_for_generated_mask: (B, 1, H_D, W_D)
            predicted_masks, _, _ = model(images, None) # GAN 관련 출력은 무시
            
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

    # U-Net 체크포인트 경로 확인
    if not os.path.exists(cfg.model.unet_checkpoint):
        raise FileNotFoundError(f"U-Net checkpoint not found at: {cfg.model.unet_checkpoint}")
    
    # ⭐ 수정: NpySegDataset에 전달할 이미지/마스크 경로 리스트 생성
    # 가정: data_dir 내부에 'images' 폴더와 'masks' 폴더가 있고, 파일 확장자는 .npy와 .png
    train_val_image_paths = sorted(glob.glob(os.path.join(cfg.data.data_dir, "images", "*.npy")))
    train_val_mask_paths = sorted(glob.glob(os.path.join(cfg.data.data_dir, "masks", "*.png"))) # mask는 .png로 가정

    if len(train_val_image_paths) == 0 or len(train_val_image_paths) != len(train_val_mask_paths):
        raise ValueError(f"No image/mask files found or mismatch in {cfg.data.data_dir}. Check paths and file types (.npy for images, .png for masks).")

    # Dataset 초기화 (KFold를 위해 전체 데이터셋을 한 번에 로드)
    dataset = NpySegDataset(
        image_paths=train_val_image_paths,
        mask_paths=train_val_mask_paths,
        augment=cfg.data.augment,
        img_size=cfg.data.image_size,
        normalize_type=cfg.data.normalize_type # ⭐ 추가: normalize_type 전달
    )

    all_indices = list(range(len(dataset)))

    kf = KFold(n_splits=cfg.kfold.n_splits, shuffle=True, random_state=cfg.seed) # 이 줄을 추가합니다.    
    
    fold_results = []
    
    start_time = time.time()

    for fold, (train_indices, val_indices) in enumerate(kf.split(all_indices)):
        print(f"\n--- Fold {fold+1}/{cfg.kfold.n_splits} ---")
        
        # KFold subset은 기존 dataset의 __getitem__을 사용하므로, 별도 수정 필요 없음
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

        # Initialize loss criteria
        seg_criterion = get_segmentation_loss()
        adv_criterion_D = get_discriminator_loss()
        adv_criterion_G = get_generator_adversarial_loss()

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
                
                # 모델 저장 (Generator와 Discriminator 모두 저장)
                torch.save({
                    'epoch': epoch,
                    'model_G_state_dict': model.sam.state_dict(), # SAM (Generator) 부분만 저장
                    'model_D_state_dict': model.discriminator.state_dict(), # Discriminator 저장
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
        # Load best model for evaluation
        checkpoint = torch.load(os.path.join(fold_output_dir, "best_model.pth"), map_location=device)
        model.sam.load_state_dict(checkpoint['model_G_state_dict']) # Generator 로드
        model.discriminator.load_state_dict(checkpoint['model_D_state_dict']) # Discriminator 로드
        
        # 모델을 평가 모드로 설정
        model.eval()
        
        # evaluate 함수가 Segmentation 성능만 측정한다고 가정
        metrics = evaluate(model, val_loader, device)
        fold_results.append(metrics)
        print(f"Fold {fold+1} Test Metrics (on validation set): {metrics}")
        
        # Clear CUDA cache
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
    if cfg.data.test_img_dir and os.path.exists(cfg.data.test_img_dir): # ⭐ 수정: cfg.test_img_dir -> cfg.data.test_img_dir
        print("\n--- Running final test on independent test set ---")
        # ⭐ 수정: test_img_dir로부터 이미지/마스크 경로 리스트 생성
        test_image_paths = sorted(glob.glob(os.path.join(cfg.data.test_img_dir, "images", "*.npy")))
        test_mask_paths = sorted(glob.glob(os.path.join(cfg.data.test_img_dir, "masks", "*.png"))) # mask는 .png로 가정

        if len(test_image_paths) == 0 or len(test_image_paths) != len(test_mask_paths):
            print(f"Warning: No image/mask files found or mismatch in test_img_dir {cfg.data.test_img_dir}. Skipping final test evaluation.")
            # test_img_dir이 유효하지 않으면 테스트를 건너뜀
            cfg.data.test_img_dir = None # 다음 단계에서 test_img_dir 검사 시 False가 되도록 설정
        else:
            test_dataset = NpySegDataset(
                image_paths=test_image_paths,
                mask_paths=test_mask_paths,
                augment=False, # 테스트셋은 증강하지 않음
                img_size=cfg.data.image_size,
                normalize_type=cfg.data.normalize_type # ⭐ 추가: normalize_type 전달
            )
            test_loader = DataLoader(test_dataset, batch_size=cfg.dataloader.batch_size, shuffle=False, num_workers=cfg.dataloader.num_workers, pin_memory=True)
            
            test_metrics = evaluate(model, test_loader, device)
            coverage_stats = compute_mask_coverage(model, test_loader, device) # compute_mask_coverage는 모델을 받도록 수정 필요
            
            print(f"Final Test Metrics: {test_metrics}")
            test_result = {
                **test_metrics,
                "total_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), # 전체 학습 가능한 파라미터 수 (SAM + Discriminator)
                "gt_total": coverage_stats['gt_pixels'].item() if isinstance(coverage_stats['gt_pixels'], torch.Tensor) else coverage_stats['gt_pixels'],
                "pred_total": coverage_stats['pred_pixels'].item() if isinstance(coverage_stats['pred_pixels'], torch.Tensor) else coverage_stats['pred_pixels'],
                "inter_total": coverage_stats['intersection'].item() if isinstance(coverage_stats['intersection'], torch.Tensor) else coverage_stats['intersection'],
                "mask_coverage_ratio": coverage_stats['coverage']
            }
            pd.DataFrame([test_result]).to_csv(os.path.join(output_dir, "final_test_result.csv"), index=False)
            print(f"Final test results saved to: {os.path.join(output_dir, 'final_test_result.csv')}")

    else:
        print(f"No independent test set found or invalid path specified. Skipping final test evaluation.") # ⭐ 메시지 변경

    total_elapsed = time.time() - start_time
    print(f"\nTotal cross-validation and test process time: {total_elapsed/60:.2f} minutes")

# if __name__ == "__main__": (이 부분은 외부 스크립트에서 main 함수를 호출하는 방식으로 변경되었으므로, 불필요)
#     import sys
#     if len(sys.argv) > 1:
#         config_path = sys.argv[1]
#     else:
#         print("Please provide a config file path (e.g., python train.py configs/gan_medsam.yaml)")
#         sys.exit(1)
#
#     cfg = OmegaConf.load(config_path)
#     run_training_pipeline(cfg)

