# train.py

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

# 모델 임포트 변경: MedSAM 대신 MedSAM_GAN 임포트
from .models.medsam_gan import MedSAM_GAN # Assuming you save MedSAM_GAN in models/medsam_gan.py
# If MedSAM_GAN is in the same medsam.py, then:
# from .models.medsam import MedSAM_GAN


from .dataset import NpySegDataset
import cv2

# 기존 import를 유지하되 GAN Loss 함수 추가
from .losses_GAN import get_segmentation_loss, get_discriminator_loss, get_generator_adversarial_loss
from .evaluate import evaluate, compute_mask_coverage

# 새로 추가되거나 변경된 부분 (MedSAM_GAN에 맞게)
# from .models.medsam import MedSAM # 기존 MedSAM은 더 이상 직접 사용 안함

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

    for batch_idx, (images, masks, _) in enumerate(pbar):
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
            gen_masks, _, disc_output_gen = model(images, None)
        
        # Discriminator에 실제 마스크와 가짜 마스크 입력하여 손실 계산
        # D는 진짜를 진짜(1)로, 가짜를 가짜(0)로 예측하도록 학습
        # model.forward()에서 real_low_res_mask가 있으면 4번째 리턴 값으로 real에 대한 D의 출력이 나옴
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
        gen_masks, iou_predictions, disc_output_gen = model(images, None)
        
        # Segmentation Loss
        seg_loss = seg_criterion(gen_masks, masks)
        total_seg_loss += seg_loss.item()

        # Generator Adversarial Loss: Discriminator를 속여 진짜처럼 보이게 하는 손실
        # D_output_gen_for_G는 D가 G의 출력을 '진짜'라고 예측하도록 유도
        g_adv_loss = adv_criterion_G(disc_output_gen)
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
        for batch_idx, (images, masks, _) in enumerate(pbar):
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
    
    # 모델 초기화 (MedSAM_GAN 사용)
    model = MedSAM_GAN(
        sam_checkpoint=cfg.model.sam_checkpoint,
        unet_checkpoint=cfg.model.unet_checkpoint,
        out_channels=cfg.model.out_channels
    ).to(device)

    # 옵티마이저 분리: Generator(SAM parts)와 Discriminator
    # Generator는 SAM의 학습 가능한 파라미터 (image_encoder, prompt_encoder, mask_decoder)
    optimizer_G = torch.optim.AdamW(
        params=[
            {'params': model.sam.image_encoder.parameters(), 'lr': cfg.optimizer.g_lr},
            {'params': model.sam.prompt_encoder.parameters(), 'lr': cfg.optimizer.g_lr},
            {'params': model.sam.mask_decoder.parameters(), 'lr': cfg.optimizer.g_lr}
        ],
        lr=cfg.optimizer.g_lr, # 기본 lr 설정
        weight_decay=cfg.optimizer.weight_decay
    )
    # Discriminator 옵티마이저
    optimizer_D = torch.optim.AdamW(
        model.discriminator.parameters(),
        lr=cfg.optimizer.d_lr, # Discriminator를 위한 별도의 lr
        weight_decay=cfg.optimizer.weight_decay
    )

    # 손실 함수 설정
    seg_criterion = get_segmentation_loss().to(device)
    adv_criterion_D = get_discriminator_loss().to(device)
    adv_criterion_G = get_generator_adversarial_loss().to(device)

    # 스케줄러 설정 (선택 사항, 필요에 따라 G와 D 각각 설정 가능)
    # scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5, verbose=True)
    # scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=5, verbose=True)


    # K-Fold Cross Validation 설정
    kf = KFold(n_splits=cfg.kfold.n_splits, shuffle=True, random_state=cfg.seed)
    
    dataset = NpySegDataset(
        data_dir=cfg.data.data_dir,
        image_size=cfg.data.image_size,
        num_classes=cfg.model.out_channels,
        augment=cfg.augment
    )

    all_indices = list(range(len(dataset)))
    
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
        # 중요: K-Fold마다 모델의 파라미터를 새로 로드하거나 초기화해야 함
        model = MedSAM_GAN(
            sam_checkpoint=cfg.model.sam_checkpoint,
            unet_checkpoint=cfg.model.unet_checkpoint,
            out_channels=cfg.model.out_channels
        ).to(device)

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
            # train_one_epoch 함수에 GAN 관련 파라미터 전달
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
                # SAM과 Discriminator의 state_dict를 따로 저장하거나, 모델 전체를 저장
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
    if cfg.test_img_dir and os.path.exists(cfg.test_img_dir):
        print("\n--- Running final test on independent test set ---")
        test_dataset = NpySegDataset(
            data_dir=cfg.test_img_dir,
            image_size=cfg.data.image_size,
            num_classes=cfg.model.out_channels,
            augment=False # 테스트셋은 증강하지 않음
        )
        test_loader = DataLoader(test_dataset, batch_size=cfg.dataloader.batch_size, shuffle=False, num_workers=cfg.dataloader.num_workers, pin_memory=True)
        
        # Best performing model from K-Fold (or a final combined model) should be used here
        # For simplicity, we can load the last saved best_model.pth (from the last fold)
        # or train a final model on all data. For now, let's just use the model as it is after K-Fold.
        
        # If you want to load a specific best model (e.g., from a specific fold or a new combined model):
        # model = MedSAM_GAN(...).to(device)
        # checkpoint = torch.load(os.path.join(output_dir, f"fold_X/best_model.pth"), map_location=device) # Choose best fold or final model
        # model.sam.load_state_dict(checkpoint['model_G_state_dict'])
        # model.discriminator.load_state_dict(checkpoint['model_D_state_dict'])
        # model.eval() # Ensure eval mode for testing

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
        print(f"No independent test set found in {cfg.test_img_dir}. Skipping final test evaluation.")

    total_elapsed = time.time() - start_time
    print(f"\nTotal cross-validation and test process time: {total_elapsed/60:.2f} minutes")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # 기본 unet.yaml 대신 새로운 gan_medsam.yaml 등을 사용하도록 변경 필요
        print("Please provide a config file path (e.g., python train.py configs/gan_medsam.yaml)")
        sys.exit(1) # 종료 또는 기본 config 로드

    cfg = OmegaConf.load(config_path)
    run_training_pipeline(cfg)