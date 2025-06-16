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
import glob
import inspect 

# For AMP (Automatic Mixed Precision)
from torch.amp import GradScaler, autocast

# 모델 임포트
from .models.medsam_gan import MedSAM_GAN

from .dataset import NpySegDataset
import cv2

# GAN Loss 함수 및 평가 스크립트 임포트
from .losses_GAN import get_segmentation_loss, get_discriminator_loss, get_generator_adversarial_loss 

# ⭐ 변경 사항: evaluate_GAN.py에서 evaluate와 compute_mask_coverage 임포트
from .evaluate_GAN import evaluate, compute_mask_coverage 

def seed_everything(seed=42):
    """
    재현 가능한 결과를 위해 난수 시드를 설정합니다.
    Args:
        seed (int): 난수 시드 값.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True) # 필요 시 추가 (성능 저하 가능성 있음)

def set_requires_grad(model_part: nn.Module, requires_grad: bool):
    """
    모델의 특정 부분(모듈)에 대한 파라미터의 requires_grad 속성을 설정합니다.
    GAN 학습 시 Generator와 Discriminator의 학습을 분리하기 위해 사용됩니다.
    Args:
        model_part (nn.Module): requires_grad를 설정할 모델의 부분 (예: model.sam, model.discriminator).
        requires_grad (bool): requires_grad를 True로 설정할지 False로 설정할지.
    """
    for param in model_part.parameters():
        param.requires_grad = requires_grad

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    seg_criterion: nn.Module,
    adv_criterion_D: nn.Module, # WGAN-GP Discriminator Loss
    adv_criterion_G: nn.Module, # WGAN-GP Generator Loss
    device: torch.device,
    epoch: int,
    log_interval: int,
    d_update_interval: int = 1,
    max_grad_norm: float = None,
    segmentation_weight: float = 1.0,
    adversarial_weight: float = 0.005,
    scaler_G: GradScaler = None,
    scaler_D: GradScaler = None
) -> tuple[float, float, float, float]:
    """
    GAN 모델의 한 에폭 훈련을 수행합니다.
    Generator와 Discriminator를 번갈아 학습시킵니다.
    """
    model.train() # 모델을 학습 모드로 설정
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    
    total_seg_loss = 0.0
    total_g_adv_loss = 0.0
    total_g_loss = 0.0
    total_d_loss = 0.0

    # g_step_count = 0
    # d_step_count = 0
    # epoch_start_time = time.time()

    # 각 배치에서 이미지와 마스크를 가져옵니다.
    for batch_idx, (images, masks) in enumerate(pbar): 
        images = images.to(device)
        masks = masks.to(device)
        
        # real_low_res_masks 준비: Discriminator 학습 시 '진짜' 마스크로 사용
        real_low_res_masks = F.interpolate(
            masks, size=(256, 256), mode='nearest'
        ).float()
        
        # Discriminator 입력용 이미지 준비: 1채널 CT 이미지를 3채널 RGB처럼 복제
        # Discriminator의 입력은 (Batch, 4, H, W) 이므로 이미지(3채널) + 마스크(1채널) 구성
        resized_image_rgb_for_D = F.interpolate(
            images.float(), size=(256, 256), mode='bilinear', align_corners=False
        ).repeat(1, 3, 1, 1)


        # --- 1. Discriminator (D) 학습 단계 ---
        # Discriminator 업데이트 빈도를 제어합니다.
        if (batch_idx + 1) % d_update_interval == 0:
            set_requires_grad(model.sam, False) # Generator (SAM)의 파라미터 그래디언트 비활성화
            set_requires_grad(model.discriminator, True) # Discriminator 파라미터 그래디언트 활성화
            
            optimizer_D.zero_grad()

            with autocast(device_type='cuda', enabled=scaler_D is not None): 
                # 1. 실제 마스크에 대한 Discriminator 출력 (D(real_samples))
                # MedSAM_GAN.forward는 (masks_1024_gen, iou_predictions_gen, discriminator_output_for_generated_mask, low_res_masks_256_gen, discriminator_output_for_real_mask)를 반환
                # ⭐ 수정: `real_low_res_masks`를 키워드 인자로 전달
                _, _, _, low_res_masks_256_gen, discriminator_output_for_real_mask = model(images, real_low_res_mask=real_low_res_masks)
                
                # 2. Generator (SAM)를 통해 가짜 마스크 생성 (D 학습 시 G는 고정)
                # `real_low_res_mask=None`을 키워드 인자로 전달
                _, _, discriminator_output_for_generated_mask_for_D_input, _, _ = model(images, real_low_res_mask=None)
                
                # WGAN-GP Discriminator 손실 계산 (pred_real, pred_fake, real_samples, fake_samples, discriminator_model)
                d_loss = adv_criterion_D(
                    discriminator_output_for_real_mask,          # D(real_samples)
                    discriminator_output_for_generated_mask_for_D_input, # D(fake_samples)
                    torch.cat([resized_image_rgb_for_D, real_low_res_masks], dim=1).detach(), # Real samples (for GP)
                    torch.cat([resized_image_rgb_for_D, low_res_masks_256_gen], dim=1).detach(), # Fake samples (for GP)
                    model.discriminator                          # Discriminator model (for GP)
                )

            if scaler_D is not None:
                scaler_D.scale(d_loss).backward()
                if max_grad_norm is not None and max_grad_norm > 0:
                    scaler_D.unscale_(optimizer_D) # 스케일링 해제
                    torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=max_grad_norm)
                scaler_D.step(optimizer_D)
                scaler_D.update()
                # d_step_count += 1 
            else:
                d_loss.backward()
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=max_grad_norm)
                optimizer_D.step()

            total_d_loss += d_loss.item()
            total_seg_loss += seg_loss.item()
            total_g_adv_loss += g_adv_loss.item()


        # --- 2. Generator (G) 학습 단계 ---
        set_requires_grad(model.discriminator, False)
        set_requires_grad(model.sam, True)

        optimizer_G.zero_grad()

        with autocast(device_type='cuda', enabled=scaler_G is not None): 
            # Generator를 통해 마스크를 생성하고, 이에 대한 Discriminator의 출력을 받습니다.
            # MedSAM_GAN.forward는 (masks_1024_gen, iou_predictions_gen, discriminator_output_for_generated_mask, low_res_masks_256_gen, discriminator_output_for_real_mask)를 반환
            gen_masks, iou_predictions, discriminator_output_for_generated_mask_for_G, _, _ = model(images, real_low_res_mask=None)

            # Segmentation Loss를 계산합니다. (생성된 마스크와 실제 마스크 간의 유사도)
            seg_loss = seg_criterion(gen_masks, masks)
            
            print(f"[MedSAM-GAN] Prediction mask shape: {gen_masks.shape}")  # e.g., [4, 1, 256, 256]
            print(f"[MedSAM-GAN] Ground truth mask shape: {masks.shape}")    # e.g., [4, 1, 256, 256]

            # Generator Adversarial Loss를 계산합니다. (WGAN-GP는 -D(G(z)) )
            g_adv_loss = adv_criterion_G(discriminator_output_for_generated_mask_for_G)


            # Generator의 총 손실은 Segmentation Loss와 Adversarial Loss를 가중치로 합산
            g_loss = (seg_loss * segmentation_weight) + (g_adv_loss * adversarial_weight)

        if scaler_G is not None:
            scaler_G.scale(g_loss).backward()

            if max_grad_norm is not None and max_grad_norm > 0:
                scaler_G.unscale_(optimizer_G) # 스케일링 해제
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.sam.parameters() if p.requires_grad],
                    max_norm=max_grad_norm
                )
            scaler_G.step(optimizer_G)
            scaler_G.update()
            # g_step_count += 1
        else:
            g_loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.sam.parameters() if p.requires_grad],
                    max_norm=max_grad_norm
                )
            optimizer_G.step()
        
        total_g_loss += g_loss.item()

        # 프로그레스 바에 현재 배치의 손실 값들을 표시합니다.
        pbar.set_postfix({
            "D_Loss": f"{d_loss.item():.4f}" if (batch_idx + 1) % d_update_interval == 0 else "N/A",
            "G_Seg_Loss": f"{seg_loss.item():.4f}",
            "G_Adv_Loss": f"{g_adv_loss.item():.4f}",
            "G_Total_Loss": f"{g_loss.item():.4f}"
        })

    # 에폭의 평균 손실 값들을 계산하여 반환합니다.
    avg_seg_loss = total_seg_loss / len(dataloader)
    avg_g_adv_loss = total_g_adv_loss / len(dataloader)
    avg_g_loss = total_g_loss / len(dataloader)
    avg_d_loss = total_d_loss / (len(dataloader) / d_update_interval)

    # epoch_time = time.time() - epoch_start_time
    # print(f"[Epoch {epoch+1}] Time: {epoch_time:.2f}s | G steps: {g_step_count}, D steps: {d_step_count}")

    return avg_g_loss, avg_d_loss, avg_seg_loss, avg_g_adv_loss

def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    seg_criterion: nn.Module,
    device: torch.device,
    cfg: OmegaConf
) -> tuple[float, dict]:
    """
    모델의 한 에폭 검증을 수행하고 Segmentation 손실 및 성능 지표를 반환합니다.
    """
    model.eval() # 모델을 평가 모드로 설정
    pbar = tqdm(dataloader, desc="Validation")
    
    total_seg_loss = 0.0
    
    val_inference_times = []

    set_requires_grad(model.discriminator, False)
    set_requires_grad(model.sam, True)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar): 
            images = images.to(device)
            masks = masks.to(device)

            torch.cuda.synchronize() 
            start_inference = time.time()

            # MedSAM_GAN.forward는 (masks_1024_gen, iou_predictions_gen, discriminator_output_for_generated_mask, low_res_masks_256_gen, discriminator_output_for_real_mask)를 반환
            # validate_one_epoch은 Generator의 출력 (predicted_masks)만 필요
            predicted_masks, _, _, _, _ = model(images, real_low_res_mask=None)
            
            torch.cuda.synchronize()
            end_inference = time.time()
            val_inference_times.append(end_inference - start_inference)
            
            seg_loss = seg_criterion(predicted_masks, masks)
            total_seg_loss += seg_loss.item()
            pbar.set_postfix({"Seg_Loss": f"{seg_loss.item():.4f}"})
            
    avg_seg_loss = total_seg_loss / len(dataloader)
    
    # evaluate_GAN.py의 evaluate 함수는 딕셔너리를 반환하므로, 그대로 받아서 사용
    metrics = evaluate(model, dataloader, device, thr=cfg.data.threshold)
    
    metrics['val_inference_time_per_batch_sec'] = np.mean(val_inference_times) if val_inference_times else 0.0
    
    return avg_seg_loss, metrics

def run_training_pipeline(cfg: OmegaConf):
    """
    K-Fold Cross-Validation을 사용하여 GAN 모델을 훈련하고 평가하는 전체 파이프라인을 실행합니다.
    """
    start_time = time.time()
    seed_everything(cfg.seed)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() and cfg.gpu is not None else "cpu")
    print(f"Using device: {device}")

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.output_dir, f"{cfg.model.name}_{current_time}")
    os.makedirs(output_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))

    if not os.path.exists(cfg.model.unet_checkpoint):
        raise FileNotFoundError(f"U-Net checkpoint not found at: {cfg.model.unet_checkpoint}")
    
    all_image_full_paths = []
    all_mask_full_paths = []

    train_img_base = os.path.join(cfg.data.data_dir, 'train', 'images')
    train_mask_base = os.path.join(cfg.data.data_dir, 'train', 'masks')
    if os.path.exists(train_img_base):
        train_files = sorted([f for f in os.listdir(train_img_base) if f.endswith('.npy')])
        for f in train_files:
            all_image_full_paths.append(os.path.join(train_img_base, f))
            # 마스크 파일은 .npy 확장자를 .png로 대체하여 매칭합니다.
            all_mask_full_paths.append(os.path.join(train_mask_base, f.replace('.npy','.png')))
    
    val_img_base = os.path.join(cfg.data.data_dir, 'val', 'images')
    val_mask_base = os.path.join(cfg.data.data_dir, 'val', 'masks')
    if os.path.exists(val_img_base):
        val_files = sorted([f for f in os.listdir(val_img_base) if f.endswith('.npy')])
        for f in val_files:
            all_image_full_paths.append(os.path.join(val_img_base, f))
            all_mask_full_paths.append(os.path.join(val_mask_base, f.replace('.npy','.png')))

    if not all_image_full_paths:
        raise FileNotFoundError(f"No .npy files found in {train_img_base} or {val_img_base}. Please check your data path.")
    if len(all_image_full_paths) != len(all_mask_full_paths):
        raise ValueError("Mismatch between image and mask file counts for K-Fold data. Ensure every image has a corresponding mask.")

    print(f"DEBUG: Data Root Dir: {cfg.data.data_dir}")
    print(f"DEBUG: Found {len(all_image_full_paths)} total images for K-Fold training/validation.")
    print(f"DEBUG: Found {len(all_mask_full_paths)} total masks for K-Fold training/validation.")
    if len(all_image_full_paths) > 0:
        print(f"DEBUG: First image path: {all_image_full_paths[0]}")
    if len(all_mask_full_paths) > 0:
        print(f"DEBUG: First mask path: {all_mask_full_paths[0]}")


    kf = KFold(n_splits=cfg.kfold.n_splits, shuffle=True, random_state=cfg.seed)

    all_fold_best_metrics = []
    
    total_trainable_parameters = 0
    
    normalize_type = cfg.data.get("normalize_type", "default")

    # ⭐ 수정: GradScaler 초기화 방법 변경
    scaler_G = GradScaler(enabled=torch.cuda.is_available())
    scaler_D = GradScaler(enabled=torch.cuda.is_available())

    for fold, (train_index, val_index) in enumerate(kf.split(all_image_full_paths)):
        print(f"\n--- Starting Fold {fold + 1}/{cfg.kfold.n_splits} ---")

        fold_save_dir = os.path.join(output_dir, f"fold_{fold+1}")
        os.makedirs(fold_save_dir, exist_ok=True)

        fold_train_image_paths = [all_image_full_paths[i] for i in train_index]
        fold_train_mask_paths = [all_mask_full_paths[i] for i in train_index]
        fold_val_image_paths = [all_image_full_paths[i] for i in val_index]
        fold_val_mask_paths = [all_mask_full_paths[i] for i in val_index]

        train_ds = NpySegDataset(
            image_paths=fold_train_image_paths,
            mask_paths=fold_train_mask_paths,
            augment=cfg.data.augment,
            img_size=cfg.data.image_size,
            normalize_type=normalize_type
        )
        val_ds = NpySegDataset(
            image_paths=fold_val_image_paths,
            mask_paths=fold_val_mask_paths,
            augment=False,
            img_size=cfg.data.image_size,
            normalize_type=normalize_type
        )

        g = torch.Generator()
        g.manual_seed(cfg.seed + fold)

        train_dl = DataLoader(train_ds, batch_size=cfg.dataloader.batch_size, shuffle=True,
                              num_workers=cfg.dataloader.num_workers, pin_memory=True, generator=g)
        val_dl = DataLoader(val_ds, batch_size=cfg.dataloader.batch_size, shuffle=False,
                            num_workers=cfg.dataloader.num_workers, pin_memory=True)

        model = MedSAM_GAN(
            sam_checkpoint=cfg.model.sam_checkpoint,
            unet_checkpoint=cfg.model.unet_checkpoint,
            out_channels=cfg.model.out_channels
        ).to(device)

        # 모델 총 파라미터 수 확인
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[모델 정보] 총 파라미터 수: {total_params:,}")
        print(f"[모델 정보] 학습 가능한 파라미터 수: {trainable_params:,}")

        if fold == 0:
            # ⭐ 디버그 추가: MedSAM_GAN.forward의 실제 시그니처 출력
            print(f"DEBUG: Actual MedSAM_GAN.forward signature: {inspect.signature(model.forward)}")
            total_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model has {total_trainable_parameters:,} trainable parameters.")

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
        
        lambda_gp = cfg.losses.get('lambda_gp', 10.0)

        w_dice = cfg.losses.dice_focal_loss_weights.get('w_dice', 0.5)
        w_focal = cfg.losses.dice_focal_loss_weights.get('w_focal', 0.5)
        alpha_focal = cfg.losses.focal_loss_params.get('alpha', 0.8)
        gamma_focal = cfg.losses.focal_loss_params.get('gamma', 2.0)

        seg_criterion = get_segmentation_loss(
            w_dice=w_dice,
            w_focal=w_focal,
            alpha_focal=alpha_focal,
            gamma_focal=gamma_focal
        ).to(device)
        adv_criterion_D = get_discriminator_loss(lambda_gp=lambda_gp).to(device)
        adv_criterion_G = get_generator_adversarial_loss().to(device)

        segmentation_weight = cfg.losses.get('segmentation_weight', 1.0)
        adversarial_weight = cfg.losses.get('adversarial_weight', 0.005)

        best_val_dice = -float('inf')
        patience_counter = 0
        
        fold_epoch_metrics = []

        for epoch in range(1, cfg.epochs + 1):
            epoch_start_time = time.time()
            train_g_loss, train_d_loss, train_seg_loss, train_g_adv_loss = train_one_epoch(
                model, train_dl, optimizer_G, optimizer_D,
                seg_criterion, adv_criterion_D, adv_criterion_G,
                device, epoch, cfg.log_interval,
                cfg.optimizer.d_update_interval,
                cfg.optimizer.get('max_grad_norm', None),
                segmentation_weight,
                adversarial_weight,
                scaler_G,
                scaler_D
            )
            epoch_train_time_sec = time.time() - epoch_start_time
            
            val_seg_loss, val_metrics_dict = validate_one_epoch(model, val_dl, seg_criterion, device, cfg)
            
            val_inference_time_per_batch_sec = val_metrics_dict.get('val_inference_time_per_batch_sec', 0.0)

            epoch_log = {
                'fold': fold + 1,
                'epoch': epoch,
                'train_g_total_loss': train_g_loss,
                'train_d_loss': train_d_loss,
                'train_seg_loss': train_seg_loss,
                'train_g_adv_loss': train_g_adv_loss,
                'train_time_sec': epoch_train_time_sec,
                'val_seg_loss': val_seg_loss,
                'val_dice': val_metrics_dict.get('dice_score', -1.0),
                'val_iou': val_metrics_dict.get('iou_score', -1.0),
                'val_inference_time_per_batch_sec': val_inference_time_per_batch_sec,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            fold_epoch_metrics.append(epoch_log)
            print(epoch_log)

            current_val_dice = val_metrics_dict.get('dice_score', 0.0)
            if current_val_dice > best_val_dice:
                best_val_dice = current_val_dice
                torch.save({
                    'epoch': epoch,
                    'model_G_state_dict': model.sam.state_dict(),
                    'model_D_state_dict': model.discriminator.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                    'best_val_dice': best_val_dice,
                    'cfg': cfg
                }, os.path.join(fold_save_dir, "model_best.pth"))
                patience_counter = 0
                print(f"    Saved best model for Fold {fold+1} with validation dice: {best_val_dice:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement in val_dice. EarlyStopping counter: {patience_counter}/{cfg.early_stopping_patience}")
                if patience_counter >= cfg.early_stopping_patience:
                    print(f"Early stopping triggered for Fold {fold+1}.")
                    break
            
            gc.collect()
            torch.cuda.empty_cache()

        if fold_epoch_metrics:
            df_fold_epochs = pd.DataFrame(fold_epoch_metrics)
            fold_epoch_metrics_path = os.path.join(fold_save_dir, "epoch_metrics.csv")
            df_fold_epochs.to_csv(fold_epoch_metrics_path, index=False)
            print(f"    Epoch-wise metrics for Fold {fold+1} saved to: {fold_epoch_metrics_path}")

        print(f"\n--- Fold {fold + 1} Training Finished ---")
        best_model_path_this_fold = os.path.join(fold_save_dir, "model_best.pth")

        if os.path.exists(best_model_path_this_fold):
            model_eval = MedSAM_GAN(
                sam_checkpoint=cfg.model.sam_checkpoint,
                unet_checkpoint=cfg.model.unet_checkpoint,
                out_channels=cfg.model.out_channels
            ).to(device)
            checkpoint = torch.load(best_model_path_this_fold, map_location=device, weights_only=False)
            model_eval.sam.load_state_dict(checkpoint['model_G_state_dict'])
            model_eval.discriminator.load_state_dict(checkpoint['model_D_state_dict'])
            model_eval.eval()
        else:
            print(f"Warning: No best model saved for Fold {fold+1}. Cannot perform final fold evaluation. Using last trained model state.")
            model_eval = model
        
        final_fold_metrics = evaluate(model_eval, val_dl, device, thr=cfg.data.threshold)
        coverage_stats_val = compute_mask_coverage(model_eval, val_dl, device, cfg.data.threshold)

        fold_best_metrics_entry = {
            'fold': fold + 1,
            'best_val_dice': final_fold_metrics.get('dice_score', -1.0),
            'best_val_loss': val_seg_loss,
            'best_model_val_iou': final_fold_metrics.get('iou_score', -1.0),
            'val_mask_coverage_gt_pixels': coverage_stats_val['gt_pixels'].item() if isinstance(coverage_stats_val['gt_pixels'], torch.Tensor) else coverage_stats_val['gt_pixels'],
            'val_mask_coverage_pred_pixels': coverage_stats_val['pred_pixels'].item() if isinstance(coverage_stats_val['pred_pixels'], torch.Tensor) else coverage_stats_val['pred_pixels'],
            "inter_total": coverage_stats_val['intersection'].item() if isinstance(coverage_stats_val['intersection'], torch.Tensor) else coverage_stats_val['intersection'],
            'val_mask_coverage_coverage': coverage_stats_val['coverage'],
            'val_mask_coverage_overpredict': coverage_stats_val['overpredict'],
            'val_inference_time_per_batch_sec': val_metrics_dict.get('val_inference_time_per_batch_sec', 0.0)
        }
        all_fold_best_metrics.append(fold_best_metrics_entry)
        
        print(f"Fold {fold+1} Validation Results (from best model) ➜ Dice:{final_fold_metrics.get('dice_score', -1.0):.4f} IoU:{final_fold_metrics.get('iou_score', -1.0):.4f}")
        for k, v in coverage_stats_val.items():
            if isinstance(v, torch.Tensor): v = v.item()
            print(f"    {k}: {v}")

        del model_eval
        gc.collect()
        torch.cuda.empty_cache()


    print("\n--- All Folds Completed ---")
    df_best_fold_results = pd.DataFrame(all_fold_best_metrics)
    print("\nIndividual Fold Best Validation Results:")
    print(df_best_fold_results)

    avg_final_results = df_best_fold_results.mean(numeric_only=True).to_dict()
    avg_final_results['total_trainable_parameters'] = total_trainable_parameters

    print("\nAverage Best Validation Results Across Folds:")
    for k, v in avg_final_results.items():
        if k == 'total_trainable_parameters':
            print(f"    {k}: {int(v):,}")
        else:
            print(f"    {k}: {v:.4f}")

    output_dir_final = os.path.join(cfg.output_dir)
    os.makedirs(output_dir_final, exist_ok=True)
    
    detailed_results_path = os.path.join(output_dir_final, f"{cfg.model.name}_k_fold_cross_validation_best_val_results_{current_time}.csv")
    df_best_fold_results.to_csv(detailed_results_path, index=False)
    print(f"\nDetailed best validation results per fold saved to: {detailed_results_path}")

    avg_results_df = pd.DataFrame([avg_final_results])
    avg_results_path = os.path.join(output_dir_final, f"{cfg.model.name}_k_fold_cross_validation_average_results_{current_time}.csv")
    avg_results_df.to_csv(avg_results_path, index=False)
    print(f"Average results across folds saved to: {avg_results_path}")

    print("\n--- Independent Test Set Evaluation ---")
    test_img_base = os.path.join(cfg.data.data_dir, 'test', 'images')
    test_mask_base = os.path.join(cfg.data.data_dir, 'test', 'masks')

    test_files = sorted([f for f in os.listdir(test_img_base) if f.endswith('.npy')])
    test_image_paths = [os.path.join(test_img_base, f) for f in test_files]
    test_mask_paths = [os.path.join(test_mask_base, f.replace('.npy', '.png')) for f in test_files]

    if test_image_paths and len(test_image_paths) == len(test_mask_paths):
        final_model = MedSAM_GAN(
            sam_checkpoint=cfg.model.sam_checkpoint,
            unet_checkpoint=cfg.model.unet_checkpoint,
            out_channels=cfg.model.out_channels
        ).to(device)

        if not df_best_fold_results.empty:
            best_fold_row = df_best_fold_results.loc[df_best_fold_results['best_val_dice'].idxmax()]
            best_fold_num = int(best_fold_row['fold'])
            best_model_path_for_test = os.path.join(output_dir, f"fold_{best_fold_num}", "model_best.pth")
            
            if os.path.exists(best_model_path_for_test):
                checkpoint = torch.load(best_model_path_for_test, map_location=device, weights_only=False)
                final_model.sam.load_state_dict(checkpoint['model_G_state_dict'])
                final_model.discriminator.load_state_dict(checkpoint['model_D_state_dict'])
                print(f"Loaded best performing model from Fold {best_fold_num} ({best_model_path_for_test}) for final test evaluation.")
            else:
                print(f"Warning: Best model for Fold {best_fold_num} not found at {best_model_path_for_test}. Using newly initialized model for test evaluation. This may lead to poor test results.")
        else:
            print("No K-Fold results found to select the best model. Using newly initialized model for test evaluation. This may lead to poor test results.")

        final_model.eval()
        
        test_ds = NpySegDataset(
            image_paths=test_image_paths,
            mask_paths=test_mask_paths,
            augment=False,
            img_size=cfg.data.image_size,
            normalize_type=normalize_type
        )
        ts_dl = DataLoader(test_ds, batch_size=cfg.dataloader.batch_size, shuffle=False, num_workers=cfg.dataloader.num_workers, pin_memory=True)

        test_loss_sum = 0.0
        test_inference_times = []
        
        with torch.no_grad():
            for x_test, y_test in tqdm(ts_dl, desc="Evaluating Test Set"):
                x_test, y_test = x_test.to(device), y_test.to(device)
                
                torch.cuda.synchronize()
                start_inference = time.time()
                # `MedSAM_GAN.forward`는 `image`를 첫 번째 위치 인자로, `real_low_res_mask`를 키워드 인자로 받습니다.
                # `predicted_masks_test, _, _, _, _ = final_model(x_test, real_low_res_mask=None)`
                # 이미 이 부분은 올바르게 수정되어 있습니다.
                predicted_masks_test, _, _, _, _ = final_model(x_test, real_low_res_mask=None)
                torch.cuda.synchronize()
                end_inference = time.time()
                test_inference_times.append(end_inference - start_inference)
                
                loss_val = seg_criterion(predicted_masks_test, y_test)
                test_loss_sum += loss_val.item()
        
        avg_test_inference_time_per_batch = np.mean(test_inference_times) if test_inference_times else 0.0
        avg_test_loss = test_loss_sum / len(ts_dl) if len(ts_dl) > 0 else 0.0

        td_ti_metrics = evaluate(final_model, ts_dl, device, thr=cfg.data.threshold)
        td = td_ti_metrics.get('dice_score', -1.0)
        ti = td_ti_metrics.get('iou_score', -1.0)

        print(f"FINAL TEST ➜ Dice:{td:.4f} IoU:{ti:.4f} Test Loss:{avg_test_loss:.4f} "
              f"Avg Inference Time per Batch:{avg_test_inference_time_per_batch:.6f}s")

        coverage_stats = compute_mask_coverage(final_model, ts_dl, device, cfg.data.threshold)
        for k, v in coverage_stats.items():
            if isinstance(v, torch.Tensor): v = v.item()
            print(f"    Test Mask {k}: {v}")

        pd.DataFrame([coverage_stats]).to_csv(os.path.join(output_dir, "test_coverage.csv"), index=False)

        test_result = {
            "test_dice": td,
            "test_iou": ti,
            "test_loss": round(avg_test_loss, 4),
            "test_inference_time_per_batch_sec": round(avg_test_inference_time_per_batch, 6),
            "param_count": total_trainable_parameters,
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