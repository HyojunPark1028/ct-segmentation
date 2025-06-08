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
import glob # glob은 여전히 테스트셋 로딩 및 일반적인 경로 확인에 유용하므로 유지

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
    # torch.use_deterministic_algorithms(True) # 필요 시 추가

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
    
    total_seg_loss = 0
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)

            predicted_masks, _, _ = model(images, None)
            
            loss = seg_criterion(predicted_masks, masks)
            total_seg_loss += loss.item()
            pbar.set_postfix({"Seg_Loss": f"{loss.item():.4f}"})
            
    avg_seg_loss = total_seg_loss / len(dataloader)
    
    metrics = evaluate(model, dataloader, device)
    
    return avg_seg_loss, metrics

def run_training_pipeline(cfg):
    start_time = time.time() # 전체 파이프라인 시작 시간
    seed_everything(cfg.seed) # 'cfg.experiment.seed' 대신 'cfg.seed' 사용
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() and cfg.gpu is not None else "cpu")
    print(f"Using device: {device}")

    # Ensure save directory exists
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.output_dir, f"{cfg.model.name}_{current_time}")
    os.makedirs(output_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml")) # "used_config.yaml" 대신 "config.yaml"로 통일


    # U-Net 체크포인트 경로 확인
    if not os.path.exists(cfg.model.unet_checkpoint):
        raise FileNotFoundError(f"U-Net checkpoint not found at: {cfg.model.unet_checkpoint}")
    
    # --- 1. K-Fold Cross-Validation을 위한 데이터 준비 ---
    # K-Fold를 위해 모든 train과 val 데이터를 합쳐서 사용 (기존 train.py 로직)
    all_image_full_paths = []
    all_mask_full_paths = []

    # train 폴더의 파일 목록 추가
    train_img_base = os.path.join(cfg.data.data_dir, 'train', 'images') # cfg.data.root_dir 대신 cfg.data.data_dir
    train_mask_base = os.path.join(cfg.data.data_dir, 'train', 'masks') # cfg.data.root_dir 대신 cfg.data.data_dir
    if os.path.exists(train_img_base):
        train_files = sorted([f for f in os.listdir(train_img_base) if f.endswith('.npy')])
        for f in train_files:
            all_image_full_paths.append(os.path.join(train_img_base, f))
            all_mask_full_paths.append(os.path.join(train_mask_base, f.replace('.npy','.png')))
    
    # val 폴더의 파일 목록 추가
    val_img_base = os.path.join(cfg.data.data_dir, 'val', 'images') # cfg.data.root_dir 대신 cfg.data.data_dir
    val_mask_base = os.path.join(cfg.data.data_dir, 'val', 'masks') # cfg.data.root_dir 대신 cfg.data.data_dir
    if os.path.exists(val_img_base):
        val_files = sorted([f for f in os.listdir(val_img_base) if f.endswith('.npy')])
        for f in val_files:
            all_image_full_paths.append(os.path.join(val_img_base, f))
            all_mask_full_paths.append(os.path.join(val_mask_base, f.replace('.npy','.png')))

    if not all_image_full_paths:
        raise FileNotFoundError(f"No .npy files found in {train_img_base} or {val_img_base}. Please check your data path.")

    # 디버깅 출력 추가 (데이터 로딩 확인용)
    print(f"DEBUG: Data Root Dir: {cfg.data.data_dir}")
    print(f"DEBUG: Found {len(all_image_full_paths)} total images for K-Fold training/validation.")
    print(f"DEBUG: Found {len(all_mask_full_paths)} total masks for K-Fold training/validation.")
    if len(all_image_full_paths) > 0:
        print(f"DEBUG: First image path: {all_image_full_paths[0]}")
    if len(all_mask_full_paths) > 0:
        print(f"DEBUG: First mask path: {all_mask_full_paths[0]}")
    if len(all_image_full_paths) != len(all_mask_full_paths):
        raise ValueError("Mismatch between image and mask file counts for K-Fold data.")


    # KFold 설정 (cfg.kfold.n_splits 사용)
    kf = KFold(n_splits=cfg.kfold.n_splits, shuffle=True, random_state=cfg.seed) # cfg.train.k_folds 대신 cfg.kfold.n_splits 사용

    all_fold_best_metrics = [] # 각 폴드의 '최고' 성능 결과 저장
    
    # 모델의 총 파라미터 수를 저장할 변수 (첫 폴드에서 한 번만 계산)
    total_trainable_parameters = 0
    
    normalize_type = cfg.data.get("normalize_type", "default") # normalize_type config에서 가져오기

    # --- 2. K-Fold 루프 시작 ---
    for fold, (train_index, val_index) in enumerate(kf.split(all_image_full_paths)):
        print(f"\n--- Starting Fold {fold + 1}/{cfg.kfold.n_splits} ---")

        # 현재 폴드에 해당하는 훈련/검증 파일 전체 경로 리스트 생성
        fold_train_image_paths = [all_image_full_paths[i] for i in train_index]
        fold_train_mask_paths = [all_mask_full_paths[i] for i in train_index]
        fold_val_image_paths = [all_image_full_paths[i] for i in val_index]
        fold_val_mask_paths = [all_mask_full_paths[i] for i in val_index]

        # Dataset 및 DataLoader 생성
        train_ds = NpySegDataset(
            image_paths=fold_train_image_paths,
            mask_paths=fold_train_mask_paths,
            augment=cfg.data.augment, # cfg.augment 대신 cfg.data.augment
            img_size=cfg.data.image_size, # cfg.model.img_size 대신 cfg.data.image_size
            normalize_type=normalize_type
        )
        val_ds = NpySegDataset(
            image_paths=fold_val_image_paths,
            mask_paths=fold_val_mask_paths,
            augment=False, # Validation set should not be augmented
            img_size=cfg.data.image_size, # cfg.model.img_size 대신 cfg.data.image_size
            normalize_type=normalize_type
        )

        g = torch.Generator()
        g.manual_seed(cfg.seed + fold) # cfg.experiment.seed 대신 cfg.seed

        train_dl = DataLoader(train_ds, batch_size=cfg.dataloader.batch_size, shuffle=True, # cfg.train.batch_size 대신 cfg.dataloader.batch_size
                              num_workers=cfg.dataloader.num_workers, pin_memory=True, worker_init_fn=None, generator=g) # worker_init_fn=seed_worker 제거 (Glob 로딩 시 불필요)
        val_dl = DataLoader(val_ds, batch_size=cfg.dataloader.batch_size, shuffle=False, # cfg.train.batch_size 대신 cfg.dataloader.batch_size
                            num_workers=cfg.dataloader.num_workers, pin_memory=True)

        # 매 폴드마다 새로운 모델 초기화
        # get_model 함수는 기존 train.py에 정의되어 있음. MedSAM_GAN은 직접 초기화
        model = MedSAM_GAN(
            sam_checkpoint=cfg.model.sam_checkpoint,
            unet_checkpoint=cfg.model.unet_checkpoint,
            out_channels=cfg.model.out_channels
        ).to(device)

        # ⭐ 모델 파라미터 수 계산 (첫 번째 폴드에서만)
        if fold == 0:
            total_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model has {total_trainable_parameters:,} trainable parameters.")

        # Optimizer 및 Criterion 설정 (GAN에 맞게)
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
        
        seg_criterion = get_segmentation_loss().to(device)
        adv_criterion_D = get_discriminator_loss().to(device)
        adv_criterion_G = get_generator_adversarial_loss().to(device)

        best_val_dice = -float('inf')
        patience_counter = 0
        
        fold_epoch_metrics = []

        for epoch in range(1, cfg.epochs + 1): # cfg.train.epochs 대신 cfg.epochs
            train_g_loss, train_d_loss, train_seg_loss, train_g_adv_loss = train_one_epoch(
                model, train_dl, optimizer_G, optimizer_D,
                seg_criterion, adv_criterion_D, adv_criterion_G,
                device, epoch, cfg.log_interval, cfg.gan_lambda_adv
            )
            
            val_seg_loss, val_metrics = validate_one_epoch(model, val_dl, seg_criterion, device)
            
            # 여기서 val_inference_time_per_batch_sec 값을 가져와야 함 (evaluate 함수가 반환하는 경우)
            # 현재 evaluate 함수는 metrics만 반환하므로, evaluate 함수를 수정하거나
            # val_metrics 딕셔너리에 'val_inference_time_per_batch_sec' 키가 있다고 가정
            val_inference_time_per_batch_sec = val_metrics.get('val_inference_time_per_batch_sec', 0.0)

            epoch_log = {
                'fold': fold + 1,
                'epoch': epoch,
                'train_g_total_loss': train_g_loss,
                'train_d_loss': train_d_loss,
                'train_seg_loss': train_seg_loss,
                'train_g_adv_loss': train_g_adv_loss,
                'train_time_sec': 0.0, # ⭐ 측정 로직 필요
                'val_seg_loss': val_seg_loss,
                'val_dice': val_metrics.get('dice_score', -1.0), # evaluate에서 반환되는 키 확인
                'val_iou': val_metrics.get('iou_score', -1.0),   # evaluate에서 반환되는 키 확인
                'val_inference_time_per_batch_sec': val_inference_time_per_batch_sec,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            fold_epoch_metrics.append(epoch_log)
            print(epoch_log)

            current_val_dice = val_metrics.get('dice_score', 0.0)
            if current_val_dice > best_val_dice:
                best_val_dice = current_val_dice
                # model.state_dict()는 MedSAM_GAN 전체를 저장. 세부적으로 G와 D를 나누어 저장하는 것이 더 좋음.
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
                print(f"No improvement in val_dice. EarlyStopping counter: {patience_counter}/{cfg.early_stopping_patience}") # cfg.train.patience 대신 cfg.early_stopping_patience
                if patience_counter >= cfg.early_stopping_patience:
                    print(f"Early stopping triggered for Fold {fold+1}.")
                    break
            
            gc.collect()
            torch.cuda.empty_cache()

        if fold_epoch_metrics:
            df_fold_epochs = pd.DataFrame(fold_epoch_metrics)
            fold_epoch_metrics_path = os.path.join(fold_save_dir, f"fold_{fold+1}_epoch_metrics.csv")
            df_fold_epochs.to_csv(fold_epoch_metrics_path, index=False)
            print(f"    Epoch-wise metrics for Fold {fold+1} saved to: {fold_epoch_metrics_path}")

        print(f"\n--- Fold {fold + 1} Training Finished ---")
        best_model_path_this_fold = os.path.join(fold_save_dir, "model_best.pth")
        if os.path.exists(best_model_path_this_fold):
            # MedSAM_GAN 객체를 새로 만들고 state_dict를 로드
            model_eval = MedSAM_GAN(
                sam_checkpoint=cfg.model.sam_checkpoint,
                unet_checkpoint=cfg.model.unet_checkpoint,
                out_channels=cfg.model.out_channels
            ).to(device)
            checkpoint = torch.load(best_model_path_this_fold, map_location=device)
            model_eval.sam.load_state_dict(checkpoint['model_G_state_dict'])
            model_eval.discriminator.load_state_dict(checkpoint['model_D_state_dict'])
            model_eval.eval()
        else:
            print(f"Warning: No best model saved for Fold {fold+1}. Cannot perform final fold evaluation.")
            model_eval = model # 현재 학습된 마지막 모델 사용
        
        # val_dl 대신 val_ds를 사용하여 평가해야 하는지 확인 (evaluate 함수의 입력에 따라 다름)
        final_fold_metrics = evaluate(model_eval, val_dl, device, cfg.data.threshold, vis=False) # evaluate 함수 호출
        coverage_stats_val = compute_mask_coverage(model_eval, val_dl, device, cfg.data.threshold)

        fold_best_metrics_entry = {
            'fold': fold + 1,
            'best_val_dice': final_fold_metrics.get('dice_score', -1.0),
            'best_val_loss': val_seg_loss, # 마지막 에폭의 val_seg_loss 사용
            'best_model_val_iou': final_fold_metrics.get('iou_score', -1.0),
            'val_mask_coverage_gt_pixels': coverage_stats_val['gt_pixels'].item() if isinstance(coverage_stats_val['gt_pixels'], torch.Tensor) else coverage_stats_val['gt_pixels'],
            'val_mask_coverage_pred_pixels': coverage_stats_val['pred_pixels'].item() if isinstance(coverage_stats_val['pred_pixels'], torch.Tensor) else coverage_stats_val['pred_pixels'],
            'val_mask_coverage_intersection': coverage_stats_val['intersection'].item() if isinstance(coverage_stats_val['intersection'], torch.Tensor) else coverage_stats_val['intersection'],
            'val_mask_coverage_coverage': coverage_stats_val['coverage'],
            'val_mask_coverage_overpredict': coverage_stats_val['overpredict'],
            'val_inference_time_per_batch_sec': val_metrics.get('val_inference_time_per_batch_sec', 0.0) # 해당 폴드 베스트 모델의 인퍼런스 시간
        }
        all_fold_best_metrics.append(fold_best_metrics_entry)
        
        print(f"Fold {fold+1} Validation Results (from best model) ➜ Dice:{final_fold_metrics.get('dice_score', -1.0):.4f} IoU:{final_fold_metrics.get('iou_score', -1.0):.4f}")
        for k, v in coverage_stats_val.items():
            if isinstance(v, torch.Tensor): v = v.item()
            print(f"    {k}: {v}")

        del model_eval # 평가용 모델 삭제
        gc.collect()
        torch.cuda.empty_cache()


    # --- 3. 모든 폴드 완료 후 최종 결과 집계 및 보고 ---
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

    output_dir_final = os.path.join(cfg.output_dir) # 최종 결과를 저장할 기본 outputs 폴더
    os.makedirs(output_dir_final, exist_ok=True)
    
    detailed_results_path = os.path.join(output_dir_final, "k_fold_cross_validation_best_val_results.csv")
    df_best_fold_results.to_csv(detailed_results_path, index=False)
    print(f"\nDetailed best validation results per fold saved to: {detailed_results_path}")

    avg_results_df = pd.DataFrame([avg_final_results])
    avg_results_path = os.path.join(output_dir_final, "k_fold_cross_validation_average_results.csv")
    avg_results_df.to_csv(avg_results_path, index=False)
    print(f"Average results across folds saved to: {avg_results_path}")

    # --- 4. 최종 테스트 세트 평가 (K-Fold에 포함되지 않은 독립적인 세트) ---
    print("\n--- Independent Test Set Evaluation ---")
    # cfg.data.test_img_dir는 test 폴더의 상위 경로가 아니라, test 폴더 자체여야 함
    test_img_base = os.path.join(cfg.data.data_dir, 'test', 'images') # cfg.data.root_dir 대신 cfg.data.data_dir
    test_mask_base = os.path.join(cfg.data.data_dir, 'test', 'masks') # cfg.data.root_dir 대신 cfg.data.data_dir


    test_files = sorted([f for f in os.listdir(test_img_base) if f.endswith('.npy')])
    test_image_paths = [os.path.join(test_img_base, f) for f in test_files]
    test_mask_paths = [os.path.join(test_mask_base, f.replace('.npy', '.png')) for f in test_files]

    if test_image_paths and len(test_image_paths) == len(test_mask_paths): # 유효한 테스트 파일이 있을 때만 진행
        final_model = MedSAM_GAN( # 최종 모델도 MedSAM_GAN으로 초기화
            sam_checkpoint=cfg.model.sam_checkpoint,
            unet_checkpoint=cfg.model.unet_checkpoint,
            out_channels=cfg.model.out_channels
        ).to(device)

        # K-Fold 결과에서 가장 높은 best_val_dice를 기록한 폴드의 모델을 로드
        if not df_best_fold_results.empty:
            best_fold_row = df_best_fold_results.loc[df_best_fold_results['best_val_dice'].idxmax()]
            best_fold_num = int(best_fold_row['fold'])
            best_model_path_for_test = os.path.join(cfg.output_dir, f"{cfg.model.name}_{current_time}", f"fold_{best_fold_num}", "model_best.pth") # 저장 경로 수정
            
            if os.path.exists(best_model_path_for_test):
                # G와 D의 state_dict를 각각 로드
                checkpoint = torch.load(best_model_path_for_test, map_location=device)
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
        ts_dl = DataLoader(test_ds, batch_size=cfg.dataloader.batch_size, shuffle=False, num_workers=cfg.dataloader.num_workers, pin_memory=True) # cfg.train.num_workers 대신 cfg.dataloader.num_workers

        # Loss criterion은 test에서는 사용하지 않음
        # criterion = get_loss() # 이 부분은 불필요 (evaluate 함수에서 자체적으로 처리)
        test_loss_sum = 0
        test_inference_times = []
        
        # evaluate 함수는 loss를 반환하지 않으므로, Test Loss는 여기서 직접 계산
        with torch.no_grad():
            for x_test, y_test in tqdm(ts_dl, desc="Evaluating Test Set"):
                x_test, y_test = x_test.to(device), y_test.to(device)
                
                torch.cuda.synchronize()
                start_inference = time.time()
                # MedSAM_GAN은 (마스크, IoU, D_output_gen) 튜플을 반환
                pred, _, _ = final_model(x_test, None) # GAN의 D 출력은 테스트 시 불필요
                torch.cuda.synchronize()
                end_inference = time.time()
                test_inference_times.append(end_inference - start_inference)

                if isinstance(pred, tuple): pred = pred[0] # 메인 마스크만 사용
                
                # Test Loss 계산
                loss_val = seg_criterion(pred, y_test) # seg_criterion 사용
                test_loss_sum += loss_val.item()
        
        avg_test_inference_time_per_batch = np.mean(test_inference_times) if test_inference_times else 0.0
        avg_test_loss = test_loss_sum / len(ts_dl) if len(ts_dl) > 0 else 0.0

        # evaluate 함수는 vis=cfg.eval.visualize 인자를 사용 (evaluate 함수가 이 인자를 받는다면)
        # cfg.eval.visualize는 configs/medsam.yaml에 있지만, configs/medsam_gan.yaml에는 없음
        # 따라서 evaluate 호출 시 vis 인자 제거 또는 config에 추가
        td_ti_metrics = evaluate(final_model, ts_dl, device, cfg.data.threshold) # vis=False 인자 제거 (evaluate 함수 정의에 따라)
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
        print(f"No independent test set found in {test_img_base}. Skipping final test evaluation.")

    total_elapsed = time.time() - start_time
    print(f"\nTotal cross-validation and test process time: {total_elapsed/60:.2f} minutes")

# if __name__ == "__main__": (이 부분은 외부 스크립트에서 main 함수를 호출하는 방식으로 변경되었으므로, 불필요)
#     import sys
#     if len(sys.argv) > 1:
#         config_path = sys.argv[1]
#     else:
#         config_path = 'unet.yaml'
#         
#     cfg = OmegaConf.load(config_path)
#     main(cfg)

