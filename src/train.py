# train.py

import time
import gc
import os, torch, pandas as pd
import random, numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset # Dataset 임포트 추가
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import KFold
from torch.amp import GradScaler, autocast
import shutil

# 모델 임포트
from .models.unet import UNet
from .models.transunet import TransUNet
from .models.swinunet import SwinUNet
from .models.utransvision import UTransVision
from .models.medsam import MedSAM
from .models.medsam2 import MedSAM2

from .dataset import NpySegDataset
import cv2 # cv2는 dataset.py에서 사용되므로, 여기에 직접적인 필요는 없을 수 있으나, 일반적으로 유틸리티로 임포트해두는 경우 있음

# 기존 import를 유지하되 KFoldNpySegDataset을 사용
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

def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_model(cfg, device):
    """Refactored to return a new model instance for each fold"""
    model_name = cfg.model.name.lower() # 소문자로 변환하여 일관성 유지
    in_channels = cfg.model.get('in_channels', 1)
    out_channels = cfg.model.get('out_channels', 1)
    use_pretrained = cfg.model.get("use_pretrained", False) # UNet, TransUNet, SwinUNet, UTransVision
    img_size = cfg.model.get('img_size') # TransUNet, SwinUNet, UTransVision, MedSAM2

    if model_name == 'unet':
        model = UNet(use_pretrained=use_pretrained, in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'transunet':
        model = TransUNet(img_size=img_size, num_classes=out_channels, use_pretrained=use_pretrained)
    elif model_name == 'swinunet':
        model = SwinUNet(img_size=img_size, num_classes=out_channels, use_pretrained=use_pretrained)
    elif model_name == 'utransvision':
        model = UTransVision(img_size=img_size, num_classes=out_channels, use_pretrained=use_pretrained)
    elif model_name == 'sam2unet':
        from .models.sam2unet import SAM2UNet # SAM2UNet은 필요할 때만 임포트
        model = SAM2UNet(checkpoint=cfg.model.checkpoint, config=cfg.model.config)
    elif model_name == 'medsam':
        model = MedSAM(sam_checkpoint=cfg.model.checkpoint,unet_checkpoint=cfg.model.unet_checkpoint)
    elif model_name == 'medsam2':
        model = MedSAM2(checkpoint=cfg.model.checkpoint, image_size=cfg.model.img_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def main(cfg):
    start_time = time.time()
    seed_everything(cfg.experiment.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure save directory exists
    os.makedirs(cfg.train.save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.train.save_dir, "used_config.yaml"))


    # --- 1. K-Fold Cross-Validation을 위한 데이터 준비 ---
    # K-Fold를 위해 모든 train과 val 데이터를 합쳐서 사용
    all_image_full_paths = []
    all_mask_full_paths = []

    # train 폴더의 파일 목록 추가
    train_img_base = os.path.join(cfg.data.root_dir, 'train', 'images')
    train_mask_base = os.path.join(cfg.data.root_dir, 'train', 'masks')
    if os.path.exists(train_img_base):
        train_files = sorted([f for f in os.listdir(train_img_base) if f.endswith('.npy')])
        for f in train_files:
            all_image_full_paths.append(os.path.join(train_img_base, f))
            all_mask_full_paths.append(os.path.join(train_mask_base, f.replace('.npy','.png')))
    
    # val 폴더의 파일 목록 추가
    val_img_base = os.path.join(cfg.data.root_dir, 'val', 'images')
    val_mask_base = os.path.join(cfg.data.root_dir, 'val', 'masks')
    if os.path.exists(val_img_base):
        val_files = sorted([f for f in os.listdir(val_img_base) if f.endswith('.npy')])
        for f in val_files:
            all_image_full_paths.append(os.path.join(val_img_base, f))
            all_mask_full_paths.append(os.path.join(val_mask_base, f.replace('.npy','.png')))

    if not all_image_full_paths:
        raise FileNotFoundError(f"No .npy files found in {train_img_base} or {val_img_base}. Please check your data path.")

    # KFold 설정
    kf = KFold(n_splits=cfg.train.k_folds, shuffle=True, random_state=cfg.experiment.seed) # k_folds 추가

    all_fold_best_metrics = [] # 각 폴드의 '최고' 성능 결과 저장
    
    # 모델의 총 파라미터 수를 저장할 변수 (첫 폴드에서 한 번만 계산)
    total_trainable_parameters = 0
    
    normalize_type = cfg.data.get("normalize_type", "default") # normalize_type config에서 가져오기

    # --- 2. K-Fold 루프 시작 ---
    for fold, (train_index, val_index) in enumerate(kf.split(all_image_full_paths)):
        print(f"\n--- Starting Fold {fold + 1}/{cfg.train.k_folds} ---")
        scaler = GradScaler()
        # 현재 폴드에 해당하는 훈련/검증 파일 전체 경로 리스트 생성
        fold_train_image_paths = [all_image_full_paths[i] for i in train_index]
        fold_train_mask_paths = [all_mask_full_paths[i] for i in train_index]
        fold_val_image_paths = [all_image_full_paths[i] for i in val_index]
        fold_val_mask_paths = [all_mask_full_paths[i] for i in val_index]

        # Dataset 및 DataLoader 생성 (KFoldNpySegDataset 사용)
        train_ds = NpySegDataset(
            image_paths=fold_train_image_paths,
            mask_paths=fold_train_mask_paths,
            augment=True,
            img_size=cfg.model.img_size,
            normalize_type=normalize_type
        )
        val_ds = NpySegDataset(
            image_paths=fold_val_image_paths,
            mask_paths=fold_val_mask_paths,
            augment=False, # Validation set should not be augmented
            img_size=cfg.model.img_size,
            normalize_type=normalize_type
        )

        g = torch.Generator() # DataLoader에 사용할 generator (seed_worker용)
        g.manual_seed(cfg.experiment.seed + fold) # 각 폴드마다 다른 시드 적용

        train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.train.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.train.num_workers, pin_memory=True) # val_dl은 shuffle=False, generator 불필요

        # 매 폴드마다 새로운 모델 초기화
        model = get_model(cfg, device)
        
        # ⭐ 모델 파라미터 수 계산 (첫 번째 폴드에서만)
        if fold == 0:
            total_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model has {total_trainable_parameters:,} trainable parameters.")

        # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
        criterion = get_loss()

        best_val_dice = -float('inf') # ⭐ Early Stopping 기준을 val_dice로 변경 (높을수록 좋음)
        patience_counter = 0

        # 폴드별 결과 저장 디렉토리 생성
        fold_save_dir = os.path.join(cfg.train.save_dir, f"fold_{fold+1}")
        os.makedirs(fold_save_dir, exist_ok=True)
        
        # 각 폴드 내의 에포크별 메트릭을 저장할 리스트
        fold_epoch_metrics = []

        for epoch in range(1, cfg.train.epochs + 1):
            model.train()
            train_loss = 0
            train_start = time.time()
            
            # 훈련 루프
            loop = tqdm(train_dl, desc=f"Fold {fold+1} Epoch {epoch}/{cfg.train.epochs}", leave=False)
            for x, y in loop:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad();
                
                # ⭐ MedSAM 모델일 경우에만 prompt_masks(y) 전달
                with autocast(device_type='cuda'):
                    if isinstance(model, MedSAM):
                        preds, preds_iou_for_log = model(x) # MedSAM은 (마스크, IoU) 튜플 반환
                        loss = criterion(preds, y)
                    else:
                        preds = model(x) # output 이름은 preds로 통일
                        # ⭐ Deep Supervision Loss 처리
                        if isinstance(preds, tuple):
                            pred_main, *pred_deep = preds
                            loss = 0.98 * criterion(pred_main, y)
                            for i, p in enumerate(pred_deep):
                                try:
                                    h, w = y.shape[2], y.shape[3]
                                    p = nn.Conv2d(p.shape[1], 1, kernel_size=1).to(p.device)(p)
                                    p_up = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
                                    loss += 0.02 * criterion(p_up, y)
                                except Exception as e:
                                    print(f"[ERROR] in deep supervision loss for p[{i}]: {e}")
                                    # raise # 에러 발생 시 훈련 중단 대신 메시지 출력 후 계속 진행 (또는 필요시 중단)
                                    # 현재는 에러 시 중단하는 기존 동작 유지

                        else: # 단일 출력 모델
                            loss = criterion(preds, y)
                
                scaler.scale(loss).backward()
                # ⭐ Gradient Clipping 적용
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                
                # 훈련 중 현재 배치에 대한 평균 예측값 로깅
                if isinstance(model, MedSAM):
                    loop.set_postfix(loss=loss.item(), iou_pred=preds_iou_for_log.mean().item()) 
                else:
                    pred_for_log = preds[0] if isinstance(preds, tuple) else preds
                    loop.set_postfix(loss=loss.item(), mean_pred=torch.sigmoid(pred_for_log).mean().item())
                
                del x, y, loss, preds # 공통 변수 및 preds 삭제
                if isinstance(model, MedSAM): # MedSAM 관련 변수 추가 삭제
                    del preds_iou_for_log 
                torch.cuda.empty_cache()
                gc.collect()

            train_end = time.time(); train_elapsed = train_end - train_start
            avg_train_loss = train_loss / len(train_dl)

            model.eval()
            val_loss = 0
            val_inference_times = [] # ⭐ 배치당 인퍼런스 시간 측정
            val_start = time.time()
            with torch.no_grad():
                for x_val, y_val in val_dl:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    

                    torch.cuda.synchronize() # GPU 작업이 완료될 때까지 기다림
                    start_inference = time.time()
                    if isinstance(model, MedSAM):
                        v_pred, v_preds_iou_for_log = model(x_val)
                    else:
                        v_pred = model(x_val)
                    torch.cuda.synchronize() # GPU 작업이 완료될 때까지 기다림
                    end_inference = time.time()
                    val_inference_times.append(end_inference - start_inference)

                    if isinstance(v_pred, tuple): v_pred = v_pred[0]
                    v_loss = criterion(v_pred, y_val)
                    val_loss += v_loss.item()
            val_elapsed = time.time() - val_start
            avg_val_loss = val_loss / len(val_dl)
            avg_val_inference_time_per_batch = np.mean(val_inference_times) if val_inference_times else 0.0

            # 전체 validation set에 대한 Dice, IoU 평가 (시각화 없음)
            vd, vi = evaluate(model, val_dl, device, cfg.data.threshold, vis=False)

            # ⭐ 에포크별 진행 상황 딕셔너리 출력
            epoch_log = {
                'fold': fold + 1,
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'train_time_sec': round(train_elapsed, 2),
                'val_loss': avg_val_loss,
                'val_dice': vd,
                'val_iou': vi,
                'val_inference_time_per_batch_sec': round(avg_val_inference_time_per_batch, 6),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            fold_epoch_metrics.append(epoch_log)
            print(epoch_log) # 딕셔너리 형태로 출력

            # ⭐ Early Stopping 및 Model Saving (val_dice 기준으로 변경)
            if vd > best_val_dice: # Dice Score는 높을수록 좋으므로 > 사용
                best_val_dice = vd
                torch.save(model.state_dict(), os.path.join(fold_save_dir, "model_best.pth"))
                patience_counter = 0
                print(f"    Saved best model for Fold {fold+1} with validation dice: {best_val_dice:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement in val_dice. EarlyStopping counter: {patience_counter}/{cfg.train.patience}")
                if patience_counter >= cfg.train.patience:
                    print(f"Early stopping triggered for Fold {fold+1}.")
                    break

            # Garbage collection
            gc.collect()
            torch.cuda.empty_cache()

        # --- Fold 학습 완료 후 처리 ---
        # 폴드별 에포크 메트릭 저장
        if fold_epoch_metrics:
            df_fold_epochs = pd.DataFrame(fold_epoch_metrics)
            fold_epoch_metrics_path = os.path.join(fold_save_dir, f"fold_{fold+1}_epoch_metrics.csv")
            df_fold_epochs.to_csv(fold_epoch_metrics_path, index=False)
            print(f"    Epoch-wise metrics for Fold {fold+1} saved to: {fold_epoch_metrics_path}")

        # Fold 학습 완료 후 최종 평가 (최적 모델 로드)
        print(f"\n--- Fold {fold + 1} Training Finished ---")
        best_model_path_this_fold = os.path.join(fold_save_dir, "model_best.pth")
        if os.path.exists(best_model_path_this_fold):
            model.load_state_dict(torch.load(best_model_path_this_fold))
        else:
            print(f"Warning: No best model saved for Fold {fold+1}. Using current model state.")
        model.eval()

        val_dice_at_best, val_iou_at_best = evaluate(model, val_dl, device, cfg.data.threshold, vis=False)
        coverage_stats_val = compute_mask_coverage(model, val_dl, device, cfg.data.threshold)

        # 현재 폴드에서 기록된 가장 좋은 val_dice를 가진 모델의 인퍼런스 시간을 사용
        best_epoch_log = next((item for item in reversed(fold_epoch_metrics) if item['val_dice'] == best_val_dice), None)
        best_fold_inference_time = best_epoch_log['val_inference_time_per_batch_sec'] if best_epoch_log else 0.0

        fold_best_metrics_entry = {
            'fold': fold + 1,
            'best_val_dice': best_val_dice, # Early stopping 기준이었던 best_val_dice를 기록
            'best_val_loss': avg_val_loss, # 마지막 에포크의 val_loss 혹은 best_val_loss (초기화 방식에 따라)
            'best_model_val_iou': val_iou_at_best,
            'val_mask_coverage_gt_pixels': coverage_stats_val['gt_pixels'].item() if isinstance(coverage_stats_val['gt_pixels'], torch.Tensor) else coverage_stats_val['gt_pixels'],
            'val_mask_coverage_pred_pixels': coverage_stats_val['pred_pixels'].item() if isinstance(coverage_stats_val['pred_pixels'], torch.Tensor) else coverage_stats_val['pred_pixels'],
            'val_mask_coverage_intersection': coverage_stats_val['intersection'].item() if isinstance(coverage_stats_val['intersection'], torch.Tensor) else coverage_stats_val['intersection'],
            'val_mask_coverage_coverage': coverage_stats_val['coverage'],
            'val_mask_coverage_overpredict': coverage_stats_val['overpredict'],
            'val_inference_time_per_batch_sec': best_fold_inference_time # ⭐ 해당 폴드 베스트 모델의 인퍼런스 시간
        }
        all_fold_best_metrics.append(fold_best_metrics_entry)
        
        print(f"Fold {fold+1} Validation Results (from best model) ➜ Dice:{val_dice_at_best:.4f} IoU:{val_iou_at_best:.4f}")
        for k, v in coverage_stats_val.items():
            if isinstance(v, torch.Tensor): v = v.item()
            print(f"    {k}: {v}")

        del model, optimizer, criterion # 매 폴드마다 초기화
        gc.collect()
        torch.cuda.empty_cache()


    # --- 3. 모든 폴드 완료 후 최종 결과 집계 및 보고 ---
    print("\n--- All Folds Completed ---")
    df_best_fold_results = pd.DataFrame(all_fold_best_metrics)
    print("\nIndividual Fold Best Validation Results:")
    print(df_best_fold_results)

    avg_final_results = df_best_fold_results.mean(numeric_only=True).to_dict()
    # ⭐ 모델 파라미터 수 최종 결과에 추가
    avg_final_results['total_trainable_parameters'] = total_trainable_parameters

    print("\nAverage Best Validation Results Across Folds:")
    for k, v in avg_final_results.items():
        if k == 'total_trainable_parameters':
            print(f"    {k}: {int(v):,}")
        else:
            print(f"    {k}: {v:.4f}")

    # 최종 결과 저장 디렉토리 생성
    output_dir = cfg.train.save_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 상세 폴드별 최고 성능 결과 저장
    detailed_results_path = os.path.join(output_dir, "k_fold_cross_validation_best_val_results.csv")
    df_best_fold_results.to_csv(detailed_results_path, index=False)
    print(f"\nDetailed best validation results per fold saved to: {detailed_results_path}")

    # 평균 최종 결과 저장
    avg_results_df = pd.DataFrame([avg_final_results])
    avg_results_path = os.path.join(output_dir, "k_fold_cross_validation_average_results.csv")
    avg_results_df.to_csv(avg_results_path, index=False)
    print(f"Average results across folds saved to: {avg_results_path}")

    # --- 4. 최종 테스트 세트 평가 (K-Fold에 포함되지 않은 독립적인 세트) ---
    print("\n--- Independent Test Set Evaluation ---")
    test_img_dir = os.path.join(cfg.data.root_dir, 'test', 'images')
    test_mask_dir = os.path.join(cfg.data.root_dir, 'test', 'masks')

    test_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.npy')])
    test_image_paths = [os.path.join(test_img_dir, f) for f in test_files]
    test_mask_paths = [os.path.join(test_mask_dir, f.replace('.npy', '.png')) for f in test_files]

    if test_image_paths:

        final_model = get_model(cfg, device)

        # K-Fold 결과에서 가장 높은 best_val_dice를 기록한 폴드 찾기
        if not df_best_fold_results.empty:
            best_fold_row = df_best_fold_results.loc[df_best_fold_results['best_val_dice'].idxmax()]
            best_fold_num = int(best_fold_row['fold'])
            best_model_path_for_test = os.path.join(cfg.train.save_dir, f"fold_{best_fold_num}", "model_best.pth")
            
            if os.path.exists(best_model_path_for_test):
                final_model.load_state_dict(torch.load(best_model_path_for_test))
                print(f"Loaded best performing model from Fold {best_fold_num} ({best_model_path_for_test}) for final test evaluation.")
            else:
                print(f"Warning: Best model for Fold {best_fold_num} not found at {best_model_path_for_test}. Using newly initialized model for test evaluation.")
        else:
            print("No K-Fold results found to select the best model. Using newly initialized model for test evaluation.")

        final_model.eval()
        
        test_ds = NpySegDataset(
            image_paths=test_image_paths,
            mask_paths=test_mask_paths,
            augment=False,
            img_size=cfg.model.img_size,
            normalize_type=normalize_type
        )
        ts_dl = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

        criterion = get_loss() 
        test_loss_sum = 0
        test_inference_times = []
        with torch.no_grad():
            for x_test, y_test in tqdm(ts_dl, desc="Evaluating Test Set"):
                x_test, y_test = x_test.to(device), y_test.to(device)
                
                # if y_test.ndim == 4 and y_test.shape[-1] == 1: 
                #     y_test = y_test.permute(0, 3, 1, 2) 

                torch.cuda.synchronize()
                start_inference = time.time()
                if isinstance(final_model, MedSAM):
                    # MedSAM은 (마스크, IoU) 튜플을 반환. pred에는 마스크만 할당.
                    pred, pred_iou_for_log = final_model(x_test)
                else:
                    pred = final_model(x_test)
                torch.cuda.synchronize()
                end_inference = time.time()
                test_inference_times.append(end_inference - start_inference)

                if isinstance(pred, tuple): pred = pred[0]
                loss = criterion(pred, y_test)
                test_loss_sum += loss.item()
        
        avg_test_inference_time_per_batch = np.mean(test_inference_times) if test_inference_times else 0.0
        avg_test_loss = test_loss_sum / len(ts_dl) if len(ts_dl) > 0 else 0.0

        # evaluate 함수는 vis=cfg.eval.visualize 인자를 사용
        td, ti = evaluate(final_model, ts_dl, device, cfg.data.threshold, vis=cfg.eval.visualize)
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
            "param_count": total_trainable_parameters, # K-Fold에서 계산된 총 파라미터 수
            "gt_total": coverage_stats['gt_pixels'].item() if isinstance(coverage_stats['gt_pixels'], torch.Tensor) else coverage_stats['gt_pixels'],
            "pred_total": coverage_stats['pred_pixels'].item() if isinstance(coverage_stats['pred_pixels'], torch.Tensor) else coverage_stats['pred_pixels'],
            "inter_total": coverage_stats['intersection'].item() if isinstance(coverage_stats['intersection'], torch.Tensor) else coverage_stats['intersection'],
            "mask_coverage_ratio": coverage_stats['coverage']
        }
        pd.DataFrame([test_result]).to_csv(os.path.join(output_dir, "final_test_result.csv"), index=False)
        print(f"Final test results saved to: {os.path.join(output_dir, 'final_test_result.csv')}")

    else:
        print(f"No independent test set found in {test_img_dir}. Skipping final test evaluation.")

    total_elapsed = time.time() - start_time
    print(f"\nTotal cross-validation and test process time: {total_elapsed/60:.2f} minutes")

if __name__ == "__main__":
    # unet.yaml 파일 경로를 직접 지정하거나, 인자로 받을 수 있도록 변경
    # 예를 들어, python train.py unet.yaml
    import sys
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # 기본 unet.yaml 경로 (예: 프로젝트 루트의 unet.yaml)
        # config 파일을 로드하기 전에, 해당 파일의 위치를 확인해야 합니다.
        config_path = 'unet.yaml' 
        
    cfg = OmegaConf.load(config_path)
    main(cfg)