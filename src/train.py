# train.py

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
from sklearn.model_selection import KFold
import shutil

from .models.unet import UNet
from .models.transunet import TransUNet
from .models.swinunet import SwinUNet
from .models.utransvision import UTransVision
from .models.medsam import MedSAM
from .models.medsam2 import MedSAM2
from .dataset import NpySegDataset # 수정된 Dataset 임포트 (이름을 유지하거나 CrossFoldNpySegDataset으로 변경)
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
    model_name = cfg.model.name
    in_channels = cfg.model.in_channels if hasattr(cfg.model, 'in_channels') else 1
    out_channels = cfg.model.out_channels if hasattr(cfg.model, 'out_channels') else 1

    if model_name == 'unet':
        model = UNet(use_pretrained=cfg.model.use_pretrained, in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'transunet':
        model = TransUNet(img_size=cfg.model.img_size, in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'swinunet':
        model = SwinUNet(img_size=cfg.model.img_size, in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'utransvision':
        model = UTransVision(img_size=cfg.model.img_size, in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'medsam':
        model = MedSAM(
            image_encoder_model_type=cfg.model.image_encoder_model_type,
            image_encoder_ckpt_path=cfg.model.image_encoder_ckpt_path,
            decoder_model_type=cfg.model.decoder_model_type,
            decoder_ckpt_path=cfg.model.decoder_ckpt_path,
            in_channels=in_channels,
            out_channels=out_channels
        )
    elif model_name == 'medsam2':
        model = MedSAM2(
            medsam_ckpt_path=cfg.model.medsam_ckpt_path,
            prompt_encoder_ckpt_path=cfg.model.prompt_encoder_ckpt_path,
            mask_decoder_ckpt_path=cfg.model.mask_decoder_ckpt_path,
            in_channels=in_channels,
            out_channels=out_channels
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def main(cfg):
    start_time = time.time()
    seed_everything(cfg.experiment.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. K-Fold Cross-Validation을 위한 데이터 준비 ---
    train_img_base = os.path.join(cfg.data.root_dir, 'train', 'images')
    val_img_base = os.path.join(cfg.data.root_dir, 'val', 'images')
    train_mask_base = os.path.join(cfg.data.root_dir, 'train', 'masks')
    val_mask_base = os.path.join(cfg.data.root_dir, 'val', 'masks')

    all_image_full_paths = []
    all_mask_full_paths = []

    # train 폴더의 파일 목록 추가
    if os.path.exists(train_img_base):
        train_files = sorted([f for f in os.listdir(train_img_base) if f.endswith('.npy')])
        for f in train_files:
            all_image_full_paths.append(os.path.join(train_img_base, f))
            all_mask_full_paths.append(os.path.join(train_mask_base, f.replace('.npy','.png')))
    
    # val 폴더의 파일 목록 추가
    if os.path.exists(val_img_base):
        val_files = sorted([f for f in os.listdir(val_img_base) if f.endswith('.npy')])
        for f in val_files:
            all_image_full_paths.append(os.path.join(val_img_base, f))
            all_mask_full_paths.append(os.path.join(val_mask_base, f.replace('.npy','.png')))

    if not all_image_full_paths:
        raise FileNotFoundError(f"No .npy files found in {train_img_base} or {val_img_base}. Please check your data path.")

    # KFold 설정
    kf = KFold(n_splits=5, shuffle=True, random_state=cfg.experiment.seed)

    all_fold_best_metrics = [] # 각 폴드의 '최고' 성능 결과 저장
    
    # --- 2. K-Fold 루프 시작 ---
    for fold, (train_index, val_index) in enumerate(kf.split(all_image_full_paths)): # kf.split은 리스트 길이에 따라 인덱스 반환
        print(f"\n--- Starting Fold {fold + 1}/5 ---")

        # 현재 폴드에 해당하는 훈련/검증 파일 전체 경로 리스트 생성
        fold_train_image_paths = [all_image_full_paths[i] for i in train_index]
        fold_train_mask_paths = [all_mask_full_paths[i] for i in train_index]
        fold_val_image_paths = [all_image_full_paths[i] for i in val_index]
        fold_val_mask_paths = [all_mask_full_paths[i] for i in val_index]

        # Dataset 및 DataLoader 생성
        train_ds = NpySegDataset( # NpySegDataset 이름 유지
            image_paths=fold_train_image_paths,
            mask_paths=fold_train_mask_paths,
            augment=True,
            img_size=cfg.model.img_size,
            normalize_type=cfg.data.normalize_type if hasattr(cfg.data, 'normalize_type') else "default"
        )
        val_ds = NpySegDataset( # NpySegDataset 이름 유지
            image_paths=fold_val_image_paths,
            mask_paths=fold_val_mask_paths,
            augment=False,
            img_size=cfg.model.img_size,
            normalize_type=cfg.data.normalize_type if hasattr(cfg.data, 'normalize_type') else "default"
        )

        train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.train.num_workers, pin_memory=True, worker_init_fn=seed_worker)
        val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                             num_workers=cfg.train.num_workers, pin_memory=True, worker_init_fn=seed_worker)

        # 매 폴드마다 새로운 모델 초기화
        model = get_model(cfg, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
        criterion = get_loss()

        best_val_loss = float('inf')
        patience_counter = 0

        # 폴드별 결과 저장 디렉토리 생성
        fold_save_dir = os.path.join(cfg.train.save_dir, f"fold_{fold+1}")
        os.makedirs(fold_save_dir, exist_ok=True)
        
        # 각 폴드 내의 에포크별 메트릭을 저장할 리스트
        fold_epoch_metrics = []

        for epoch in range(1, cfg.train.epochs + 1):
            model.train()
            train_loss = 0
            train_dice = 0
            train_iou = 0
            train_start = time.time()
            
            # 훈련 루프
            for k, (x, y) in enumerate(tqdm(train_dl, desc=f"Fold {fold+1} Epoch {epoch}")):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x)
                if isinstance(output, tuple):
                    output = output[0]
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # 훈련 중 Dice, IoU는 배치 단위 평가
                d, i = evaluate(model, [(x,y)], device, cfg.data.threshold, vis=False)
                train_dice += d
                train_iou += i

            train_end = time.time(); train_elapsed = train_end - train_start
            avg_train_loss = train_loss / len(train_dl)
            avg_train_dice = train_dice / len(train_dl)
            avg_train_iou = train_iou / len(train_dl)

            model.eval()
            val_loss = 0
            val_start = time.time()
            with torch.no_grad():
                for x_val, y_val in val_dl:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    pred = model(x_val)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    loss = criterion(pred, y_val)
                    val_loss += loss.item()
            val_end = time.time(); val_elapsed = val_end - val_start
            avg_val_loss = val_loss / len(val_dl)

            # 전체 validation set에 대한 Dice, IoU 평가
            vd, vi = evaluate(model, val_dl, device, cfg.data.threshold, vis=False)

            print(f"Fold:{fold+1} Epoch:{epoch:03d} "
                  f"TRAIN ➜ Loss:{avg_train_loss:.4f} Dice:{avg_train_dice:.4f} IoU:{avg_train_iou:.4f} Time:{train_elapsed:.2f}s | "
                  f"VAL ➜ Loss:{avg_val_loss:.4f} Dice:{vd:.4f} IoU:{vi:.4f} Time:{val_elapsed:.2f}s")
            
            # 에포크별 메트릭 저장
            fold_epoch_metrics.append({
                'fold': fold + 1,
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'train_dice': avg_train_dice,
                'train_iou': avg_train_iou,
                'val_loss': avg_val_loss,
                'val_dice': vd,
                'val_iou': vi
            })

            # Early Stopping 및 Model Saving
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(fold_save_dir, "model_best.pth"))
                print(f"    Saved best model for Fold {fold+1} with validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= cfg.train.patience:
                    print(f"    Early stopping for Fold {fold+1} at epoch {epoch}")
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

        # 이 폴드의 검증 세트에 대한 최종 평가 (최적 모델 사용)
        val_dice_at_best, val_iou_at_best = evaluate(model, val_dl, device, cfg.data.threshold, vis=False)
        coverage_stats_val = compute_mask_coverage(model, val_dl, device, cfg.data.threshold)

        # best_val_loss는 Early Stopping에 사용된 로스 값 (저장된 모델의 기준)
        # val_dice_at_best, val_iou_at_best는 best_val_loss를 달성한 시점의 모델로 재평가한 값
        fold_best_metrics_entry = {
            'fold': fold + 1,
            'best_val_loss': best_val_loss,
            'best_model_val_dice': val_dice_at_best,
            'best_model_val_iou': val_iou_at_best,
            'val_mask_coverage_gt_total': coverage_stats_val['gt_total'].item(),
            'val_mask_coverage_pred_total': coverage_stats_val['pred_total'].item(),
            'val_mask_coverage_inter_total': coverage_stats_val['inter_total'].item(),
            'val_mask_coverage_ratio': coverage_stats_val['mask_coverage_ratio']
        }
        all_fold_best_metrics.append(fold_best_metrics_entry)
        
        print(f"Fold {fold+1} Validation Results (from best model) ➜ Dice:{val_dice_at_best:.4f} IoU:{val_iou_at_best:.4f}")
        for k, v in coverage_stats_val.items():
            if isinstance(v, torch.Tensor): v = v.item()
            print(f"  {k}: {v}")

        # 매 폴드마다 모델 가중치를 초기화했기 때문에, 다음 폴드 이전에 GPU 캐시를 비워줍니다.
        del model, optimizer, criterion
        gc.collect()
        torch.cuda.empty_cache()


    # --- 3. 모든 폴드 완료 후 최종 결과 집계 및 보고 ---
    print("\n--- All Folds Completed ---")
    df_best_fold_results = pd.DataFrame(all_fold_best_metrics)
    print("\nIndividual Fold Best Validation Results:")
    print(df_best_fold_results)

    avg_final_results = df_best_fold_results.mean(numeric_only=True).to_dict()
    print("\nAverage Best Validation Results Across Folds:")
    for k, v in avg_final_results.items():
        print(f"  {k}: {v:.4f}")

    # 최종 결과 저장 디렉토리 생성
    output_dir = cfg.train.save_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 상세 폴드별 최고 성능 결과 저장
    detailed_results_path = os.path.join(output_dir, "5_fold_cross_validation_best_val_results.csv")
    df_best_fold_results.to_csv(detailed_results_path, index=False)
    print(f"\nDetailed best validation results per fold saved to: {detailed_results_path}")

    # 평균 최종 결과 저장
    avg_results_df = pd.DataFrame([avg_final_results])
    avg_results_path = os.path.join(output_dir, "5_fold_cross_validation_average_results.csv")
    avg_results_df.to_csv(avg_results_path, index=False)
    print(f"Average results across folds saved to: {avg_results_path}")

    # --- 4. 최종 테스트 세트 평가 (선택 사항, K-Fold에 포함되지 않은 독립적인 세트) ---
    print("\n--- Independent Test Set Evaluation (Optional) ---")
    test_img_dir = os.path.join(cfg.data.root_dir, 'test', 'images')
    test_mask_dir = os.path.join(cfg.data.root_dir, 'test', 'masks')

    test_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.npy')])
    test_image_paths = [os.path.join(test_img_dir, f) for f in test_files]
    test_mask_paths = [os.path.join(test_mask_dir, f.replace('.npy', '.png')) for f in test_files]

    if test_image_paths:
        print("To get a final, unbiased test set performance, it's highly recommended to:")
        print("1. Train a new model from scratch using ALL available training data (i.e., data in train/ and val/ folders).")
        print("2. Evaluate this newly trained model on the independent test set (data in test/ folder) once.")
        print("The 5-fold CV results provide a robust estimate of generalization, but a final test set validates the chosen model.")
    else:
        print(f"No independent test set found in {test_img_dir}. Skipping final test evaluation.")

    total_elapsed = time.time() - start_time
    print(f"\nTotal cross-validation process time: {total_elapsed/60:.2f} minutes")

if __name__ == "__main__":
    cfg = OmegaConf.load('unet.yaml')
    main(cfg)