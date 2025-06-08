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
import glob # 데이터 경로 탐색에 여전히 유용하므로 유지

# 모델 임포트
from .models.medsam_gan import MedSAM_GAN

from .dataset import NpySegDataset
import cv2 # dataset.py에서 사용되므로 임포트 유지

# GAN Loss 함수 및 평가 스크립트 임포트
from .losses_GAN import get_segmentation_loss, get_discriminator_loss, get_generator_adversarial_loss
from .evaluate import evaluate, compute_mask_coverage

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

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    seg_criterion: nn.Module,
    adv_criterion_D: nn.Module,
    adv_criterion_G: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int,
    gan_lambda_adv: float = 0.1
) -> tuple[float, float, float, float]:
    """
    GAN 모델의 한 에폭 훈련을 수행합니다.
    Generator와 Discriminator를 번갈아 학습시킵니다.
    """
    model.train() # 모델을 학습 모드로 설정
    # tqdm은 훈련 진행 상황을 시각적으로 보여주는 프로그레스 바입니다.
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    
    total_seg_loss = 0.0
    total_g_adv_loss = 0.0
    total_g_loss = 0.0
    total_d_loss = 0.0

    # 각 배치에서 이미지와 마스크를 가져옵니다.
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device) # 원본 GT 마스크 (H, W, 단일 채널)

        # Discriminator에 입력할 저해상도 GT 마스크를 준비합니다 (예: 256x256).
        # 이 마스크는 이진 마스크이므로 'nearest' 보간법을 사용합니다.
        real_low_res_masks = F.interpolate(
            masks, size=(256, 256), mode='nearest'
        ).float()


        # --- 1. Discriminator (D) 학습 단계 ---
        # Discriminator 옵티마이저의 그래디언트를 0으로 초기화합니다.
        optimizer_D.zero_grad()

        # Generator의 출력을 얻습니다. 이 단계에서는 Generator의 파라미터를 업데이트하지 않으므로
        # torch.no_grad() 블록을 사용하여 그래디언트 계산을 비활성화합니다.
        with torch.no_grad():
            # MedSAM_GAN 모델의 forward는 real_low_res_mask가 None일 때,
            # 생성된 마스크, IoU 예측, 생성된 마스크에 대한 D의 출력을 반환합니다.
            gen_masks, _, _ = model(images, None)
        
        # Discriminator에 실제 마스크와 Generator가 생성한 (가짜) 마스크를 모두 입력하여
        # 각 마스크에 대한 Discriminator의 판별 결과를 얻습니다.
        # `model.forward()`는 real_low_res_mask가 제공되면, 4번째 반환 값으로
        # 실제 마스크에 대한 Discriminator의 출력을 추가로 반환합니다.
        _, _, disc_output_gen_for_D, disc_output_real = model(images, real_low_res_masks)

        # Discriminator 손실을 계산합니다. Discriminator는 진짜는 1로, 가짜는 0으로 예측해야 합니다.
        d_loss = adv_criterion_D(disc_output_real, disc_output_gen_for_D)
        # 손실에 대한 역전파를 수행하여 그래디언트를 계산합니다.
        d_loss.backward()
        # Discriminator 옵티마이저를 사용하여 파라미터를 업데이트합니다.
        optimizer_D.step()

        # 총 Discriminator 손실에 현재 배치의 손실을 더합니다.
        total_d_loss += d_loss.item()


        # --- 2. Generator (G) 학습 단계 ---
        # Generator 옵티마이저의 그래디언트를 0으로 초기화합니다.
        optimizer_G.zero_grad()

        # Generator를 통해 마스크를 다시 생성합니다. 이번에는 Generator의 파라미터를 업데이트하기 위함입니다.
        # Discriminator는 G 학습 시에는 고정(eval 모드)되거나 그래디언트 계산이 비활성화됩니다.
        # MedSAM_GAN의 forward는 항상 G 파라미터를 계산하므로, D 파라미터 업데이트를 막는 것이 중요합니다.
        gen_masks, iou_predictions, disc_output_gen_for_G = model(images, None)

        # Segmentation Loss를 계산합니다. (생성된 마스크와 실제 마스크 간의 유사도)
        seg_loss = seg_criterion(gen_masks, masks)
        total_seg_loss += seg_loss.item()

        # Generator Adversarial Loss를 계산합니다.
        # Generator는 Discriminator를 속여 생성된 마스크가 '진짜'라고 예측하게 만들어야 합니다.
        g_adv_loss = adv_criterion_G(disc_output_gen_for_G)
        total_g_adv_loss += g_adv_loss.item()

        # Generator의 총 손실은 Segmentation Loss와 Adversarial Loss를 가중치(gan_lambda_adv)로 합산한 값입니다.
        g_loss = seg_loss + gan_lambda_adv * g_adv_loss
        # 총 손실에 대한 역전파를 수행합니다.
        g_loss.backward()
        # Generator 옵티마이저를 사용하여 파라미터를 업데이트합니다.
        optimizer_G.step()
        
        total_g_loss += g_loss.item()

        # 프로그레스 바에 현재 배치의 손실 값들을 표시합니다.
        pbar.set_postfix({
            "D_Loss": f"{d_loss.item():.4f}",
            "G_Seg_Loss": f"{seg_loss.item():.4f}",
            "G_Adv_Loss": f"{g_adv_loss.item():.4f}",
            "G_Total_Loss": f"{g_loss.item():.4f}"
        })

    # 에폭의 평균 손실 값들을 계산하여 반환합니다.
    avg_seg_loss = total_seg_loss / len(dataloader)
    avg_g_adv_loss = total_g_adv_loss / len(dataloader)
    avg_g_loss = total_g_loss / len(dataloader)
    avg_d_loss = total_d_loss / len(dataloader)

    return avg_g_loss, avg_d_loss, avg_seg_loss, avg_g_adv_loss

def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    seg_criterion: nn.Module,
    device: torch.device
) -> tuple[float, dict]:
    """
    모델의 한 에폭 검증을 수행하고 Segmentation 손실 및 성능 지표를 반환합니다.
    """
    model.eval() # 모델을 평가 모드로 설정 (드롭아웃, 배치 정규화 비활성화 등)
    pbar = tqdm(dataloader, desc="Validation") # 검증 진행률 바
    
    total_seg_loss = 0.0 # 총 Segmentation 손실 추적
    
    # 배치당 추론 시간 측정을 위한 리스트 초기화
    val_inference_times = []

    with torch.no_grad(): # 그래디언트 계산 비활성화 (평가 시 메모리 절약 및 속도 향상)
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)

            # GPU 작업이 완료될 때까지 기다려 정확한 추론 시간 측정
            torch.cuda.synchronize() 
            start_inference = time.time()

            # 모델의 예측을 수행합니다. (GAN의 Discriminator 출력은 검증 시 필요 없음)
            predicted_masks, _, _ = model(images, None)
            
            torch.cuda.synchronize() # GPU 작업 동기화
            end_inference = time.time()
            val_inference_times.append(end_inference - start_inference)
            
            # Segmentation Loss 계산
            loss = seg_criterion(predicted_masks, masks)
            total_seg_loss += loss.item()
            # 프로그레스 바에 현재 배치의 Segmentation Loss를 표시합니다.
            pbar.set_postfix({"Seg_Loss": f"{loss.item():.4f}"})
            
    # 에폭의 평균 Segmentation Loss를 계산합니다.
    avg_seg_loss = total_seg_loss / len(dataloader)
    
    # evaluate 함수를 호출하여 Dice, IoU 등의 정량적 지표를 계산합니다.
    # evaluate 함수는 (dice_score, iou_score) 튜플을 반환하는 것으로 가정합니다.
    # 여기서는 threshold, vis 인자는 evaluate.py의 evaluate 함수 정의에 따라 그대로 전달합니다.
    # 단, train_GAN.py의 cfg.data.threshold를 사용합니다.
    vd, vi = evaluate(model, dataloader, device, thr=cfg.data.threshold) # cfg.data.threshold 인자 전달
    
    # 계산된 지표들을 딕셔너리 형태로 묶어 반환합니다.
    metrics = {
        'dice_score': vd,
        'iou_score': vi,
        'val_inference_time_per_batch_sec': np.mean(val_inference_times) if val_inference_times else 0.0
    }
    
    return avg_seg_loss, metrics # 평균 Segmentation 손실과 지표 딕셔너리를 반환

def run_training_pipeline(cfg: OmegaConf):
    """
    K-Fold Cross-Validation을 사용하여 GAN 모델을 훈련하고 평가하는 전체 파이프라인을 실행합니다.
    """
    start_time = time.time() # 전체 파이프라인 시작 시간 기록
    seed_everything(cfg.seed) # 재현 가능한 결과를 위해 시드 설정
    # 학습에 사용할 장치 (GPU 또는 CPU)를 설정합니다.
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() and cfg.gpu is not None else "cpu")
    print(f"Using device: {device}")

    # 현재 시간을 기반으로 출력 디렉토리 이름을 생성하고 생성합니다.
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.output_dir, f"{cfg.model.name}_{current_time}")
    os.makedirs(output_dir, exist_ok=True)
    # 사용된 설정 파일을 출력 디렉토리에 저장합니다.
    OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))


    # U-Net 체크포인트 파일의 존재 여부를 확인합니다.
    if not os.path.exists(cfg.model.unet_checkpoint):
        raise FileNotFoundError(f"U-Net checkpoint not found at: {cfg.model.unet_checkpoint}")
    
    # --- 1. K-Fold Cross-Validation을 위한 데이터 준비 ---
    # K-Fold를 위해 'train'과 'val' 폴더의 모든 데이터를 합쳐서 사용합니다.
    all_image_full_paths = []
    all_mask_full_paths = []

    # 'train' 폴더의 이미지 및 마스크 파일 목록을 추가합니다.
    train_img_base = os.path.join(cfg.data.data_dir, 'train', 'images')
    train_mask_base = os.path.join(cfg.data.data_dir, 'train', 'masks')
    if os.path.exists(train_img_base): # 'images' 폴더가 존재하는지 확인
        train_files = sorted([f for f in os.listdir(train_img_base) if f.endswith('.npy')])
        for f in train_files:
            all_image_full_paths.append(os.path.join(train_img_base, f))
            # 마스크 파일은 .npy 확장자를 .png로 대체하여 매칭합니다.
            all_mask_full_paths.append(os.path.join(train_mask_base, f.replace('.npy','.png')))
    
    # 'val' 폴더의 이미지 및 마스크 파일 목록을 추가합니다.
    val_img_base = os.path.join(cfg.data.data_dir, 'val', 'images')
    val_mask_base = os.path.join(cfg.data.data_dir, 'val', 'masks')
    if os.path.exists(val_img_base): # 'images' 폴더가 존재하는지 확인
        val_files = sorted([f for f in os.listdir(val_img_base) if f.endswith('.npy')])
        for f in val_files:
            all_image_full_paths.append(os.path.join(val_img_base, f))
            all_mask_full_paths.append(os.path.join(val_mask_base, f.replace('.npy','.png')))

    # 전체 이미지 경로가 없거나 이미지와 마스크 파일 수가 일치하지 않으면 오류를 발생시킵니다.
    if not all_image_full_paths:
        raise FileNotFoundError(f"No .npy files found in {train_img_base} or {val_img_base}. Please check your data path.")
    if len(all_image_full_paths) != len(all_mask_full_paths):
        raise ValueError("Mismatch between image and mask file counts for K-Fold data. Ensure every image has a corresponding mask.")

    # 데이터 로딩 상태를 디버그 메시지로 출력합니다.
    print(f"DEBUG: Data Root Dir: {cfg.data.data_dir}")
    print(f"DEBUG: Found {len(all_image_full_paths)} total images for K-Fold training/validation.")
    print(f"DEBUG: Found {len(all_mask_full_paths)} total masks for K-Fold training/validation.")
    if len(all_image_full_paths) > 0:
        print(f"DEBUG: First image path: {all_image_full_paths[0]}")
    if len(all_mask_full_paths) > 0:
        print(f"DEBUG: First mask path: {all_mask_full_paths[0]}")


    # KFold 객체를 설정합니다. (n_splits는 config에서 가져옵니다)
    kf = KFold(n_splits=cfg.kfold.n_splits, shuffle=True, random_state=cfg.seed)

    all_fold_best_metrics = [] # 각 K-Fold의 '최고' 성능 지표를 저장할 리스트
    
    # 모델의 총 학습 가능한 파라미터 수를 저장할 변수 (첫 번째 폴드에서 한 번만 계산)
    total_trainable_parameters = 0
    
    # 데이터 정규화 타입을 config에서 가져옵니다.
    normalize_type = cfg.data.get("normalize_type", "default")

    # --- 2. K-Fold 루프 시작 ---
    for fold, (train_index, val_index) in enumerate(kf.split(all_image_full_paths)):
        print(f"\n--- Starting Fold {fold + 1}/{cfg.kfold.n_splits} ---")

        # 현재 폴드에 해당하는 훈련 및 검증 파일의 전체 경로 리스트를 생성합니다.
        fold_train_image_paths = [all_image_full_paths[i] for i in train_index]
        fold_train_mask_paths = [all_mask_full_paths[i] for i in train_index]
        fold_val_image_paths = [all_image_full_paths[i] for i in val_index]
        fold_val_mask_paths = [all_mask_full_paths[i] for i in val_index]

        # 훈련 및 검증 Dataset 객체를 생성합니다.
        train_ds = NpySegDataset(
            image_paths=fold_train_image_paths,
            mask_paths=fold_train_mask_paths,
            augment=cfg.data.augment, # 데이터 증강 여부
            img_size=cfg.data.image_size, # 이미지 크기
            normalize_type=normalize_type # 정규화 타입
        )
        val_ds = NpySegDataset(
            image_paths=fold_val_image_paths,
            mask_paths=fold_val_mask_paths,
            augment=False, # 검증 세트는 증강하지 않습니다.
            img_size=cfg.data.image_size,
            normalize_type=normalize_type
        )

        # DataLoader에 사용할 Generator를 설정하여 재현성을 확보합니다.
        g = torch.Generator()
        g.manual_seed(cfg.seed + fold)

        # 훈련 및 검증 DataLoader를 생성합니다.
        train_dl = DataLoader(train_ds, batch_size=cfg.dataloader.batch_size, shuffle=True,
                              num_workers=cfg.dataloader.num_workers, pin_memory=True, generator=g)
        val_dl = DataLoader(val_ds, batch_size=cfg.dataloader.batch_size, shuffle=False,
                            num_workers=cfg.dataloader.num_workers, pin_memory=True)

        # 매 폴드마다 새로운 MedSAM_GAN 모델 인스턴스를 초기화합니다.
        model = MedSAM_GAN(
            sam_checkpoint=cfg.model.sam_checkpoint,
            unet_checkpoint=cfg.model.unet_checkpoint,
            out_channels=cfg.model.out_channels
        ).to(device)

        # 첫 번째 폴드에서만 모델의 총 학습 가능한 파라미터 수를 계산하여 출력합니다.
        if fold == 0:
            total_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model has {total_trainable_parameters:,} trainable parameters.")

        # Generator와 Discriminator를 위한 옵티마이저를 각각 설정합니다.
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
        
        # 손실 함수들을 초기화하고 장치로 이동시킵니다.
        seg_criterion = get_segmentation_loss().to(device)
        adv_criterion_D = get_discriminator_loss().to(device)
        adv_criterion_G = get_generator_adversarial_loss().to(device)

        # Early Stopping을 위한 변수 초기화 (Dice Score 기준, 높을수록 좋음)
        best_val_dice = -float('inf')
        patience_counter = 0
        
        # 현재 폴드의 에포크별 메트릭을 저장할 리스트
        fold_epoch_metrics = []

        # 각 에포크 훈련 루프
        for epoch in range(1, cfg.epochs + 1):
            epoch_start_time = time.time() # 에폭 시작 시간 기록
            # 훈련 함수 호출
            train_g_loss, train_d_loss, train_seg_loss, train_g_adv_loss = train_one_epoch(
                model, train_dl, optimizer_G, optimizer_D,
                seg_criterion, adv_criterion_D, adv_criterion_G,
                device, epoch, cfg.log_interval, cfg.gan_lambda_adv
            )
            epoch_train_time_sec = time.time() - epoch_start_time # 에폭 훈련 시간 계산
            
            # 검증 함수 호출 (손실과 지표 딕셔너리 반환)
            val_seg_loss, val_metrics_dict = validate_one_epoch(model, val_dl, seg_criterion, device)
            
            # val_metrics_dict에서 추론 시간 가져오기 (없으면 0.0)
            val_inference_time_per_batch_sec = val_metrics_dict.get('val_inference_time_per_batch_sec', 0.0)

            # 현재 에폭의 모든 로그 데이터를 딕셔너리로 구성
            epoch_log = {
                'fold': fold + 1,
                'epoch': epoch,
                'train_g_total_loss': train_g_loss,
                'train_d_loss': train_d_loss,
                'train_seg_loss': train_seg_loss,
                'train_g_adv_loss': train_g_adv_loss,
                'train_time_sec': epoch_train_time_sec, # 실제 측정된 훈련 시간
                'val_seg_loss': val_seg_loss,
                'val_dice': val_metrics_dict.get('dice_score', -1.0), # evaluate에서 반환되는 'dice_score' 키 사용
                'val_iou': val_metrics_dict.get('iou_score', -1.0),   # evaluate에서 반환되는 'iou_score' 키 사용
                'val_inference_time_per_batch_sec': val_inference_time_per_batch_sec,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            fold_epoch_metrics.append(epoch_log) # 현재 폴드의 에포크별 메트릭 리스트에 추가
            print(epoch_log) # 콘솔에 에포크 로그 딕셔너리 출력

            # Early Stopping 및 Best Model 저장 로직 (val_dice 기준으로 변경)
            current_val_dice = val_metrics_dict.get('dice_score', 0.0)
            if current_val_dice > best_val_dice: # Dice Score는 높을수록 좋으므로 > 사용
                best_val_dice = current_val_dice
                # 모델 저장 (Generator와 Discriminator의 state_dict를 각각 저장)
                torch.save({
                    'epoch': epoch,
                    'model_G_state_dict': model.sam.state_dict(),
                    'model_D_state_dict': model.discriminator.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                    'best_val_dice': best_val_dice,
                    'cfg': cfg
                }, os.path.join(output_dir, f"fold_{fold+1}", "model_best.pth")) # 동적으로 생성된 output_dir 사용
                patience_counter = 0 # 성능 개선 시 카운터 초기화
                print(f"    Saved best model for Fold {fold+1} with validation dice: {best_val_dice:.4f}")
            else:
                patience_counter += 1 # 성능 개선 없을 시 카운터 증가
                print(f"No improvement in val_dice. EarlyStopping counter: {patience_counter}/{cfg.early_stopping_patience}")
                if patience_counter >= cfg.early_stopping_patience: # 설정된 인내 횟수를 초과하면 Early Stopping
                    print(f"Early stopping triggered for Fold {fold+1}.")
                    break
            
            gc.collect() # 가비지 컬렉션 수행
            torch.cuda.empty_cache() # CUDA 캐시 비우기

        # --- Fold 학습 완료 후 처리 ---
        # 폴드별 에포크 메트릭을 CSV 파일로 저장
        if fold_epoch_metrics:
            df_fold_epochs = pd.DataFrame(fold_epoch_metrics)
            fold_epoch_metrics_path = os.path.join(output_dir, f"fold_{fold+1}", "epoch_metrics.csv") # 폴드별 서브디렉토리에 저장
            df_fold_epochs.to_csv(fold_epoch_metrics_path, index=False)
            print(f"    Epoch-wise metrics for Fold {fold+1} saved to: {fold_epoch_metrics_path}")

        print(f"\n--- Fold {fold + 1} Training Finished ---")
        best_model_path_this_fold = os.path.join(output_dir, f"fold_{fold+1}", "model_best.pth") # 최적 모델 경로

        # 현재 폴드의 최적 모델을 로드하여 최종 평가를 수행합니다.
        if os.path.exists(best_model_path_this_fold):
            # MedSAM_GAN 객체를 새로 만들고 Generator와 Discriminator의 state_dict를 로드
            model_eval = MedSAM_GAN(
                sam_checkpoint=cfg.model.sam_checkpoint,
                unet_checkpoint=cfg.model.unet_checkpoint,
                out_channels=cfg.model.out_channels
            ).to(device)
            checkpoint = torch.load(best_model_path_this_fold, map_location=device)
            model_eval.sam.load_state_dict(checkpoint['model_G_state_dict'])
            model_eval.discriminator.load_state_dict(checkpoint['model_D_state_dict'])
            model_eval.eval() # 평가 모드 설정
        else:
            print(f"Warning: No best model saved for Fold {fold+1}. Cannot perform final fold evaluation. Using last trained model state.")
            model_eval = model # 최적 모델이 없으면 마지막 학습된 모델 사용
        
        # 검증 데이터셋에 대한 최종 지표 계산
        final_fold_metrics = evaluate(model_eval, val_dl, device, thr=cfg.data.threshold) # evaluate 함수 호출
        # 마스크 커버리지 통계 계산
        coverage_stats_val = compute_mask_coverage(model_eval, val_dl, device, cfg.data.threshold)

        # 폴드별 최고 성능 지표 엔트리를 구성하여 전체 리스트에 추가합니다.
        fold_best_metrics_entry = {
            'fold': fold + 1,
            'best_val_dice': final_fold_metrics.get('dice_score', -1.0),
            'best_val_loss': val_seg_loss, # 마지막 에폭의 val_seg_loss (또는 필요시 best_val_loss를 따로 추적)
            'best_model_val_iou': final_fold_metrics.get('iou_score', -1.0),
            'val_mask_coverage_gt_pixels': coverage_stats_val['gt_pixels'].item() if isinstance(coverage_stats_val['gt_pixels'], torch.Tensor) else coverage_stats_val['gt_pixels'],
            'val_mask_coverage_pred_pixels': coverage_stats_val['pred_pixels'].item() if isinstance(coverage_stats_val['pred_pixels'], torch.Tensor) else coverage_stats_val['pred_pixels'],
            'val_mask_coverage_intersection': coverage_stats_val['intersection'].item() if isinstance(coverage_stats_val['intersection'], torch.Tensor) else coverage_stats_val['intersection'],
            'val_mask_coverage_coverage': coverage_stats_val['coverage'],
            'val_mask_coverage_overpredict': coverage_stats_val['overpredict'],
            'val_inference_time_per_batch_sec': val_metrics_dict.get('val_inference_time_per_batch_sec', 0.0) # 해당 폴드 베스트 모델의 인퍼런스 시간
        }
        all_fold_best_metrics.append(fold_best_metrics_entry)
        
        # 콘솔에 현재 폴드의 최종 검증 결과를 출력합니다.
        print(f"Fold {fold+1} Validation Results (from best model) ➜ Dice:{final_fold_metrics.get('dice_score', -1.0):.4f} IoU:{final_fold_metrics.get('iou_score', -1.0):.4f}")
        for k, v in coverage_stats_val.items():
            if isinstance(v, torch.Tensor): v = v.item()
            print(f"    {k}: {v}")

        # 평가용 모델 삭제 및 메모리 정리
        del model_eval
        gc.collect()
        torch.cuda.empty_cache()


    # --- 3. 모든 폴드 완료 후 최종 결과 집계 및 보고 ---
    print("\n--- All Folds Completed ---")
    df_best_fold_results = pd.DataFrame(all_fold_best_metrics)
    print("\nIndividual Fold Best Validation Results:")
    print(df_best_fold_results)

    # 모든 폴드의 평균 결과 계산
    avg_final_results = df_best_fold_results.mean(numeric_only=True).to_dict()
    # 총 학습 가능한 파라미터 수를 평균 결과에 추가합니다.
    avg_final_results['total_trainable_parameters'] = total_trainable_parameters

    print("\nAverage Best Validation Results Across Folds:")
    for k, v in avg_final_results.items():
        if k == 'total_trainable_parameters':
            print(f"    {k}: {int(v):,}") # 정수 형태로 포맷팅
        else:
            print(f"    {k}: {v:.4f}") # 소수점 4자리까지 포맷팅

    # 최종 결과 저장 디렉토리를 설정하고 생성합니다.
    output_dir_final = os.path.join(cfg.output_dir) # 최상위 outputs 폴더
    os.makedirs(output_dir_final, exist_ok=True)
    
    # 상세 폴드별 최고 성능 결과를 CSV 파일로 저장합니다.
    detailed_results_path = os.path.join(output_dir_final, f"{cfg.model.name}_k_fold_cross_validation_best_val_results_{current_time}.csv")
    df_best_fold_results.to_csv(detailed_results_path, index=False)
    print(f"\nDetailed best validation results per fold saved to: {detailed_results_path}")

    # 모든 폴드의 평균 결과를 CSV 파일로 저장합니다.
    avg_results_df = pd.DataFrame([avg_final_results])
    avg_results_path = os.path.join(output_dir_final, f"{cfg.model.name}_k_fold_cross_validation_average_results_{current_time}.csv")
    avg_results_df.to_csv(avg_results_path, index=False)
    print(f"Average results across folds saved to: {avg_results_path}")

    # --- 4. 최종 테스트 세트 평가 (K-Fold에 포함되지 않은 독립적인 세트) ---
    print("\n--- Independent Test Set Evaluation ---")
    # 'test' 폴더의 이미지 및 마스크 파일 경로를 설정합니다.
    test_img_base = os.path.join(cfg.data.data_dir, 'test', 'images')
    test_mask_base = os.path.join(cfg.data.data_dir, 'test', 'masks')

    # 테스트 이미지 및 마스크 파일 목록을 가져옵니다.
    test_files = sorted([f for f in os.listdir(test_img_base) if f.endswith('.npy')])
    test_image_paths = [os.path.join(test_img_base, f) for f in test_files]
    test_mask_paths = [os.path.join(test_mask_base, f.replace('.npy', '.png')) for f in test_files]

    # 유효한 테스트 파일이 있고 개수가 일치할 때만 최종 테스트를 수행합니다.
    if test_image_paths and len(test_image_paths) == len(test_mask_paths):
        # 최종 평가를 위한 MedSAM_GAN 모델을 초기화합니다.
        final_model = MedSAM_GAN(
            sam_checkpoint=cfg.model.sam_checkpoint,
            unet_checkpoint=cfg.model.unet_checkpoint,
            out_channels=cfg.model.out_channels
        ).to(device)

        # K-Fold 결과에서 가장 높은 'best_val_dice'를 기록한 폴드의 모델을 로드합니다.
        if not df_best_fold_results.empty:
            best_fold_row = df_best_fold_results.loc[df_best_fold_results['best_val_dice'].idxmax()]
            best_fold_num = int(best_fold_row['fold'])
            # 모델 저장 경로를 동적으로 구성합니다.
            best_model_path_for_test = os.path.join(output_dir, f"fold_{best_fold_num}", "model_best.pth")
            
            if os.path.exists(best_model_path_for_test):
                # Generator와 Discriminator의 state_dict를 각각 로드합니다.
                checkpoint = torch.load(best_model_path_for_test, map_location=device)
                final_model.sam.load_state_dict(checkpoint['model_G_state_dict'])
                final_model.discriminator.load_state_dict(checkpoint['model_D_state_dict'])
                print(f"Loaded best performing model from Fold {best_fold_num} ({best_model_path_for_test}) for final test evaluation.")
            else:
                print(f"Warning: Best model for Fold {best_fold_num} not found at {best_model_path_for_test}. Using newly initialized model for test evaluation. This may lead to poor test results.")
        else:
            print("No K-Fold results found to select the best model. Using newly initialized model for test evaluation. This may lead to poor test results.")

        final_model.eval() # 최종 모델을 평가 모드로 설정
        
        # 테스트 Dataset 및 DataLoader를 생성합니다.
        test_ds = NpySegDataset(
            image_paths=test_image_paths,
            mask_paths=test_mask_paths,
            augment=False, # 테스트 세트는 증강하지 않습니다.
            img_size=cfg.data.image_size,
            normalize_type=normalize_type
        )
        ts_dl = DataLoader(test_ds, batch_size=cfg.dataloader.batch_size, shuffle=False, num_workers=cfg.dataloader.num_workers, pin_memory=True)

        test_loss_sum = 0.0
        test_inference_times = []
        
        # 테스트 세트에 대한 추론 및 손실 계산
        with torch.no_grad():
            for x_test, y_test in tqdm(ts_dl, desc="Evaluating Test Set"):
                x_test, y_test = x_test.to(device), y_test.to(device)
                
                torch.cuda.synchronize() # GPU 작업 동기화
                start_inference = time.time()
                # MedSAM_GAN은 (마스크, IoU 예측, Discriminator 출력) 튜플을 반환
                pred, _, _ = final_model(x_test, None) # GAN의 D 출력은 테스트 시 불필요
                torch.cuda.synchronize() # GPU 작업 동기화
                end_inference = time.time()
                test_inference_times.append(end_inference - start_inference)

                if isinstance(pred, tuple): pred = pred[0] # 튜플인 경우 메인 마스크만 사용
                
                # 테스트 손실 계산 (Segmentation Loss)
                loss_val = seg_criterion(pred, y_test) # seg_criterion 사용
                test_loss_sum += loss_val.item()
        
        # 평균 추론 시간 및 테스트 손실 계산
        avg_test_inference_time_per_batch = np.mean(test_inference_times) if test_inference_times else 0.0
        avg_test_loss = test_loss_sum / len(ts_dl) if len(ts_dl) > 0 else 0.0

        # evaluate 함수를 사용하여 Dice, IoU 등의 지표 계산
        # evaluate.py의 evaluate 함수는 thr 인자를 받습니다.
        td_ti_metrics = evaluate(final_model, ts_dl, device, thr=cfg.data.threshold)
        td = td_ti_metrics.get('dice_score', -1.0) # Dice Score 가져오기
        ti = td_ti_metrics.get('iou_score', -1.0)   # IoU Score 가져오기

        # 최종 테스트 결과 콘솔 출력
        print(f"FINAL TEST ➜ Dice:{td:.4f} IoU:{ti:.4f} Test Loss:{avg_test_loss:.4f} "
              f"Avg Inference Time per Batch:{avg_test_inference_time_per_batch:.6f}s")

        # 마스크 커버리지 통계 계산 및 출력
        coverage_stats = compute_mask_coverage(final_model, ts_dl, device, cfg.data.threshold)
        for k, v in coverage_stats.items():
            if isinstance(v, torch.Tensor): v = v.item()
            print(f"    Test Mask {k}: {v}")

        # 테스트 커버리지 통계를 CSV 파일로 저장
        pd.DataFrame([coverage_stats]).to_csv(os.path.join(output_dir, "test_coverage.csv"), index=False)

        # 최종 테스트 결과를 딕셔너리로 구성하여 CSV 파일로 저장
        test_result = {
            "test_dice": td,
            "test_iou": ti,
            "test_loss": round(avg_test_loss, 4),
            "test_inference_time_per_batch_sec": round(avg_test_inference_time_per_batch, 6),
            "param_count": total_trainable_parameters, # K-Fold에서 계산된 총 파라미터 수 포함
            "gt_total": coverage_stats['gt_pixels'].item() if isinstance(coverage_stats['gt_pixels'], torch.Tensor) else coverage_stats['gt_pixels'],
            "pred_total": coverage_stats['pred_pixels'].item() if isinstance(coverage_stats['pred_pixels'], torch.Tensor) else coverage_stats['pred_pixels'],
            "inter_total": coverage_stats['intersection'].item() if isinstance(coverage_stats['intersection'], torch.Tensor) else coverage_stats['intersection'],
            "mask_coverage_ratio": coverage_stats['coverage']
        }
        pd.DataFrame([test_result]).to_csv(os.path.join(output_dir, "final_test_result.csv"), index=False)
        print(f"Final test results saved to: {os.path.join(output_dir, 'final_test_result.csv')}")

    else:
        print(f"No independent test set found or mismatch in {test_img_base}. Skipping final test evaluation.")

    total_elapsed = time.time() - start_time
    print(f"\nTotal cross-validation and test process time: {total_elapsed/60:.2f} minutes")

