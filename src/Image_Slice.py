# Image_Slice.py

import os
import numpy as np
import cv2
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 엑셀 경로에 따라 조정 (예: /content/dataset_registry.xlsx)
df = pd.read_excel("src\dataset_registry.xlsx")

# 마스크 존재하는 데이터만 필터링 (Class A)
class_a_df = df[df["mask_file"].notna()].copy()
train_val_ids, test_ids = train_test_split(class_a_df, test_size=0.15, random_state=42)
train_ids, val_ids = train_test_split(train_val_ids, test_size=0.1765, random_state=42)

split_map = {
    "train": train_ids,
    "val": val_ids,
    "test": test_ids
}

# --- IMPORTANT: normalize_image 함수는 더 이상 사용하지 않습니다. 제거하거나 주석 처리하세요. ---
# def normalize_image(img):
#     img = np.clip(img, -1024, 1024)
#     img = (img + 1024) / 2048
#     img = (img * 255).astype(np.uint8)
#     return img

def save_case_slices(study_file, mask_file, split, case_id, output_root="output_dataset"):
    ct_vol = sitk.GetArrayFromImage(sitk.ReadImage(study_file))
    mask_vol = sitk.GetArrayFromImage(sitk.ReadImage(mask_file))
    
    for i in range(ct_vol.shape[0]):
        # ⭐ 변경된 부분: HU 값 클리핑만 수행하고, uint8 변환은 하지 않습니다.
        # 폐 CT COVID-19 병변에 적합한 HU 윈도우를 사용합니다.
        # -1024에서 1024로 클리핑하여 불필요한 극단값을 제거하고,
        # 폐 실질 및 병변에 관련된 HU 값 범위를 유지합니다.
        ct_slice = np.clip(ct_vol[i], -1024, 1024) 
        
        # ⭐ 클리핑 후 float32 타입 그대로 저장
        ct_slice = ct_slice.astype(np.float32) 
        
        # 마스크 슬라이스는 기존과 동일하게 0 또는 1 (uint8)로 처리
        mask_slice = (mask_vol[i] > 0).astype(np.uint8) 
        
        img_dir = os.path.join(output_root, split, "images")
        mask_dir = os.path.join(output_root, split, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        # 이미지 파일은 .npy로 저장 (float32 HU 값)
        filename_npy = f"{case_id}_slice_{i:03d}.npy"
        np.save(os.path.join(img_dir, filename_npy), ct_slice)
        
        # 마스크 파일은 .png로 저장 (0 또는 255)
        filename_png = f"{case_id}_slice_{i:03d}.png"
        cv2.imwrite(os.path.join(mask_dir, filename_png), mask_slice * 255) # 0 또는 255로 저장

# 실행 (이 부분은 기존과 동일)
for split, df_split in split_map.items():
    for idx, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"[{split}]"):
        study_file = "src"+row['study_file'].replace(".gz","")
        mask_file = "src"+row['mask_file'].replace(".gz","")
        case_id = f"study_{int(row['study_id']):04d}"
        save_case_slices(study_file, mask_file, split, case_id)