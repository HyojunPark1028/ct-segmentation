import os
import numpy as np
import cv2
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 엑셀 경로에 따라 조정 (예: /content/dataset_registry.xlsx)
df = pd.read_excel("dataset_registry.xlsx")

# 마스크 존재하는 데이터만 필터링 (Class A)
class_a_df = df[df["mask_file"].notna()].copy()
train_val_ids, test_ids = train_test_split(class_a_df, test_size=0.15, random_state=42)
train_ids, val_ids = train_test_split(train_val_ids, test_size=0.1765, random_state=42)

split_map = {
    "train": train_ids,
    "val": val_ids,
    "test": test_ids
}

def normalize_image(img):
    img = np.clip(img, -1024, 1024)
    img = (img + 1024) / 2048
    img = (img * 255).astype(np.uint8)
    return img

def save_case_slices(study_file, mask_file, split, case_id, output_root="output_dataset"):
    ct_vol = sitk.GetArrayFromImage(sitk.ReadImage(study_file))
    mask_vol = sitk.GetArrayFromImage(sitk.ReadImage(mask_file))
    for i in range(ct_vol.shape[0]):
        ct_slice = normalize_image(ct_vol[i])
        mask_slice = (mask_vol[i] > 0).astype(np.uint8) * 255
        img_dir = os.path.join(output_root, split, "images")
        mask_dir = os.path.join(output_root, split, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        filename = f"{case_id}_slice_{i:03d}.png"
        cv2.imwrite(os.path.join(img_dir, filename), ct_slice)
        cv2.imwrite(os.path.join(mask_dir, filename), mask_slice)

# 실행
for split, df_split in split_map.items():
    for idx, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"[{split}]"):
        study_file = "."+row['study_file'].replace(".gz","")
        mask_file = "."+row['mask_file'].replace(".gz","")
        case_id = f"study_{int(row['study_id']):04d}"
        save_case_slices(study_file, mask_file, split, case_id)
