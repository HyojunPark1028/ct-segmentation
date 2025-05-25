# dataset.py

import os, cv2, torch, numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class NpySegDataset(Dataset):
    """Loads npy slice + png mask from full file paths provided by KFold."""
    def __init__(self, image_paths: list, mask_paths: list, augment=False, img_size=None, normalize_type="default"):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.normalize_type = normalize_type

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("Number of image files and mask files must be the same.")

        resize = [A.Resize(img_size, img_size)] if img_size else []
        
        # 베이스 증강 파이프라인 (크롭핑을 포함)
        base_aug_ops = resize + [ 
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            ToTensorV2(), # NumPy 배열을 PyTorch 텐서로 변환
        ]
        # 증강을 사용하지 않을 때의 파이프라인 (크롭핑 제외)
        no_aug_ops = resize + [ToTensorV2()]

        self.tf = A.Compose(base_aug_ops if augment else no_aug_ops)

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # ⭐ 확인: Image_Slice.py에서 float32로 저장되었으므로, 여기서도 float32로 로드
        img = np.load(img_path).astype(np.float32)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        
        # 마스크 값 정규화: 0 또는 1 (png 파일에서 0/255로 저장했으므로)
        mask = mask / 255.0 

        # ⭐ 기존 normalize_type: sam 로직은 그대로 유지 (이제 올바른 HU 값을 받게 됨)
        if self.normalize_type == "sam":
            # 이 로직은 [-1024, 1024] 범위의 float32 HU 값을 [0, 255] 범위로 스케일링합니다.
            img = img - img.min() # 최소값을 0으로 만듦 (예: -1024가 0으로)
            img = (img / (img.max() + 1e-8)) * 255.0 # 최대값을 255로 만듦
        else:
            img = (img + 1024) / 2048.0
            
        # Albumentations 적용을 위한 1채널 이미지 처리: (H, W) -> (H, W, 1)
        # Albumentations는 기본적으로 H, W, C 형태를 기대합니다.
        data = {"image": img, "mask": mask}
        transformed_data = self.tf(**data)
        
        image = transformed_data["image"] # 이미 ToTensorV2가 적용되어 (C, H, W) 형태
        mask = transformed_data["mask"]   # 이미 ToTensorV2가 적용되어 (C, H, W) 형태

        return image, mask