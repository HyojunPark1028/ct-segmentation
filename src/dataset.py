# dataset.py (수정된 NpySegDataset 정의)

import os, cv2, torch, numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class NpySegDataset(Dataset): # 이름을 유지하거나 CrossFoldNpySegDataset으로 변경 가능
    """Loads npy slice + png mask from full file paths provided by KFold."""
    def __init__(self, image_paths: list, mask_paths: list, augment=False, img_size=None, normalize_type="default"):
        # image_paths: 이미지 파일의 전체 경로 리스트 (예: 'data/train/images/image_001.npy')
        # mask_paths: 마스크 파일의 전체 경로 리스트 (예: 'data/train/masks/image_001.png')
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.normalize_type = normalize_type

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("Number of image files and mask files must be the same.")

        resize = [A.Resize(img_size, img_size)] if img_size else []
        base_aug_ops = resize + [
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            ToTensorV2(),
        ]
        no_aug_ops = resize + [ToTensorV2()]

        self.tf = A.Compose(base_aug_ops if augment else no_aug_ops)

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img = np.load(img_path).astype(np.float32)

        if self.normalize_type == "sam":
            img = img - img.min()
            img = (img / (img.max() + 1e-8)) * 255.0
        elif self.normalize_type == "default":
            img = img / 255.0

        msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        msk = (msk > 127).astype(np.float32)

        img, msk = img[...,None], msk[...,None] # Add channel dimension for Albumentations

        out = self.tf(image=img, mask=msk)
        return out['image'], out['mask']