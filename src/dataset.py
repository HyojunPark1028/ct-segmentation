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

        # ⭐ 기존 normalize_type: sam 로직은 그대로 유지 (이제 올바른 HU 값을 받게 됨)
        if self.normalize_type == "sam":
            # 이 로직은 [-1024, 1024] 범위의 float32 HU 값을 [0, 255] 범위로 스케일링합니다.
            img = img - img.min() # 최소값을 0으로 만듦 (예: -1024가 0으로)
            img = (img / (img.max() + 1e-8)) * 255.0 # 최대값을 255로 만듦
        else:
            img = (img + 1024) / 2048.0

        msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        msk = (msk > 127).astype(np.float32)
        
        print(f"DEBUG: Image path: {img_path}")
        print(f"DEBUG: Mask path: {mask_path}")
        print(f"DEBUG: Loaded Image shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
        print(f"DEBUG: Loaded Mask shape: {msk.shape}, dtype: {msk.dtype}, unique values: {np.unique(msk)}")

        # 이미지와 마스크를 시각화하여 실제 데이터를 확인 (주피터 노트북에서 실행 시 유용)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img[0] if img.ndim == 3 else img, cmap='gray') # 3D인 경우 첫 채널 시각화
        plt.title('Loaded Image')
        plt.subplot(1, 2, 2)
        plt.imshow(msk, cmap='gray')
        plt.title('Loaded Mask (binary)')
        plt.show()

        img, msk = img[...,None], msk[...,None] # Add channel dimension for Albumentations

        out = self.tf(image=img, mask=msk)
        img_t = out['image']                     # C x H x W (already)
        msk_t = out['mask']                     # H x W  or 1 x H x W depending on Albumentations

        if msk_t.ndim == 3 and msk_t.shape[0] != 1:  # H x W x 1  →  1 x H x W
            msk_t = msk_t.permute(2, 0, 1)
        elif msk_t.ndim == 2:                       # H x W  →  1 x H x W
            msk_t = msk_t.unsqueeze(0)

        return img_t.float(), msk_t.float()