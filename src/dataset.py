import os, cv2, torch, numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

base_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    ToTensorV2(),
])
no_aug  = A.Compose([ToTensorV2()])

class NpySegDataset(Dataset):
    """Loads npy slice + png mask from predefined split dir (train/val/test)"""
    def __init__(self, split_dir: str, augment=False):
        self.split_dir = split_dir
        self.img_dir = os.path.join(split_dir, 'images') if os.path.isdir(os.path.join(split_dir,'images')) else split_dir
        self.mask_dir= os.path.join(split_dir, 'masks')  if os.path.isdir(os.path.join(split_dir,'masks'))  else split_dir
        self.files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.npy')])
        self.tf = base_aug if augment else no_aug
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        fname = self.files[idx]
        img = np.load(os.path.join(self.img_dir, fname)).astype(np.float32)
        msk = cv2.imread(os.path.join(self.mask_dir, fname.replace('.npy','.png')), cv2.IMREAD_GRAYSCALE)
        msk = (msk>127).astype(np.float32)
        img, msk = img[...,None], msk[...,None]
        out = self.tf(image=img, mask=msk)
        img_t = out['image']                     # C x H x W (already)
        msk_t = out['mask']                     # H x W  or 1 x H x W depending on Albumentations
        if msk_t.ndim == 3 and msk_t.shape[0] != 1:  # H x W x 1  →  1 x H x W
            msk_t = msk_t.permute(2, 0, 1)
        elif msk_t.ndim == 2:                       # H x W  →  1 x H x W
            msk_t = msk_t.unsqueeze(0)

        return img_t.float(), msk_t.float()