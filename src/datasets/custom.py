import os
import glob
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import torch.nn.functional as F
from PIL import Image
from utils.helper_funcs import normalize

np_normalize = lambda x: (x-x.min())/(x.max()-x.min())


class CustomDatasetFast(Dataset):
    def __init__(self,
                 mode,
                 data_dir=None,
                 one_hot=True,
                 image_size=224,
                 aug=None,
                 aug_empty=None,
                 transform=None,
                 img_transform=None,
                 msk_transform=None,
                 add_boundary_mask=False,
                 add_boundary_dist=False,
                 logger=None,
                 **kwargs):
        self.print = logger.info if logger else print
        
        # pre-set variables
        self.data_dir = data_dir if data_dir else "/path/to/datasets/custom"

        # input parameters
        self.one_hot = one_hot
        self.image_size = image_size
        self.aug = aug
        self.aug_empty = aug_empty
        self.transform = transform
        self.img_transform = img_transform
        self.msk_transform = msk_transform
        self.mode = mode
        
        self.add_boundary_mask = add_boundary_mask
        self.add_boundary_dist = add_boundary_dist

        # Load data based on mode
        if mode == "tr":
            self.img_dir = os.path.join(self.data_dir, "train", "images")
            self.msk_dir = os.path.join(self.data_dir, "train", "masks")
            self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
            self.msk_paths = sorted(glob.glob(os.path.join(self.msk_dir, "*.png")))
        elif mode == "vl":
            self.img_dir = os.path.join(self.data_dir, "val", "images")
            self.msk_dir = os.path.join(self.data_dir, "val", "masks")
            self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
            self.msk_paths = sorted(glob.glob(os.path.join(self.msk_dir, "*.png")))
        elif mode == "te":
            self.img_dir = os.path.join(self.data_dir, "test", "images")
            self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
            self.msk_paths = None  # Test data has no masks
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.print(f"{mode.upper()} dataset: Found {len(self.img_paths)} images")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        data_id = os.path.basename(self.img_paths[idx]).replace('.jpg', '')
        
        # Load image
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = transforms.Resize((self.image_size, self.image_size))(img)
        img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        
        # Load mask (if available)
        if self.msk_paths is not None and idx < len(self.msk_paths):
            msk = Image.open(self.msk_paths[idx]).convert('L')
            msk = transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.NEAREST)(msk)
            msk = torch.tensor(np.array(msk)).unsqueeze(0).float() / 255.0
            msk = torch.where(msk > 0.5, 1.0, 0.0)  # Binarize
        else:
            # Create dummy mask for test data
            msk = torch.zeros(1, self.image_size, self.image_size)

        if self.one_hot and self.msk_paths is not None:
            msk = (msk - msk.min()) / (msk.max() - msk.min() + 1e-8)
            msk = F.one_hot(torch.squeeze(msk).to(torch.int64))
            msk = torch.moveaxis(msk, -1, 0).to(torch.float)

        # Apply transforms
        if self.img_transform:
            img = self.img_transform(img)
        if self.msk_transform and self.msk_paths is not None:
            msk = self.msk_transform(msk)
            
        img = img.nan_to_num(0.5)
        msk = msk.nan_to_num(-1)
        
        sample = {"image": img, "mask": msk, "id": data_id}
        return sample
