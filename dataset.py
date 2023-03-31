import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import kornia
from kornia import augmentation as K
import torch

class MuticlassImageSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, masks_df, noclases, img_size = 128, use_augmentation = True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.masks_df = masks_df
        self.img_size = img_size

        self.no_clases = noclases

        self.aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=30.0, p=0.5),
            K.RandomBrightness(0.2, p=0.5),
            K.ColorJitter(0.2, 0.2, 0.2, 0.2, p=0.5),
            K.RandomAffine(degrees=0, translate=(0.1, 0.1), p=0.5),
            data_keys=["input", "mask"],
        )

        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.masks_df)
    
    def __getitem__(self, idx):
        row = self.masks_df.iloc[idx]
        image_name = row['image']
        mask_val = row['mask']

        image_path = os.path.join(self.image_dir, 'img_' + image_name + '.png')
        mask_path = os.path.join(self.mask_dir, 'seg_' + image_name + '.png')
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.transpose(2, 0, 1)
        image = image.astype(np.float32)
        image = image / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask == mask_val, 1, 0)
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)

        mask_one_hot = np.zeros((self.no_clases,), dtype=np.float32)
        mask_one_hot[mask_val-1] = 1.0

        image, mask = torch.from_numpy(image).float(), torch.from_numpy(mask).float()
        mask_one_hot = torch.from_numpy(mask_one_hot).float()

        if self.use_augmentation:
            image, mask = self.aug(image, mask)

        return image[0], mask[0], mask_one_hot