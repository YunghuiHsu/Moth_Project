import torchvision.utils as vutils
import argparse
import logging
import pathlib
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import skimage.io as io

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.data_loading_moth import BasicDataset, MothDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet

# ---------------------------------------------------
# data prepare

dir_img = Path('./data/data_for_Sup_train/imgs/')
dir_mask = Path('./data/data_for_Sup_train/masks/')
dir_checkpoint = Path('./checkpoints/')

img_paths = list(dir_img.glob('*.png'))
mask_paths = list(dir_mask.glob('*.png'))

X_train, X_valid, y_train, y_valid = train_test_split(
    img_paths, mask_paths, test_size=0.2, random_state=1)
len(X_train), len(X_valid), len(y_train), len(y_valid)

# dataset = BasicDataset(dir_img, dir_mask, img_scale)
# img_scale = 1.
train_set = MothDataset(X_train, y_train, img_aug=True)
val_set = MothDataset(X_valid, y_valid, img_aug=False)

# 2. Split into train / validation partitions
# val_percent = 0.1
# n_val = int(len(dataset) * val_percent)
# n_train = len(dataset) - n_val
# # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
# train_set, val_set = random_split(dataset, [n_train, n_val])

# 3. Create data loaders
batch_size = 16
loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)



dir_save = Path('tmp/imgaug_test')
dir_save.mkdir(parents=True, exist_ok=True)
# ---------------------------------------------------


batch_size = len(train_loader)
# epochs = 10
# for epoch in range(epochs):

for iter, batch in enumerate(train_loader):
    print(iter)
    images = batch['image']
    true_masks = batch['mask']


    if iter % 1 is 0:
        vutils.save_image(
            torch.cat(
                [images,
                 torch.stack([true_masks, true_masks, true_masks], dim=1)
                 ], dim=0).data.cpu(),
            dir_save.joinpath(f'check_imgaug_iter{iter}.jpg'),
            nrow=8)
        print(f'check_imgaug_iter{iter}.jpg saved')
    if iter == 5 :break
        
images[0].shape
true_masks.shape
images[0].shape
true_masks[0].dtype
len(batch)


true_masks[0].numpy().max()


mask_t = true_masks[0] > 0.9

Image.fromarray(mask_t.numpy()).show()