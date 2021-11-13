import argparse
import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import skimage.io as io
import random
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
from sklearn.model_selection import train_test_split
import torchvision.utils as vutils

from utils.data_loading_moth import MothDataset
from utils.sampler_moth import AddBatchAugmentSampler, NonBatchAugmentSampler

# =======================================================================================================
# def get_args():
parser = argparse.ArgumentParser(
    description='Train the UNet on images and target masks')

# enviroment
parser.add_argument('--gpu', '-g', dest='gpu', default='0')
parser.add_argument("--save_epoch", '-save_e', type=int,
                    default=1, help="save checkpoint per epoch")

# data
parser.add_argument('--XX_DIR', dest='XX_DIR', type=str,
                    default='../data/data_for_Sup_train/imgs/')
parser.add_argument('--YY_DIR', dest='YY_DIR', type=str,
                    default='../data/data_for_Sup_train/masks_211105/')
parser.add_argument('--SAVEDIR', dest='SAVEDIR', type=str,
                    default='model/Unet_rmbg')  # Unet_rmbg

parser.add_argument('--image_input_size', '-s_in', dest='size_in',
                    type=str, default='256,256', help='image size input')
parser.add_argument('--image_output_size', '-s_out', dest='size_out',
                    type=str, default='256,256', help='image size output')
parser.add_argument('--image_channel', '-c', dest='image_channel',
                    metavar='Channel', default=3, type=int, help='channel of image input')
parser.add_argument('--img_type', '-t', dest='img_type',
                    metavar='TYPE', type=str, default='.png', help='image type: .png, .jpg ...')
parser.add_argument('--stratify_off',  action='store_true',
                    default=False, help='whether to stratified sampling')

# model
parser.add_argument('--epochs', '-e', metavar='E',
                    type=int, default=100, help='Number of epochs')
parser.add_argument('--batch-size', '-b', dest='batch_size',
                    metavar='B', type=int, default=8, help='Batch size')
parser.add_argument('--learning-rate', '-lr', dest='lr', metavar='LR', type=float, default=1e-3,
                    help='Learning rate')
parser.add_argument('--load', '-f', type=str,
                    default=False, help='Load model from a .pth file')
parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                    help='Percent of the data that is used as validation (0-100)')
parser.add_argument('--amp', action='store_true',
                    default=False, help='Use mixed precision')
parser.add_argument('--contour_weight', '-w', dest='contour_weight',
                    metavar='Contour_Weight', type=int, default=3, help='loss weight for contour area')
parser.add_argument('--loss_metric', '-m', dest='loss_metric',
                    metavar='Metric', type=str, default='max',
                    help="Loss fuction goal: maximize Dice score > 'max' / minimize Valid Loss > 'min'")


# return parser.parse_args()
args = parser.parse_args()
# ------------------------------------------------------
# for test
img_type = '.png'
val_percent = 0.1
batch_size = 8
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = UNet(n_channels=args.image_channel, n_classes=2, bilinear=True)
# net.to(device=device)
# =======================================================================================================


def size_str_tuple(input):
    str_ = input.replace(' ', '').split(',')
    h, w = int(str_[0]), int(str_[1])
    return h, w


input_size = size_str_tuple(args.size_in)
output_size = size_str_tuple(args.size_out)


dir_img = Path('../data/data_for_Sup_train/imgs')
dir_mask = Path('../data/data_for_Sup_train/masks_211105')

dir_img_arg = Path('../data/data_for_Sup_train/imgs_batch_arg')
img_arg_paths = list(dir_img_arg.glob('**/*' + 'png'))
dir_img_arg = Path('../data/data_for_Sup_train/masks_batch_arg')
masks_arg_paths = list(dir_img_arg.glob('**/*' + 'png'))
img_arg_names = [path.stem for path in img_arg_paths]


img_type = '.png'
# Prepare dataset, Split into train / validation partitions
img_paths = list(dir_img.glob('**/*' + img_type))
mask_paths = list(dir_mask.glob('**/*' + img_type))


assert len(img_paths) == len(
    mask_paths), f'number imgs: {len(img_paths)} and masks: {len(mask_paths)} need equal '

if not args.stratify_off:
    df = pd.read_csv('imgs_label_byHSV.csv', index_col=0)
    assert len(df.Name) == len(
        img_paths), f'number of imgs: {len(img_paths)} and imgs_label_byHSV.csv: {len(df.label)} need equal '
    print(
        f'Stratified sampling by "imgs_label_byHSV.csv", clustering: {np.unique(df.label).size}')

X_train, X_valid, y_train, y_valid = train_test_split(
    img_paths, mask_paths, test_size=val_percent, random_state=1,
    stratify=df.label if not args.stratify_off else None)

X_train_arg = X_train + img_arg_paths
y_train_arg = y_train + masks_arg_paths
size_X_train = len(X_train)

train_set = MothDataset(
    X_train, y_train, input_size=input_size, output_size=output_size, img_aug=True)
val_set = MothDataset(
    X_valid, y_valid, input_size=input_size, output_size=output_size, img_aug=True)

n_val = len(val_set)
n_train = len(train_set)

# Create data loaders
loader_args = dict(batch_size=batch_size, num_workers=2,
                   pin_memory=True, drop_last=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)

val_loader = DataLoader(val_set, shuffle=False,
                        drop_last=True, **loader_args)


# =============================================================================================================
mysampler = AddBatchAugmentSampler(X_train_arg, size_X_train, batch_size)
nonsampler = NonBatchAugmentSampler(X_train_arg, size_X_train, batch_size)

len(mysampler)
len(nonsampler)
for idx, batch in enumerate(mysampler):
    print(idx, batch)

for idx, batch in enumerate(nonsampler):
    print(idx, batch)


# loader_args = dict(batch_size=None, num_workers=2, pin_memory=False, drop_last=False)


train_set = MothDataset(
    X_train_arg, y_train_arg, input_size=input_size, output_size=output_size, img_aug=True)
len(train_set)

train_loader = DataLoader(train_set, batch_sampler=mysampler)

train_loader


dir_save = Path('tmp/batch_sampler_test')
dir_save.mkdir(exist_ok=True, parents=True)

for idx, batch in enumerate(train_loader):
    print(idx)
    images = batch['image']
    masks_true = batch['mask']
    vutils.save_image(
        torch.cat([images, torch.stack([masks_true]*3, dim=1)],
                  dim=0).data.cpu(),
        dir_save.joinpath(f'batch_{idx}.jpg'),
        nrow=8)


# =============================================================================================================
'''
But what are PyTorch DataLoaders really?
https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html#Custom-Sampler
'''


class IndependentHalvesSampler(Sampler):
    def __init__(self, dataset):
        halfway_point = int(len(dataset)/2)
        self.first_half_indices = list(range(halfway_point))
        self.second_half_indices = list(range(halfway_point, len(dataset)))

    def __iter__(self):
        random.shuffle(self.first_half_indices)
        random.shuffle(self.second_half_indices)
        return iter(self.first_half_indices + self.second_half_indices)

    def __len__(self):
        return len(self.first_half_indices) + len(self.second_half_indices)


xs = list(range(10))
ys = list(range(10, 20))
print('xs values: ', xs)
print('ys values: ', ys)
dataset = list(zip(xs, ys))
dataset[0]  # returns the tuple (x[0], y[0])


our_sampler = IndependentHalvesSampler(dataset)
print('First half indices: ', our_sampler.first_half_indices)
print('Second half indices:', our_sampler.second_half_indices)

for i in our_sampler:
    print(i)


dl = DataLoader(dataset, sampler=our_sampler)
for xb, yb in dl:
    print(xb, yb)

batch_sampler = BatchSampler(our_sampler, batch_size=2, drop_last=False)
for i, batch_indices in enumerate(batch_sampler):
    print(f'Batch #{i} indices: ', batch_indices)


def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)


a = torch.arange(10).reshape(5, 2)
a
a_split = torch.split(a, 2)
a_split[0]
a_split[1]


class EachHalfTogetherBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        halfway_point = len(dataset) // 2
        self.first_half_indices = list(range(halfway_point))
        self.second_half_indices = list(range(halfway_point, len(dataset)))
        self.batch_size = batch_size

    def __iter__(self):
        random.shuffle(self.first_half_indices)
        random.shuffle(self.second_half_indices)
        first_half_batches = chunk(self.first_half_indices, self.batch_size)
        second_half_batches = chunk(self.second_half_indices, self.batch_size)
        combined = list(first_half_batches + second_half_batches)
        combined = [batch.tolist() for batch in combined]
        # random.shuffle(combined)
        return iter(combined)

    def __len__(self):
        return (len(self.first_half_indices) + len(self.second_half_indices)) // self.batch_size


batch_size = 3
each_half_together_batch_sampler = EachHalfTogetherBatchSampler(
    dataset, batch_size)
for x in each_half_together_batch_sampler:
    print(x)

for i, (xb, yb) in enumerate(DataLoader(dataset, batch_sampler=each_half_together_batch_sampler)):
    print(f'Batch #{i}. x{i}:', xb)
    print(f'          y{i}:', yb)
