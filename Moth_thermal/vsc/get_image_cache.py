import os
import sys
import argparse
import numpy as np
from pathlib import Path
import skimage.io as io

from utils.dataset import ImageDatasetFromFile, ImageDatasetFromCache

parser = argparse.ArgumentParser(
    description='Prepare image_catch as data.npz for Autoencoder model training'
)
parser.add_argument('--data', type=str, default="moth_thermal_rmbg_padding_256",
                    help=' "moth_thermal_rmbg_padding_256" or "benchmarks". \
                        data root located in vsc/data/ ')
args = parser.parse_args()

# Load dataset--------------------------

dir_data = Path(f'data/{args.data}')
image_list = list(dir_data.glob('*.png'))
print(f'Data size in {str(dir_data)} : {len(image_list)}')
assert len(image_list) > 0, f'Plz Check whether data in {str(dir_data)}'


train_list = image_list[:]
train_list

# Prepare image catch --------------------------
print('Preparing image catch')
preloads = [io.imread(img_path) for img_path in train_list]
preloads_npy = np.array(preloads)
print(f'\tpreloads_npy.shape : {preloads_npy.shape}')
path_preloads = Path(f'./data/{args.data}.npz')
print(f'\tSave as {path_preloads}...')
np.savez_compressed(path_preloads, image_catch=preloads_npy)

# Load image catch   --------------------------
# load image_catch.npz
print(f'\tLoad image_catch from {path_preloads}...')
image_cache = np.load(path_preloads)['image_catch']
print(f'\tCheck image_cache.shape : {image_cache.shape}')

# train_set = ImageDatasetFromFile(train_list, aug=True)
# train_set = ImageDatasetFromCache(image_cache, aug=True)
