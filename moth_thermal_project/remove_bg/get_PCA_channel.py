import os
from posixpath import split
import sys
import glob
import cv2
from pathlib import Path
import numpy as np
import skimage.io as io
from PIL import Image, ImageOps
from sklearn.decomposition import PCA

# pca for each file
file = 'bk_lowcontrast'
dir_target = Path(f'data/data_for_Unsup_rmbg/{file}')
file_target = list(dir_target.glob('*.png'))
print(len(file_target))

# save dir
save_dir = Path(f'data/tmp/imgs_c4_PCA')
save_dir.mkdir(parents=True, exist_ok=True)

def normalize(img: np.ndarray):
    img_ = (img - img.mean()) / img.std()   # normalize > [0., 1.]
    img_255 = (img_*255).astype(np.uint8)    # np.float64 > np.uint8
    return img_255


imgs_pca = []
pca_variance_ratios = {}
pca = PCA(n_components=3)
for idx, path in enumerate(file_target):
    name = path.stem
    img = io.imread(path)                             
    img_2d = img.reshape(-1, 3)                          # (h, w, c) > (h*w, c)
    img_low_dimension = pca.fit_transform(img_2d)        # (h*w, c)
    img_low_3d = img_low_dimension.reshape(256, 256, 3)  # (h*w, axis) > (h, w, axis)

    # get the pca axis0(the greatest axis for maximize r, g, b channel)
    pca_0 = normalize(img_low_3d)[...,0]                 # (h, w, axis0).  
    img_c4_pca = np.concatenate((img, pca_0[..., np.newaxis]),axis=2)

    # record the variance_ratio of pca axis0
    pca_score = pca.explained_variance_ratio_
    pca_variance_ratios[name] = pca_score[0]

    path_save =  save_dir.joinpath(name + '.png')
    io.imsave(path_save, img_c4_pca)
    print(f'{idx:5d}, {name:20s} saved. PCA axis0 variance ratio: {pca_score:4f}')

    
