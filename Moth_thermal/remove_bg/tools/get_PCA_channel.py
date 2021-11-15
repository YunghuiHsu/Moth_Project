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

imgs = (
    # (io.imread(path, as_gray=True)*255).astype(np.uint8)
    io.imread(path)
    for path in dir_target.iterdir()
)

imgs_pca = []
for img in imgs:
    pca = PCA(n_components=1)
    img_ = np.transpose(img, (2, 0, 1))  # (h, w, c)  > (ｃ, h, w)
    img__ = img_.reshape(3, -1)          # (c, h, w) > (c, h*w )
    img_low_dimension = pca.fit_transform(img__)
    img_pca_ = pca.inverse_transform(img_low_dimension).astype(np.uint8)
    img_pca = np.transpose(img_pca_.reshape(3, 256, 256), (1, 2, 0))
    imgs_pca.append(img_pca)
    break

Img = Image.fromarray(img_pca)
Img_gray = ImageOps.grayscale(Img)
Img_gray.show()

img_r = normalizion(img[...,0])
Image.fromarray(img_r).show()

save_dir = Path(f'data/tmp/{file}_PCA_perfile')
if not save_dir.exists():
    save_dir.mkdir()

for idx, i in enumerate(imgs_pca):
    img_name = file_target[idx].name
    Img = Image.fromarray(i)
    Img_gray = ImageOps.grayscale(Img)
    Img_gray.save(save_dir.joinpath(img_name))
    print(idx, img_name, 'saved')

# pca for batch/all file


def normalizion(img):
    img_nor = (img - img.mean()) / img.std()
    img_nor = (img_nor*255).astype(np.uint8)
    return img_nor


dir_target = Path(f'data/data_for_Unsup_rmbg')
file_target = list(path for path in dir_target.rglob('*.png'))

imgs = np.array([
    # normalizion(io.imread(path, as_gray=True))
    (io.imread(path, as_gray=True)*255).astype(np.uint8)
    for path in dir_target.rglob('*.png')
])
print(imgs.shape)

pca = PCA(n_components=1)
imgs_low_dimension = pca.fit_transform(imgs.reshape(len(imgs), -1))
print(imgs_low_dimension.shape)
imgs_pca_ = pca.inverse_transform(imgs_low_dimension).astype(np.uint8)
print(imgs_pca_.shape)
imgs_pca = imgs_pca_.reshape(len(imgs), 256, 256)
print(imgs_pca.shape)

save_dir = Path(f'data/tmp/{file}_PCA')
if not save_dir.exists():
    save_dir.mkdir()

for idx, i in enumerate(imgs_pca):
    img_name = file_target[idx].name
    Img = Image.fromarray(i)
    Img_gray = ImageOps.grayscale(Img)
    Img_gray.save(save_dir.joinpath(img_name))
    print(idx, img_name, 'saved')

Image.fromarray(imgs_pca[3]).show()

ig.min()
ig.max()
ig.mean()

Ig = Image.fromarray(ig)
Ig_gray = ImageOps.grayscale(Ig)
ig_gray = np.array(Ig_gray)
ig_gray.min()
ig_gray.max()
ig_gray.mean()
ig_gray.std()

ig_nor = ((ig_gray - ig_gray.min()) / (ig_gray.max() - ig_gray.min())
img=imgs[500]
Image.fromarray(img).show()



ig_gray=normalizion(img)
Image.fromarray(ig_gray).show()
ig_gray.min()
ig_gray.max()
ig_gray.mean()
ig_gray.std()

def normalizion_minmax(img_gray):
    img_nor=(img_gray - img_gray.min()) / (img_gray.max() - img_gray.min())
    img_nor=(img_nor*255).astype(np.uint8)
    return img_nor
ig_gray=normalizion_minmax(img)
Image.fromarray(ig_gray).show()
ig_gray.min()
ig_gray.max()
ig_gray.mean()
ig_gray.std()


imgs_rgb=np.array([
    # normalizion(io.imread(path, as_gray=True))
    io.imread(path)
    for path in dir_target.rglob('*.png')
])

img_rgb=imgs_rgb[500]
Image.fromarray(img_rgb[..., 0]).show()
Image.fromarray(img_rgb[..., 1]).show()
Image.fromarray(img_rgb[..., 2]).show()
