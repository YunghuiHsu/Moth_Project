import os
import sys
import glob
from pathlib import Path
import numpy as np
import skimage.io as io
from PIL import Image

import matplotlib.pyplot as plt
import cv2
print('cv2.__version__:', cv2.__version__)


dir_mask = Path('data/data_for_Sup_train/masks')
paths_mask = list(dir_mask.glob('*.png'))
print(len(paths_mask))

dir_save = Path('data/tmp/get_contour')
dir_save.mkdir(exist_ok=True, parents=True)

path = paths_mask[0]
name = path.stem.split('_mask')[0]


img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
# Image.fromarray(img).show()
_, img = cv2.threshold(img, 127, 255, 0)
# Image.fromarray(img).show()

iterations = 5
ksize = (3, 3)
kernel_RECT = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=ksize)
print('MORPH_RECT kernel:\n', kernel_RECT)
img_dilate1 = cv2.dilate(img, kernel_RECT, iterations=iterations)
img_erode1 = cv2.erode(img, kernel_RECT, iterations=iterations)

kernel_ELLIPSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=ksize)
print('MORPH_ELLIPSE kernel:\n', kernel_ELLIPSE)
img_dilate2 = cv2.dilate(img, kernel_ELLIPSE, iterations=iterations)
img_erode2 = cv2.erode(img, kernel_ELLIPSE, iterations=iterations)

kernel_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize=ksize)
print('MORPH_CROSS kernel_CROSS:\n', kernel_CROSS)
img_dilate3 = cv2.dilate(img, kernel_CROSS, iterations=iterations)
img_erode3 = cv2.erode(img, kernel_CROSS, iterations=iterations)

imgs_dilate = [img_dilate1, img_dilate2, img_dilate3]
imgs_erode = [img_erode1, img_erode2, img_erode3]


for kernel_, img_dilate, img_erode in zip(['RECT', 'ELLIPSE', 'CROSS'], imgs_dilate, imgs_erode):
    img_contour = img_dilate - img_erode
    io.imsave(dir_save.joinpath(name + f'_contour_{kernel_}.png'), img_contour)

img_contour = img_dilate2 - img_erode2

# =====================================================================================
