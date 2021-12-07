
import os
import glob
import numpy as np
from pathlib import Path
import matplotlib as plt
# import skimage.io as io
from skimage import io, color
import cv2
from PIL import Image, ImageOps
from skimage.transform import resize
import cv2

root = Path('./')
dir_origin = root.joinpath('origin')
imgs_origin = list(dir_origin.glob('*.png'))
print(len(imgs_origin))

# ==============================================================================================
# crop and padding for MCTT files
# ==============================================================================================


def resize_with_padding(img_PIL, expected_size=(256, 256), fill=(255, 255, 255)):
    img_PIL.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img_pil.size[0]
    delta_height = expected_size[1] - img_pil.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width -
               pad_width, delta_height - pad_height)
    return ImageOps.expand(img_PIL, padding, fill)

dir_MCTT = root.joinpath('data_raw/MCTT')

pathes_MCTT_ = list(dir_MCTT.glob('*'))
len(pathes_MCTT_)

pathes_others = [path.name for path in pathes_MCTT_
               if not path.suffix.lower() in ['.png', '.jpg', '.jpeg']]
print(pathes_others)
pathes_MCTT = [path for path in pathes_MCTT_
               if path.suffix.lower() in ['.png', '.jpg', '.jpeg']]
len(pathes_MCTT)

pathes_MCTT_png = [path for path in pathes_MCTT_
               if path.suffix.lower() in ['.png']]
len(pathes_MCTT_png)

pathes_MCTT_jpg = [path for path in pathes_MCTT_
               if path.suffix.lower() in ['.jpg', '.jpeg']]
len(pathes_MCTT_jpg)

dir_save_removeweb = dir_MCTT.joinpath('tmp_removeweb')
dir_save_removeweb.mkdir(parents=True, exist_ok=True)

dir_save_jpg = dir_MCTT.joinpath('tmp_jpg')
dir_save_jpg.mkdir(parents=True, exist_ok=True)


    # -------------------------------------------------------
for idx, path in enumerate(pathes_MCTT_jpg):
    name = path.stem
    img_ = io.imread(path)
    img_pil  = Image.fromarray(img_)
    img_padding =  resize_with_padding(img_pil)
    img_array = np.asarray(img_padding)

    # bg_white = np.where((img_.reshape(-1,3) == [255,255,255]).all(), [0,0,255], img_.reshape(-1,3)).reshape(img_.shape)

    io.imsave(dir_save_jpg.joinpath(name + '.png'),  img_array)
    print(idx, name, 'saved')

np.unique(img_[...,3])
img_ = io.imread(path)
img_[...,3].shape

mask_ =  img_[...,3]
mask_filter  = np.where(mask_>1, 255, 0 ) 
Image.fromarray(mask_filter).show()


for idx, path in enumerate(pathes_MCTT_png):
    name = path.stem
    img_ = io.imread(path)

    mask_ =  img_[...,3]
    mask_filter  = np.where(mask_>1, 255, 0 ) 
    mask_3 = np.dstack([(mask_filter/255)]*3)

    if img_.shape[2] == 4:
            img_ = img_[...,:3]

    bg_white  = (np.where(mask_3<0.5, 1, 0)*255).astype('uint8')
    img_bg_fill =  (img_ * mask_3).astype('uint8') + bg_white

    io.imsave(dir_save_removeweb.joinpath(name + '.png'),  img_bg_fill)
    print(idx, name, 'saved')
