import argparse
from PIL import Image
import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import time
from datetime import datetime
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt


from func.postprocessing import find_cntr_condition, crf, find_cntr
from func.tool import get_fname
from func.plot import plt_result


# =======================================================================================================
# def get_args():
parser = argparse.ArgumentParser(
    description='PosTprocess masks')

# enviroment
parser.add_argument('--gpu', '-g', dest='gpu', default='0')

# data
parser.add_argument('--MASKDIR', '-i', dest='MASKDIR',
                    default='../../data/label_waiting_postprocess/mask_waitinting_for_posrprocess/mask_for_Postprocess')
parser.add_argument('--ORIGINDIR', '-o', dest='ORIGINDIR',
                    default='../../data/origin')
parser.add_argument('--SAVEDIR', '-s', dest='SAVEDIR',
                    default='../../data/postprocessed')
# parser.add_argument('--postfix', '-p', dest='postfix', type=str,
#                     default='_mask', help="postfix of mask file for split out image name ")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# =======================================================================================================

# # (1) sup+cntr (2)+crf (3) +cntr
# --------------------------------------------------------------------------------------
# MASKDIR放置要做為mask的底。
# 採用sup_train產出來的結果，再經過find_cntr_condition, crf, find_cntr等方法做微調優化
# 不過初步當sup_train模型還不能使用時，需要先用unsup_train以及手動去背的結果來使用
# --------------------------------------------------------------------------------------

ORIGIN = Path(args.ORIGINDIR)
# UnsupCV2DIR = Path('model/Unsup_rmbg/result_sample/predict_img_postprocessd/')
# MASKDIR = 'data/bk_mask_convert/'
# SAVEDIR = f'data/processed/finaluse/'


MASKDIR = Path(args.MASKDIR)
time_ = datetime.now().strftime("%y%m%d_%H%M")
SAVEDIR = Path(args.SAVEDIR).joinpath(time_)

mask_path = MASKDIR.glob('*.png')
print(len(list(MASKDIR.glob('*.png'))))
# mask_path = [os.path.join(MASKDIR, i) for i in masks if os.path.splitext(i)[-1].lower() in ['.jpg', '.png', '.jpeg']]


def img_rmbg_fill(black_mask: np.ndarray, img: np.ndarray, color: str = 'blue'):
    '''
    color:: 'blue', 'black'.
    black_mask:: shape=(h,w,c), dtype=uint8.
    img:: shape=(h,w,c), dtype=uint8
    '''
    assert black_mask.dtype == 'uint8' and img.dtype == 'uint8', 'dtype must be "uint8"'
    assert black_mask.shape[2] == 3 and img.shape[
        2] == 3, 'shape and channelmust be (h,w,c) and 3'

    if color == 'black':
        color_mask = black_mask
    elif color == 'blue':
        white_mask = (255-black_mask)
        blue_mask = np.zeros_like(white_mask)
        blue_mask[..., 2] = white_mask[..., 2]
        color_mask = blue_mask

    img_rmbg_color = ((black_mask/255)*img + color_mask).astype(np.uint8)
    return img_rmbg_color


for idx, path in enumerate(mask_path):
    # get path
    fname = path.stem.split('_cropped')[0] + '_cropped'
    msk_path = path

    # prepare data to [0,255], channel=3
    img_path = ORIGIN.joinpath(fname + '.png')
    origin_img = io.imread(img_path)
    origin_img = resize(origin_img, (256, 256))
    origin_img_255 = (origin_img)*255
    origin_img_255 = origin_img_255.astype(np.uint8)

    mask = io.imread(msk_path, as_gray=True)

    if mask.shape is not (256, 256):
        mask = resize(mask, (256, 256))
    if not mask.max() > 1:
        mask = mask*255
    mask = mask.astype(np.uint8)
    mask3 = np.stack((mask, mask, mask), axis=2)

    # ============================================================================================================================================
    # cntr
    # find contour by cv2.findContours(mask), return mask [h,w,3]
    mask_unsup2cntr = find_cntr_condition(
        mask3, condition=62000)             # (h,w,c), uint8
    img_unsup2cntr = img_rmbg_fill(mask_unsup2cntr, origin_img_255)

    # ------------------------------------------------------------------------------------------
    # crf
    # find contour by crf(img, mask), only get from 'R' channel, return mask [h,w,3]
    crf_output = crf(origin_img_255, mask3)
    mask_crf = crf_output[:, :, 0] + mask3[:, :, 0]
    mask_crf = np.repeat(mask_crf, repeats=3).reshape((256, 256, 3))
    img_crf = img_rmbg_fill(mask_crf, origin_img_255, color='blue')

    # ------------------------------------------------------------------------------------------
    # crf_cntr
    # find contour by cv2.findContours(mask_crf) , return mask [h,w,3]
    mask_cntr = find_cntr_condition(mask_crf, condition=62000)
    img_crf_cntr = img_rmbg_fill(mask_cntr, origin_img_255, color='blue')

    # ------------------------------------------------------------------------------------------
    # show unsup+cv2 result
    # unsupcv2_img = io.imread(os.path.join(UnsupCV2DIR, fname+'.png'))

    # ============================================================================================================================================
    # save out

    # plotting checking.jpg
    p_img = [origin_img,
             #               unsupcv2_img,
             img_unsup2cntr,
             img_crf,
             img_crf_cntr]
    p_title = ['Original',
               #               'unsup',
               'cntr',
               'crf',
               'crf_cntr']

    fig = plt_result(p_img, p_title)
    save_to = SAVEDIR.joinpath('checking')
    save_to.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_to.joinpath(fname + '.jpg'),
                dpi=100, format='jpg', bbox_inches='tight')

    # ------------------------------------------------------------------------------------------
    imgs_dict = {'cntr': img_unsup2cntr,
                 'crf': img_crf,
                 'crf_cntr': img_crf_cntr
                 }
    masks_dict = {'cntr': mask_unsup2cntr,
                  'crf': mask_crf,
                  'crf_cntr': mask_cntr
                  }

    for (name_, img_), (_, mask_) in zip(imgs_dict.items(), masks_dict.items()):
        save_to = SAVEDIR.joinpath(name_)
        if idx == 0:
            save_to.mkdir(parents=True, exist_ok=True)
        io.imsave(save_to.joinpath(fname + f'_rmbg_{name_}.png'), img_)
        io.imsave(save_to.joinpath(fname + f'_mask_{name_}.png'), mask_)

    # print(f'\t{fname}_rmbg_{name_}.png saved')

    print(idx, fname, ' saved')
