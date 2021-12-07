import os
import sys
import glob
import re
from pathlib import Path
import numpy as np
import skimage.io as io
from PIL import Image
import math
import matplotlib.pyplot as plt
import shutil

# -------------------------------------------------------------------------------------------------
# find out segmentation for rgb failured

# file = 'spot_color'  # 'bk_clear'、'bk_mix'、'bk_lowcontrast'、'spot_color'

# # data_for_Unsup_rmbg
# root_dir = 'data/data_for_Unsup_rmbg'
# dir_for_Unsup = Path(f'{root_dir}/{file}')
# imgs_for_Unsup_name = {path.stem for path in dir_for_Unsup.glob('*.png')}
# print(len(imgs_for_Unsup_name))

# # # label_waiting_postprocess
# root_dir_rgb = 'data/label_waiting_postprocess/mask_waitinting_for_posrprocess/rgb_background_for_fill'
# dir_rgb = Path(f'{root_dir_rgb}')
# imgs_rgb = list(dir_rgb.glob('*cropped.png'))
# print(len(imgs_rgb))
# imgs_rgb_name = {path.stem
#                  for path in dir_rgb.glob('*cropped.png')}
# print(len(imgs_rgb_name))

# # # mask_picked
# dir_masks_picked = Path('data/data_for_Sup_train/masks')
# imgs_masks_names = {path.stem for path in dir_masks_picked.glob('*.png')}
# print(len(imgs_masks_names))

# save_dir = Path(f'./data/tmp/{file}_failure')
# save_dir.mkdir(parents=True, exist_ok=True)

# imgs_failure_name = imgs_for_Unsup_name - imgs_rgb_name - imgs_masks_names
# print(len(imgs_failure_name))
# assert len(imgs_for_Unsup_name) - len(imgs_rgb_name) == len(imgs_failure_name)

# for idx, img_name in enumerate(imgs_for_Unsup_name):
#     if img_name not in imgs_failure_name:
#         continue
#     path = dir_for_Unsup.joinpath(img_name + '.png')
#     img = io.imread(path)
#     io.imsave(save_dir.joinpath(img_name + '.png',img)
#     print(idx, img_name)

# for idx, img_name in enumerate(imgs_failure_name):
#     path = dir_for_Unsup.joinpath(img_name + '.png')
#     img = io.imread(path)
#     io.imsave(save_dir.joinpath(img_name + '.png'), img)
#     print(idx, img_name)

# =================================================================================================================

# ------------------------------------------------------------------------------------------------
# exclude mask picked

# imgs need for checked
# dir_mask_waiting = Path(f'../../data/label_waiting_postprocess')

# data\data_for_Sup_predict\SJRS_for_predict、SJRS_for_predict、MCTT_for_predict
# file = 'SJRS_for_predict'
# dir_target = Path(f'../../data/data_for_Sup_predict/{file}')
dir_target = Path('../../data/data_for_Sup_train/imgs')
imgs_target = list(dir_target.glob('*.png'))
print('imgs_target : ', len(imgs_target))
imgs_target_name = {path.stem for path in imgs_target}
print('imgs_target_name : ', len(imgs_target_name))


# mask_picked
dir_masks_picked = Path('../../data/data_for_Sup_train/masks')
imgs_masks_names = {path.stem for path in dir_masks_picked.glob('**/*.png')}
print('imgs_masks : ', len(imgs_masks_names))

inter_ = set(imgs_target_name) & set(imgs_masks_names)
except_ = set(imgs_target_name) - set(imgs_masks_names)
print('imgs_target & imgs_masks :', len(inter_))
print('imgs_target - imgs_masks :', len(except_))

save_dir = Path(f'data/tmp/target_tmp')
save_dir = Path(f'../../data/data_for_Sup_predict/final')
save_dir.mkdir(parents=True, exist_ok=True)

error_name = {}
for idx, img_name in enumerate(except_):
    # img_name = path.name.split('。')[0].split('-')[0].split('.')[0]
    path = dir_target.joinpath(img_name + '.png')
    try:
        path.unlink()
        # img = io.imread(path)
    except OSError as e:
        print(f"Error:{ e.strerror}")
        error_name[idx] = img_name

    # io.imsave(save_dir.joinpath(img_name + '.png'), img)

    print(f'{idx:3d}, {img_name}')

# =================================================================================================
# Transform img_rmbg from mask
# -------------------------------------------------------------------------------------------------

# # load masks
# dir_target = Path('data\data_for_Sup_predict')
# masks = list(dir_target.glob('*Mask.png'))
# masks_name = [path.stem.split('_UnetMask')[0] for path in masks]
# print(f'masks : {len(masks)}')
# print(f'masks_name : {len(masks_name)}')

# # ---------------------------------------------------------------------------------------
# # 確認是否有亂碼
# # ^: not
# # 標點符號 (Punctuation & Symbols):   \u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E
# # \w: any word character

# regex = re.compile(
#     r'[^\w\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E\s]')
# check_error_character = [n for n in masks_name if regex.findall(n) != []]
# print(len(check_error_character))
# assert check_error_character == [
# ], f'wrong character : {check_error_character}'
# # ---------------------------------------------------------------------------------------


# # load raw imgs
# dir_origin = Path('../crop/origin')


# # get img with background removed
# name_error = []
# for idx, name in enumerate(masks_name):

#     try:
#         img_ = io.imread(dir_origin.joinpath(name + '.png')
#                          )            # (h,w,c), [0, 255], uint8
#     except Exception as e:
#         print(e)
#         name_error.append(name)

#     # load mask and transform
#     mask_ = io.imread(dir_target.joinpath(
#         name + '_UnetMask.png'))  # (h,w), [0|255], uint8
#     # (h,w,3). [0|255], uint8 >　[0.0|1.0], float64
#     mask_3 = np.stack([mask_, mask_, mask_], axis=2)/255
#     white_mask = 1-mask_3
#     blue_mask = np.zeros_like(mask_3)
#     # assign channel 3 depends on binary mask(0,1) (0,0,0) > (0,0,1)
#     blue_mask[..., 2] = white_mask[..., 2]

#     # get img with background removed
#     img_rmgb = (img_ * mask_3 + blue_mask*255).astype('uint8')

#     # save img
#     io.imsave(dir_target.joinpath(name + '.png'), img_)
#     io.imsave(dir_target.joinpath(name + '_UnetRmbg.png'), img_rmgb)
#     print(f'{idx:4d}, {name} saved')
# =================================================================================================
