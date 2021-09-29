import os
import sys
import glob
from pathlib import Path
from typing import Counter
import numpy as np
import skimage.io as io
from PIL import Image


# -------------------------------------------------------------------------------------------------
# find out segmentation for rgb failured

file = 'bk_clear'  # 'bk_clear'、'bk_mix'、'bk_lowcontrast'

# data_for_Unsup_rmbg
root_dir = 'data/data_for_Unsup_rmbg'
dir_for_Unsup = Path(f'{root_dir}/{file}')
imgs_for_Unsup_name = {path.stem for path in dir_for_Unsup.glob('*.png')}
print(len(imgs_for_Unsup_name))

# label_waiting_postprocess
root_dir_rgb = 'data/label_waiting_postprocess/mask_waitinting_for_posrprocess/rgb_background_for_fill'
dir_rgb = Path(f'{root_dir_rgb}/{file}')
print(len(imgs_rgb))
imgs_rgb_name = {path.stem.split('。')[0].split('-')[0]
                 for path in dir_rgb.glob('*.png')}
print(len(imgs_rgb_name))

# mask_picked
dir_masks_picked = Path('data/data_for_Sup_train/masks')
imgs_masks_names = {path.stem for path in dir_masks_picked.glob('*.png')}
print(len(imgs_masks_names))

save_dir = Path(f'./data/tmp/{file}_failure')
if not save_dir.exists():
    save_dir.mkdir()
    print(save_dir, 'created')

imgs_failure_name = imgs_for_Unsup_name - imgs_rgb_name - imgs_masks_names
print(len(imgs_failure_name))
assert len(imgs_for_Unsup_name) - len(imgs_rgb_name) == len(imgs_failure_name)

for i, img_name in enumerate(imgs_for_Unsup_name):
    if img_name not in imgs_failure_name:
        continue
    path = dir_for_Unsup.joinpath(img_name + '.png')
    img = io.imread(path)
    Image.fromarray(img).save(save_dir.joinpath(img_name + '.png'))
    print(i, img_name)

# =================================================================================================================

# ------------------------------------------------------------------------------------------------
# exclude mask picked

# file = 'bk_mix'  # 'bk_clear'、'bk_mix'、'bk_lowcontrast'
# dir_rgb = Path(
#     f'data/label_waiting_postprocess/mask_waitinting_for_posrprocess/rgb_background_for_fill/{file}')
# imgs_rgb = list(dir_rgb.glob('*.png'))
# print(len(imgs_rgb))
# imgs_rgb_name =  {path.stem.split('。')[0].split('-')[0] for path in imgs_rgb}
# print(len(imgs_rgb_name))

# dir_masks_picked = Path('data/data_for_Sup_train/masks')
# imgs_masks_names = {path.stem for path in dir_masks_picked.glob('*.png')}
# print(len(imgs_masks_names))

# inter = set(imgs_rgb_name) & set(imgs_masks_names)
# print(len(inter), ':', inter)

# save_dir = Path(f'./data/tmp/{file}_tmp')
# if not save_dir.exists():
#     save_dir.mkdir()
#     print(save_dir, 'created')

# c=1
# for i, path in enumerate(imgs_rgb):
#     img_name = path.name.split('。')[0].split('-')[0].split('.')[0]
#     if img_name in imgs_masks_names:
#         print(f'\t{i:3d}, {img_name} will not save')
#         continue
#     img = io.imread(path)
#     Image.fromarray(img).save(save_dir.joinpath(path.name))
#     print(f'{i:3d}, {c}, {img_name} saved')
#     c+=1

# ------------------------------------------------------------------------------------------------

# data_for_Unsup_rmbg
# dir_for_Unsup = Path('data/data_for_Unsup_rmbg')
# file_for_Unsup = set(dir_for_Unsup.rglob('*.png'))
# print(len(file_for_Unsup))
# imgs_for_Unsup_name = {path.stem for path in dir_for_Unsup.glob('*.png')}
# print(len(imgs_for_Unsup_name))


# dir_mask_picked = Path('data/data_for_Sup_train/masks')
# file_mask_picked = {path.stem for path in dir_mask_picked.glob('*.png')}
# print(len(file_mask_picked))
# dir_mask_postprocess = Path('data/processed/finalstep/mask_unsup2sup_crf_cntr')
# file_mask_postprocess = {path.stem.split('。')[0].split('-')[0] for path in dir_mask_postprocess.glob('*.png')}
# print(len(file_mask_picked))

# mask_unpicked = file_mask_postprocess - file_mask_picked
# print(len(mask_unpicked))

# save_dir = Path(f'./data/tmp/mask_unpicked_tmp')
# if not save_dir.exists():
#     save_dir.mkdir()
#     print(save_dir, 'created')

# c = 0
# for i, path in enumerate(file_for_Unsup):

#     if path.stem not in mask_unpicked:
#         continue

#     img_origin = io.imread(path)
#     Image.fromarray(img_origin).save(save_dir.joinpath(path.stem + '_origin' + '.jpg'))

#     img_mask = io.imread(dir_mask_postprocess.joinpath(path.name))
#     Image.fromarray(img_mask).save(save_dir.joinpath(path.name))

#     print(i, c, path.name)
#     c += 1

