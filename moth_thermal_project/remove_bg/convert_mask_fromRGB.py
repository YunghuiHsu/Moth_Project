import os
import glob
import numpy as np
from pathlib import Path
import matplotlib as plt
import skimage.io as io
from skimage import io, color
import cv2
from PIL import Image


# ============================================================================================================================
# convert_mask_from RGB

dir_origin = Path('../crop/origin')
imgs_origin = list(dir_origin .glob('*.png'))

dir_mask = Path('data/label_waiting_postprocess/rgb_selected/rgb_fill/')
imgs_mask = list(dir_mask.glob('*.png'))
print(f'file size : {len(imgs_mask)}')

dir_save = dir_mask.joinpath('convert')
dir_save.mkdir(exist_ok=True)

for idx, path in enumerate(imgs_mask):
    img = io.imread(path, as_gray=True)  # img.shape (h,w,c)， c = r, g, b, a
    img_name = path.stem.split('。')[0]

    # for rgb色塊圖，背景已處理為黑色的mask
    img_ = (img*255).astype('uint8')
    img_mask = np.where(img_ == 0, 0, 255).astype(np.uint8)

    # loading original img by img_name
    path_origin = dir_origin.joinpath(img_name + '.png')
    img_origin = io.imread(path_origin)

    # save img_mask and img_origin into dir_mask
    for count, img_ in enumerate([img_mask, img_origin]):
        if count == 0:
            save_name = img_name + '_mask' + '.png'
        elif count == 1:
            save_name = img_name + '.png'

        save_path = dir_save.joinpath(save_name)
        Image.fromarray(img_).save(save_path)
        print(idx, img_name, 'saved')

# ==================================================================================================================

# convert_from remove.bg 

# dir_origin = Path('../crop/origin')
# imgs_origin = list(dir_origin .glob('*.png'))


# file = 'rgb_background_for_fill'  # file_for_rmbg
# dir_mask = Path(
#     f'data/label_waiting_postprocess/mask_waitinting_for_posrprocess/{file}/mask')
# imgs_mask = dir_mask.glob('*.png')


# dir_save = Path(f'data/tmp/mask_for_pick_{file}')
# if not dir_save.exists():
#     dir_save.mkdir()


# err_name = []
# for idx, path in enumerate(imgs_mask):
#     try:
#         img_rmbg = io.imread(path)  # (256, 256, 4)
#         img_name = path.stem.replace('-removebg-preview', '')

#         # reading  alpha channel as mask, background == 0(mask), target(moth) == 255
#         mask_ = img_rmbg[..., 3]  # (256, 256)
#         # convert all non black(0) to white(255)
#         img_mask = np.where(mask_ == 0, 0, 255).astype('uint8')

#         # loading original img by img_name
#         path_origin = dir_origin.joinpath(img_name + '.png')
#         img_origin = io.imread(path_origin)

#         # save img_mask and img_origin into dir_mask
#         for i, img_ in enumerate([img_mask, img_origin, img_rmbg]):
#             if i == 0:
#                 save_name = img_name + '_mask' + '.png'
#             elif i == 1:
#                 save_name = img_name + '.png'
#             elif i == 2:
#                 save_name = img_name + '_rmbg' + '.png'

#             save_path = dir_save.joinpath(save_name)
#             Image.fromarray(img_).save(save_path)

#         print(idx, img_name, 'saved')

#     except FileNotFoundError as err:
#         err_name.append(img_name)
#         print(err)


# if file == 'file_for_rmbg':
#     error_name = ['GEO104_CARS0374_-_cropped',
#                   'GEO125_SJTT2145_1_2_male_cropped', 
#                   'GEO125_SJTT2204_1_2_male_cropped']
#     correct_name = ['GEO104_CARS0374 -_cropped',
#                   'GEO125_SJTT2145_1 2_male_cropped', 
#                   'GEO125_SJTT2204_1 2_male_cropped']
# elif file == 'rgb_background_for_fill':
#     error_name = ['GEO026_SJTT2158_1_2_female_cropped',
#                   'GEO125_SJTT2144_1_2_female_cropped',
#                   'GEO125_SJTT2230_1_2_female_cropped',
#                   'GEO125_SJTT2231_1_2_male_cropped',
#                   'GEO125_SJTT2238_1_2_male_cropped',
#                   'Not_id_yet_CARS1710_cropped (1)',
#                   'URA02_SJTT0747_1__male_cropped']
#     correct_name = ['GEO026_SJTT2158_1 2_female_cropped',
#                   'GEO125_SJTT2144_1 2_female_cropped',
#                   'GEO125_SJTT2230_1 2_female_cropped',
#                   'GEO125_SJTT2231_1 2_male_cropped',
#                   'GEO125_SJTT2238_1 2_male_cropped',
#                   'Not_id_yet_CARS1710_cropped',
#                   'URA02_SJTT0747_1 _male_cropped']

# for idx, name in  enumerate(correct_name):
#     path = dir_origin.joinpath(name + '.png')
#     img = io.imread(path)
#     save_name = name + '.png'
#     save_path = dir_save.joinpath(save_name)
#     Image.fromarray(img).save(save_name)
#     print(idx, name, 'saved')

# ===============================================================================================

# convert from MS Paint3D_rmbg_inverse
# 處理小畫家3D 魔術選取、去背的圖片(圖檔為標本主體被移除、僅保留背景的圖片)
# 處理流程 : 
# 讀取僅有背景(mask)的圖檔 
# > 以灰階讀取，得到主體為白色(== 255)、背景非白(!=255)的影像 
# >  將非白的畫素全部轉為黑色(0)，即得到mask

# dir_origin = Path('../crop/origin')
# imgs_origin = list(dir_origin .glob('*.png'))

# # dir_mask = './bk_mask_manul/'
# dir_mask = Path('data/label_waiting_postprocess/mask_waitinting_for_posrprocess/for_smart_rmbg/mask')
# imgs_mask = list(dir_mask.glob('*.png'))
# print(len(imgs_mask))

# dir_save = dir_mask.joinpath('mask_convert')
# dir_save.mkdir(exist_ok=True)

# err_name = []
# for idx, path in enumerate(imgs_mask):
#     try:
#         # img_mask_ = io.imread(path)  # (256, 256, 4)
#         img_mask_ = io.imread(path, as_gray=True)
#         img_mask_ = (img_mask_*255).astype('uint8')
#         img_name = path.stem

#         # reading  alpha channel as mask, background == 0(mask), target(moth) == 255
#         # mask_ = img_mask_[..., 3]  # (256, 256)

#         # convert and inverse all non white(!=255) to black(0)
#         # img_mask = np.where(mask_ == 0, 0, 255).astype('uint8')
#         img_mask = np.where(img_mask_ == 255, 255, 0).astype('uint8')
        
#         # loading original img by img_name

#         path_origin = dir_origin.joinpath(img_name + '.png')
#         img_origin = io.imread(path_origin)

#         # save img_mask and img_origin into dir_mask
#         for i, img_ in enumerate([img_mask, img_origin]):
#             if i == 0:
#                 save_name = img_name + '_mask' + '.png'
#             elif i == 1:
#                 save_name = img_name + '.png'

#             save_path = dir_save.joinpath(save_name)
#             Image.fromarray(img_).save(save_path)

#         print(idx, img_name, 'saved')

#     except FileNotFoundError as err:
#         err_name.append(img_name)
#         print(err)

# -----------------------------------------------------------------------------------------
# masks = []
# for f in file:
#     path = dir_mask + f 
#     img = io.imread(path + '.png') # img.shape (h,w,c)， c = r, g, b, a
#     mask_ = img[..., 3]  # 僅讀取alpha通道。透明的區域會在蛾類標本本身，即數值為0的黑色區域
#     mask = np.where(mask_ == 0, 256, 0).astype('uint8')  # 黑白反轉，將mask反轉為背景(亦即讓背景數值為0)
#     masks.append(mask)
#     save_dir =  os.path.join(dir_mask, 'convert')
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     io.imsave(save_dir + '_mask' + '.png', mask)

# img_name.split('.')

# Image.fromarray(masks[0]).show()  # 檢視影像

# io.imsave(path + '_mask' + '.png', mask)
# img.shape