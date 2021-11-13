import os
import glob
import numpy as np
from pathlib import Path
import matplotlib as plt
import skimage.io as io
from skimage import io, color
import cv2
from PIL import Image, ImageOps
from skimage.transform import resize
import cv2

root = Path('../data')
dir_origin = root.joinpath('origin')
imgs_origin = list(dir_origin.glob('*.png'))
print(len(imgs_origin))
# ===============================================================================================
# convert_mask_from RGB
# ===============================================================================================

# dir_mask = Path('data/label_waiting_postprocess/mask_waitinting_for_posrprocess/for_rgb_background_fill/mask')
# imgs_mask = list(dir_mask.glob('*.png'))
# print(f'file size : {len(imgs_mask)}')

# dir_save = dir_mask.joinpath('mask_convert')
# dir_save.mkdir(exist_ok=True)

# for idx, path in enumerate(imgs_mask):
#     img = io.imread(path, as_gray=True)  # img.shape (h,w,c)， c = r, g, b, a
#     img_name = path.stem.split('_rgb')[0]

#     # for rgb色塊圖，背景已處理為黑色的mask
#     img_ = (img*255).astype('uint8')
#     img_mask = np.where(img_ == 0, 0, 255).astype(np.uint8)

#     # loading original img by img_name
#     path_origin = dir_origin.joinpath(img_name + '.png')
#     img_origin = io.imread(path_origin)

#     # save img_mask and img_origin into dir_mask
#     for count, img_ in enumerate([img_mask, img_origin]):
#         if count == 0:
#             save_name = img_name + '_mask_rgbfill' + '.png'
#         elif count == 1:
#             save_name = img_name + '.png'

#         save_path = dir_save.joinpath(save_name)
#         Image.fromarray(img_).save(save_path)
#         print(idx, img_name, 'saved')

# ==============================================================================================
# convert_from remove.bg
# ===============================================================================================


# file = 'rgb_background_for_fill'  # file_for_rmbg
dir_mask = Path(
    f'data/label_waiting_postprocess/mask_waitinting_for_posrprocess/for_removebg')
imgs_mask = list(dir_mask.glob('*.png'))
print(f'file size : {len(imgs_mask)}')

dir_save = dir_mask.joinpath('mask_convert')
dir_save.mkdir(exist_ok=True)

err_name = []
for idx, path in enumerate(imgs_mask):
    try:
        img_rmbg = io.imread(path)  # (256, 256, 4)
        img_name = path.stem.replace('-removebg-preview', '')

        # reading  alpha channel as mask, background == 0(mask), target(moth) == 255
        mask_ = img_rmbg[..., 3]  # (256, 256)
        # convert all non black(0) to white(255)
        img_mask = np.where(mask_ < 128, 0, 255).astype('uint8')

        save_name = img_name + '_mask_removeweb' + '.png'
        save_path = dir_save.joinpath(save_name)
        io.imsave(save_path, img_mask)
        print(idx, img_name, 'saved')

        # loading original img by img_name
        path_origin = dir_origin.joinpath(img_name + '.png')
        img_origin = io.imread(path_origin)

    except FileNotFoundError as err:
        err_name.append(img_name)
        print(err, idx, img_name)

# check error name
print(err_name)

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

# ===============================================================================================
# convert from MS Paint3D_rmbg_inverse
# ===============================================================================================
# 處理小畫家3D 魔術選取、去背的圖片(圖檔為標本主體被移除、僅保留背景的圖片)
# 處理流程 :
# 讀取僅有背景(mask)的圖檔
# > 以灰階讀取，得到主體為白色(== 255)、背景非白(!=255)的影像
# >  將非白的畫素全部轉為黑色(0)，即得到mask


# ## dir_mask = './bk_mask_manul/'
# dir_mask = Path('data/label_waiting_postprocess/mask_waitinting_for_posrprocess/for_smart_rmbg/mask')
# imgs_mask = list(dir_mask.glob('*.png'))
# print(len(imgs_mask))

# dir_save = dir_mask.joinpath('mask_convert')
# dir_save.mkdir(parents=True, exist_ok=True)

# err_name = []
# for idx, path in enumerate(imgs_mask):
#     try:
#         # img_mask_ = io.imread(path)  # (256, 256, 4)
#         img_mask_ = io.imread(path, as_gray=True)
#         img_mask_ = (img_mask_*255).astype('uint8')
#         img_name = path.stem.split('_cropped')[0]

#         # reading  alpha channel as mask, background == 0(mask), target(moth) == 255
#         # mask_ = img_mask_[..., 3]  # (256, 256)

#         # convert and inverse all non white(!=255) to black(0)
#         img_mask = np.where(img_mask_ == 255, 0, 255).astype('uint8')

#         # save img_mask and img_origin into dir_mask
#         save_name = img_name + '_cropped' + '_mask_painter3d' + '.png'
#         save_path = dir_save.joinpath(save_name)
#         io.imsave(save_path, img_mask)
#         print(idx, img_name, 'saved')

#         ## loading original img by img_name
#         path_origin = dir_origin.joinpath(img_name + '_cropped' + '.png')
#         img_origin = io.imread(path_origin)

#     except FileNotFoundError as err:
#         err_name.append(img_name)
#         print(err)

# ## check error name
# print(err_name)

# ============================================================================================
# 將所有mask [0 or 255], uint8, 單通道(channel=0)的，結合原圖、產出背景為藍色的去背影像
# ============================================================================================


## dir_mask = './bk_mask_manul/'
dir_mask = Path(
    'data\label_waiting_postprocess\mask_waitinting_for_posrprocess\mask_for_rmbg')

imgs_mask = list(dir_mask.glob('*.png'))
print(len(imgs_mask))
# imgs_mask = [path for path in dir_mask.iterdir() if "_mask" in path.stem]

dir_save = dir_mask.joinpath('mask_rmbg')
dir_save.mkdir(parents=True, exist_ok=True)


def img_rmbg_fill(black_mask: np.ndarray, img: np.ndarray, color: str = 'blue'):
    '''
    color:: 'blue', 'black'.
    black_mask:: shape=(h,w,c), dtype=uint8.
    img:: shape=(h,w,c), dtype=uint8.
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


for idx, path in enumerate(imgs_mask):
    # get path
    msk_path = path
    fname = path.stem.split('_cropped')[0] + '_cropped'

    # prepare data to [0,255], channel=3
    img_path = dir_origin.joinpath(fname + '.png')
    origin_img = io.imread(img_path)
    if not origin_img[..., 0].shape == (256, 256):
        # print(f'origin_img need resize: {origin_img.shape} ')
        origin_img = resize(origin_img, (256, 256))
        origin_img_255 = (origin_img)*255
        origin_img = origin_img_255.astype(np.uint8)

    mask = io.imread(msk_path, as_gray=True)

    if mask.shape is not (256, 256):
        mask = resize(mask, (256, 256))
    if not mask.max() > 1:
        mask = mask*255
    mask = mask.astype(np.uint8)
    # Create 3-channel alpha mask (h,w) > (h,w,3)。  np.dstack([mask]*3)
    mask3 = np.stack([mask]*3, axis=2)

    # get image with background removed and fill with specified color
    img_rmbg = img_rmbg_fill(mask3, origin_img, color='blue')

    # save image
    img_rmbg_name = path.stem.replace('mask', 'rmbg')
    save_path_rmbg = dir_save.joinpath(img_rmbg_name + '.png')
    io.imsave(save_path_rmbg, img_rmbg)
    print(idx, img_rmbg_name, 'saved')

    io.imsave(dir_save.joinpath(fname + '.png'), origin_img)
    print(idx, fname, 'saved')


# ============================================================================================
# 將mask做膨脹-侵蝕、邊緣平滑、改善邊緣缺失
# ============================================================================================


dir_mask = Path(
    'data/label_waiting_postprocess/mask_waitinting_for_posrprocess/mask_for_dilate')
imgs_mask = list(dir_mask.glob('*.png'))
print(f'file size : {len(imgs_mask)}')

dir_save = dir_mask.joinpath('mask_convert')
dir_save.mkdir(exist_ok=True)


# mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
# Image.fromarray(img).show()
# _, img = cv2.threshold(img, 127, 255, 0)
# Image.fromarray(img).show()


for idx, path in enumerate(imgs_mask):
    # get path
    msk_path = path
    fname = path.stem.split('_cropped')[0] + '_cropped'

    mask = io.imread(msk_path, as_gray=True)

    # dilate & erode ------------------------------------------------------------
    iterations = 1
    ksize = (3, 3)
    kernel_ELLIPSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=ksize)
    mask = cv2.dilate((mask), kernel_ELLIPSE, iterations=iterations)
    mask = cv2.erode(mask, kernel_ELLIPSE, iterations=(iterations-1))
    # ----------------------------------------------------------------------------

    # smooth ------------------------------------------------------------
    # mask_pil = Image.open(msk_path)
    # mask_smooth = mask_pil.filter(ImageFilter.ModeFilter(size=3))
    # mask = np.array(mask_smooth)
    # img_mask_name = path.stem + '_smooth' + '.png'
    # mask_smooth.save(dir_save.joinpath(img_mask_name+ '_smooth' +'.png'))
    # ----------------------------------------------------------------------------

    # save mask
    img_mask_name = path.stem + '_dilate' + '.png'
    save_path_mask = dir_save.joinpath(img_mask_name)
    io.imsave(save_path_mask, mask)

    print(idx, img_mask_name, 'saved')

    # loading original img by img_name
    # img_path = dir_origin.joinpath(fname + '.png')
    # origin_img = io.imread(img_path)
    # save image
    # io.imsave(dir_save.joinpath(fname + '.png'), origin_img)
    # print(idx, fname, 'saved')


# ============================================================================================
# convert masks rgb(w,h,c)  to grey(w,h)
# ============================================================================================


dir_mask = Path('../data/label_waiting_postprocess/tmp')

dir_mask_tmp = dir_mask.joinpath('masks_tmp')
dir_mask_tmp.mkdir(parents=True, exist_ok=True)

for i, path in enumerate(dir_mask.glob('*.png')):
    mask_name = path.stem.split('_mask')[0]
    mask_ = io.imread(path, as_gray=True)  # (h,w) uint8

    if not mask_.dtype == 'uint8':
        mask_ = (mask_*255).astype(np.uint8)

    mask_bi = np.where(mask_ > 128, 255, 0).astype('uint8')
    save_path = dir_mask_tmp.joinpath(mask_name + '.png')
    io.imsave(save_path, mask_bi)
    print(i, mask_name, 'saved')

dir_img = dir_mask.joinpath('imgs')
dir_img.mkdir(exist_ok=True, parents=True)

# ## prepare dir_img by dir_mask
dir_ori = Path('../data/data_resize_cropped/origin/')
for i, path in enumerate(dir_mask.iterdir()):
    if not path.name.endswith('.png'):
        continue
    img_name = path.stem.split('_mask')[0]
    origin_file = dir_ori.joinpath(img_name + '.png')
    img = io.imread(origin_file)
    save_path = dir_img.joinpath(img_name + '.png')
    io.imsave(save_path, img)
    print(i, img_name, 'saved')

# ============================================================================================
# get masks by img from dir_mask
# ============================================================================================
# images must in  'data/data_for_Sup_train/imgs'

root = Path('../data/data_for_Sup_train')
dir_mask = root.joinpath('masks_211105')
dir_imgs_batch_arg = root.joinpath('imgs_batch_arg')

dir_masks_batch_arg = root.joinpath('masks_batch_arg')
dir_masks_batch_arg.mkdir(exist_ok=True, parents=True)

for path in dir_imgs_batch_arg.iterdir():
    name = path.stem
    path_mask = dir_mask.joinpath(name + '.png')
    mask = io.imread(path_mask)
    io.imsave(dir_masks_batch_arg.joinpath(name + '.png'), mask)
    print(f'{name}.png saved')
