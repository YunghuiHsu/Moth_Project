import argparse
import os
import glob
import numpy as np
from pathlib import Path
import matplotlib as plt
from skimage import io, color
from skimage.transform import resize
from PIL import Image, ImageOps
import cv2


# root = Path('../../data')


# =======================================================================================================
# def get_args():
parser = argparse.ArgumentParser(
    description='Tools to preprocess all kink of masks')

# enviroment
parser.add_argument('--gpu', '-g', dest='gpu', default='0')

# data
parser.add_argument('--get_grey', '-grey',
                    action='store_true', default=False, 
                    help='convert masks from rgb(w,h,c) to grey(w,h)')
parser.add_argument('--dir_masks_grey', '-m_grey', 
                    # dest='MASKDIR_grey',
                    default='../label_waiting_postprocess/mask_to_grey',
                    help='convert masks from rgb(w,h,c) to grey(w,h)')

parser.add_argument('--convert_removebg', '-removebg', 
                    action='store_true', default=False, 
                    help='convert masks from remove.bg imgs')
parser.add_argument('--dir_masks_removebg', '-m_bg', 
                    # dest='MASKDIR_removebg',
                    default='../label_waiting_postprocess/mask_removebg',
                    help='convert masks from remove.bg imgs')

parser.add_argument('--convert_rgb_fill', '-rgb_fill', 
                    action='store_true', default=False, 
                    help='convert masks from UnSupervied rgb_background_fill imgs')
parser.add_argument('--dir_masks_rgb_fill', '-m_rgb_fill', 
                    # dest='MASKDIR_rgb_fill',
                    default='../label_waiting_postprocess/masks_rgb',
                    help='convert masks from UnSupervied rgb_background_fill imgs')

parser.add_argument('--convert_pant3d', '-pant3d', 
                    action='store_true', default=False, 
                    help='convert masks from Microsoft Paint3D_rmbg_inverse imgs')
parser.add_argument('--dir_masks_pant3d', '-m_pant3d', 
                    # dest='MASKDIR_pant3d',
                    default='../label_waiting_postprocess/masks_pant3d',
                    help='convert masks from Microsoft Paint3D_rmbg_inverse imgs')

parser.add_argument('--convert_bgfill', '-bgfill', 
                    action='store_true', default=False, 
                    help='Combine original imgs with mask to get background removed and filled imgs')
parser.add_argument('--dir_masks_bgfill', '-m_bgfill', 
                    # dest='MASKDIR_bgfill',
                    default='../label_waiting_postprocess/mask_bgfill',
                    help='Combine original imgs with mask to get background removed and filled imgs')

parser.add_argument('--get_masks_from_imgs', '-get_masks', 
                    action='store_true', default=False, 
                    help='get masks by img from dir_mask')
parser.add_argument('--dir_masks_from_imgs', '-get_masks_imgs',
                    default='../label_waiting_postprocess/imgs_batch_arg',
                    help='get masks by img from dir_mask')


parser.add_argument('--dir_origin', '-o', 
                    # dest='ORIGINDIR',
                    default='../origin')

parser.add_argument('--save_original_img', '-ori_imgs', 
                    action='store_true', default=False, 
                    help='Whether to save original imgs by masks')

parser.add_argument('--fillcolor', '-c', dest='color', type=str,
                    default='blue', help="'black' or 'blue':: color to fill backgroung")

# parser.add_argument('--dir_save', '-s', dest='SAVEDIR',
#                     default='')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(args)

# dir_origin = root.joinpath('origin')
dir_origin = Path(args.dir_origin)
imgs_origin = list(dir_origin.glob('*.png'))
print(f'Number of imgs in {dir_origin} : {len(imgs_origin):,d}')

if args.get_grey: 
    dir_mask = Path(args.dir_masks_grey)
elif args.convert_removebg:
    dir_mask = Path(args.dir_masks_removebg)
elif args.convert_rgb_fill:
    dir_mask = Path(args.dir_masks_rgb_fill)
elif args.convert_pant3d:
    dir_mask = Path(args.dir_masks_pant3d)
elif args.convert_bgfill:
    dir_mask = Path(args.dir_masks_bgfill)
    
path_mask = list(dir_mask.glob('*.png'))
print(f'Number of masks in {dir_mask} : {len(path_mask):,d}')

dir_save = dir_mask.joinpath('mask_convert')
dir_save.mkdir(parents=True, exist_ok=True)


# =======================================================================================================

# ============================================================================================
# convert masks from rgb(w,h,c) to grey(w,h)
# ============================================================================================

if args.get_grey: 

    for i, path in enumerate(path_mask):
        mask_name = path.stem.split('_cropped')[0] + '_cropped'
        mask_ = io.imread(path, as_gray=True)  # (h,w) uint8

        if not mask_.dtype == 'uint8':
            mask_ = (mask_*255).astype(np.uint8)

        mask_bi = np.where(mask_ > 128, 255, 0).astype('uint8')
        save_path = dir_save.joinpath(mask_name + '.png')
        io.imsave(save_path, mask_bi)
        print(i, mask_name, 'saved')


# ===============================================================================================
# convert_mask_from RGB
# ===============================================================================================
if args.convert_rgb_fill: 

    for idx, path in enumerate(path_mask):
        img = io.imread(path, as_gray=True)  # img.shape (h,w,c)， c = r, g, b, a
        img_name = path.stem.split('_rgb')[0]

        # for rgb色塊圖，背景已處理為黑色的mask
        img_ = (img*255).astype('uint8')
        img_mask = np.where(img_ == 0, 0, 255).astype(np.uint8)

        # loading original img by img_name
        path_origin = dir_origin.joinpath(img_name + '.png')
        img_origin = io.imread(path_origin)

        # save img_mask and img_origin into dir_mask
        for count, img_ in enumerate([img_mask, img_origin]):
            if count == 0:
                save_name = img_name + '_mask_rgbfill' + '.png'
            elif count == 1:
                save_name = img_name + '.png'

            save_path = dir_save.joinpath(save_name)
            Image.fromarray(img_).save(save_path)
            print(idx, img_name, 'saved')

# ==============================================================================================
# convert_from remove.bg
# ===============================================================================================

if args.convert_removebg: 

    err_name = []
    for idx, path in enumerate(path_mask):
        try:
            img_rmbg = io.imread(path)  # (256, 256, 4)
            img_name = path.stem.replace('-removebg-preview', '')

            # reading  alpha channel as mask, background == 0(mask), target(moth) == 255
            mask_ = img_rmbg[..., 3]  # (256, 256)
            # convert all non black(0) to white(255)
            img_mask = np.where(mask_ < 128, 0, 255).astype('uint8')

            save_name = img_name + '_mask' + '.png'
            save_path = dir_save.joinpath(save_name)
            io.imsave(save_path, img_mask)
            print(idx, img_name, 'saved')

            # loading original img by img_name
            # path_origin = dir_origin.joinpath(img_name + '.png')
            # img_origin = io.imread(path_origin)

        except FileNotFoundError as err:
            err_name.append(img_name)
            with open('err_name.txt', 'w') as f:
                f.write(f'{img_name}\n')
            print(err, idx, img_name)

    # check error name
    print(err_name)


# ===============================================================================================
# convert from MS Paint3D_rmbg_inverse
# ===============================================================================================
# 處理小畫家3D 魔術選取、去背的圖片(圖檔為標本主體被移除、僅保留背景的圖片)
# 處理流程 :
# 讀取僅有背景(mask)的圖檔
# > 以灰階讀取，得到主體為白色(== 255)、背景非白(!=255)的影像
# >  將非白的畫素全部轉為黑色(0)，即得到mask

if args.convert_pant3d: 

    err_name = []
    for idx, path in enumerate(path_mask):
        try:
            # img_mask_ = io.imread(path)  # (256, 256, 4)
            img_mask_ = io.imread(path, as_gray=True)
            img_mask_ = (img_mask_*255).astype('uint8')
            img_name = path.stem.split('_cropped')[0]

            # reading  alpha channel as mask, background == 0(mask), target(moth) == 255
            # mask_ = img_mask_[..., 3]  # (256, 256)

            # convert and inverse all non white(!=255) to black(0)
            img_mask = np.where(img_mask_ == 255, 0, 255).astype('uint8')

            # save img_mask and img_origin into dir_mask
            save_name = img_name + '_cropped' + '_mask_painter3d' + '.png'
            save_path = dir_save.joinpath(save_name)
            io.imsave(save_path, img_mask)
            print(idx, img_name, 'saved')

            ## loading original img by img_name
            path_origin = dir_origin.joinpath(img_name + '_cropped' + '.png')
            img_origin = io.imread(path_origin)

        except FileNotFoundError as err:
            err_name.append(img_name)
            with open('err_name.txt', 'w') as f:
                f.write(f'{img_name}\n')
            print(err)

    ## check error name
    print(err_name)
    
# ============================================================================================
# 依據指定的masks 從origin資料夾讀取原檔名儲存影像 
# ============================================================================================
# save original img
if args.save_original_img: 
    dir_img = dir_mask.joinpath('imgs')
    dir_img.mkdir(exist_ok=True, parents=True)

    # ## prepare dir_img by dir_mask
    # dir_origin = Path('../origin')
    img_origin = [path.stem for path in list(dir_origin.glob('*.png'))]

    for i, path in enumerate(dir_mask.iterdir()):
        if not path.name.endswith('.png'):
            continue
        img_name = path.stem.split('_cropped')[0] + '_cropped'
        origin_file = dir_origin.joinpath(img_name + '.png')
        img = io.imread(origin_file)
        save_path = dir_img.joinpath(img_name + '.png')
        io.imsave(save_path, img)
        print(i, img_name, 'saved')


# ============================================================================================
# 將所有mask [0 or 255], uint8, 單通道(channel=0)的，結合原圖、產出背景為藍色的去背影像
# ============================================================================================

if args.convert_bgfill:

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


    for idx, path in enumerate(path_mask):
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
        img_rmbg = img_rmbg_fill(mask3, origin_img, color=args.fillcolor)

        # # save image
        # img_rmbg_name = path.stem.replace('mask', 'rmbg')
        img_rmbg_name = path.stem + '_rmbg'
        save_path_rmbg = dir_save.joinpath(img_rmbg_name + '.png')
        io.imsave(save_path_rmbg, img_rmbg)
        print(idx, img_rmbg_name, 'saved')

        io.imsave(dir_save.joinpath(fname + '.png'), origin_img)
        print(idx, fname, 'saved')


# ============================================================================================
# get masks by img from dir_mask
# ============================================================================================
# images must in  'data/data_for_Sup_train/imgs'

if args.get_masks_from_imgs:

    dir_mask = Path('../data_for_Sup_train/masks')
    dir_imgs_batch_arg = Path(args.dir_masks_from_imgs)
    path_imgs_batch_arg = list(dir_imgs_batch_arg.glob('*.png'))
    print(f'Number of imgs in {dir_imgs_batch_arg} : {len(path_imgs_batch_arg)}')
    

    dir_masks_batch_arg = dir_imgs_batch_arg.joinpath('masks_batch_arg')
    dir_masks_batch_arg.mkdir(exist_ok=True, parents=True)

    for path in dir_imgs_batch_arg.iterdir():
        name = path.stem
        path_mask = dir_mask.joinpath(name + '.png')
        mask = io.imread(path_mask)
        io.imsave(dir_masks_batch_arg.joinpath(name + '.png'), mask)
        print(f'{name}.png saved')

