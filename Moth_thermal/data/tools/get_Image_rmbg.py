import os
import glob
import numpy as np
from pathlib import Path
import argparse
import skimage.io as io
from skimage import io
from skimage.transform import resize
from PIL import Image

# =======================================================================================================
# def get_args():
parser = argparse.ArgumentParser(
    description='Generate images whih background removed by masks')

# enviroment
parser.add_argument('--gpu', '-g', dest='gpu', default='1')

# data
parser.add_argument('--dir_masks', '-m', dest='masks',
                    default='../data_for_Sup_train/masks')
parser.add_argument('--dir_origin', '-o', dest='origin',
                    default='../origin')
parser.add_argument('--dir_save', '-s', dest='save',
                    default='../../vsc/data/moth_thermal_rmbg_padding_256')
parser.add_argument('--postfix', '-p', dest='postfix', type=str,
                    default='', help="postfix of image file for split out image name")
parser.add_argument('--fillcolor', '-c', dest='color', type=str,
                    default='black', help="'black' or 'blue':: color to fill backgroung")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# =======================================================================================================


dir_origin = Path(args.origin)
paths_origin = list(dir_origin.glob('*.png'))
print(f'Number of images in {str(dir_origin)}: {len(paths_origin)}')

dir_masks = Path(args.masks)
paths_masks = list(dir_masks.glob('*.png'))
print('Number of masks : ', len(paths_masks))

dir_save = Path(args.save)
# dir_save = dir_masks.joinpath('image_rmbg')
dir_save.mkdir(parents=True, exist_ok=True)
print(f'Image_rmbg will save to {str(dir_save)}')

# ============================================================================================
# 將所有mask [0 or 255], uint8, 單通道(channel=0)，結合原圖產出背景的去背影像
# ============================================================================================


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
        img_rmbg_color = ((black_mask/255)*img).astype(np.uint8)

    elif color == 'blue':
        white_mask = (255-black_mask)
        blue_mask = np.zeros_like(white_mask)
        blue_mask[..., 2] = white_mask[..., 2]
        color_mask = blue_mask
        img_rmbg_color = ((black_mask/255)*img + color_mask).astype(np.uint8)

    

    return img_rmbg_color


error_name = {}
for idx, path in enumerate(paths_masks):
    # get path
    msk_path = path
    fname = path.stem.split('_cropped')[0] + '_cropped'

    # prepare data to [0,255], channel=3
    img_path = dir_origin.joinpath(fname + '.png')
    
    try:
        origin_img = io.imread(img_path)
        if not origin_img[..., 0].shape == (256, 256):
            # print(f'origin_img need resize: {origin_img.shape} ')
            origin_img = resize(origin_img, (256, 256))
            origin_img_255 = (origin_img)*255
            origin_img = origin_img_255.astype(np.uint8)
    
        mask = io.imread(msk_path, as_gray=True)
    except Exception as e:
        print(f"Error : {e}")
        error_name[idx] = fname

    if mask.shape is not (256, 256):
        mask = resize(mask, (256, 256))

    if not mask.max() > 1:
        mask = (mask*255).astype('uint8')

    mask = np.where(mask > 128, 255, 0).astype('uint8')
    # Create 3-channel alpha mask (h,w) > (h,w,3)。  np.dstack([mask]*3)
    mask3 = np.stack([mask]*3, axis=2)

    # get image with background removed and fill with specified color
    img_rmbg = img_rmbg_fill(mask3, origin_img, color=args.color)

    # # save image
    # img_rmbg_name = path.stem.replace('mask', 'rmbg')
    img_rmbg_name = path.stem + args.postfix
    save_path_rmbg = dir_save.joinpath(img_rmbg_name + '.png')
    io.imsave(save_path_rmbg, img_rmbg)
    print(idx, img_rmbg_name, 'saved')

    # original image
    # io.imsave(dir_save.joinpath(fname + '.png'), origin_img)
    # print(idx, fname, 'saved')


print('Finished')

if error_name != {}:
    print('Number of problemed files : ', len(error_name))
    for k, v in error_name.items():
        print(k, v)
