import os
import glob
import time
from numpy.lib.function_base import percentile
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from skimage import io
from skimage.transform import resize
import argparse

from skimage.util import dtype

# ===============================================================================================
parser = argparse.ArgumentParser()

# parser.add_argument('--save_dir', default='model/Unsup_rmbg')
parser.add_argument("--file", default="CARS",
                    type=str, help="CARS" or "SJTT" or "CARS_halfcropped")
parser.add_argument("--fill_bg", action='store_true', default=False,
                    help="whether to fill padding with backgroung color")
parser.add_argument("--fill_color", default=(255, 255, 255),
                    type=tuple, help='color to be filled for padding')

args = parser.parse_args()

# ===============================================================================================

file = args.file  # "CARS" or "SJTT" or "CARS_halfcropped"
# file = "CARS_halfcropped"  # "CARS" or "SJTT" or "CARS_halfcropped"

if file == "CARS":
    basedir = f'../data_raw/{file}/'
elif file == "SJTT":
    basedir = f'../data_resize/{file}/'
elif file == "CARS_halfcropped":
    basedir = f'../crop/{file}/'

files = [path for path in glob.glob(f'{basedir}**/*', recursive=True)
         if os.path.splitext(path)[1].lower() in ['.jpg', '.jpeg', '.png']]  # path.split('.')[-1]
print(f'Prepare data :　{basedir}')
print('data size : ', len(files))


file_bboxes = f"moth_all_bboxes_{file}"
print(f'{file_bboxes}.csv readed ')
bboxes_df = pd.read_csv(f'{file_bboxes}.csv', index_col=0)

# 讀取問題檔案名稱
prob_df = pd.read_csv(f'../crop/problemed.csv', index_col=0)
len(list(prob_df.img))  # 12


# 依據yolo裁切出的bboxes座標點，擴大裁切範圍並輸出為指定大小，提供後續yolo重新偵測範圍
# bboxes : (left, top, right, bottom)

def restrict_boundry(bboxes):
    for i in [0, 1]:
        # reset left and top boundry
        bboxes[i] = 0 if bboxes[i] < 0 else bboxes[i]
    # reset right boundry
    bboxes[2] = img.shape[1] if bboxes[2] > img.shape[1] else bboxes[2]
    # reset bottom boundry
    bboxes[3] = img.shape[0] if bboxes[3] > img.shape[0] else bboxes[3]
    return bboxes


def resize_with_padding(img, expected_size=(256, 256), fill=(255, 255, 255)):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width -
               pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding, fill)


# def get_bgcolor(img, topk_color=0.003):

#     img_ = np.array(img) if not isinstance(img, np.ndarray) else img
#     slice = int(img_.reshape(-1, 3).shape[0]*topk_color)
#     max_color = np.sort(img_.reshape(-1, 3))[:slice]

#     bg_color = np.mean(max_color, axis=0).astype(np.int16)
#     idx = 0
#     while (np.std(max_color) < 5) == 0:
#         rescale = int(max_color.shape[0]*0.9)
#         max_color = max_color[:rescale].astype(np.int16)
#         bg_color = np.mean(max_color[:rescale], axis=0).astype(np.int16)
#         # print('\t', idx, np.std(max_color, axis=0),  max_color.shape, bg_color)

#         if max_color.shape[0] < slice*0.2:
#             break
#         idx += 1
#     return tuple(bg_color)


def get_bgcolor(img, pixel_size=20):
    img_ = np.array(img) if not isinstance(img, np.ndarray) else img
    h, w = img_.shape[:2]
    center_top = img_[10:30, int(w*0.5-pixel_size/2): int(w*0.5+pixel_size/2)]
    bottom_left = img_[-10-pixel_size:-10, : pixel_size]
    bottom_right = img_[-10-pixel_size:-10, -pixel_size:]

    pixels = np.concatenate((center_top.reshape(-1,3), bottom_left.reshape(-1,3), bottom_right.reshape(-1,3)))
    pixels_filtered = np.asarray(
        [row for row in list(pixels)
        if np.array(row).mean() > 255*0.6 and np.array(row).std() < 10]
    )

    bg_color = np.mean(pixels_filtered, axis=0).astype(np.int16)
    return tuple(bg_color)

# for resize bboxex range
scale_w = 0.05 if file == "SJTT" else 0.1
scale_ht, scale_hb = (0.0, 0.0) if file == "SJTT" else (0.1, 0.1)

for index, row in bboxes_df.dropna().iterrows():

    fpath = row.file
    f = os.path.split(fpath)[-1].split('.')[0]

    # 跳過問題檔案名單
    if f in list(prob_df.img):
        print(f"{f} will not be cropped")
        continue

    img = io.imread(fpath)          # 採用skimage讀入純陣列資料格式，避免讀取到EXIF資訊

    # crop depend by bbox with range expand-----------------------------------------------------------
    bboxes_ = np.asarray([float(i) for i in row.bboxes.split(',')])
    # 改變原本框選範圍

    w = bboxes_[2] - bboxes_[0]
    h = bboxes_[3] - bboxes_[1]
    x = bboxes_[0] + w/2
    y = bboxes_[1] + h/2

    bboxes = np.asarray([
        x - 0.5*(w * (1 + scale_w)),       # left
        y - 0.5*(h * (1 + scale_ht)),      # top
        x + 0.5*(w * (1 + scale_w)),       # right
        y + 0.5*(h * (1 + scale_hb))       # bottom
    ])

    image = Image.fromarray(img)
    bboxes_reset = restrict_boundry(bboxes)
    image_cropped = image.crop(bboxes_reset)

    # resize and padding------------------------------------------------------------------------------------------------
    # ratio = (bboxes[3]-bboxes[1]) / (bboxes[2]-bboxes[0])   # H/W
    # width = 256
    # height = int(width*ratio)
    # image_resized = image_cropped.resize(
    #     (width, height), resample=Image.LANCZOS)  # resized

    fill = get_bgcolor(
        image_cropped, pixel_size=20) if args.fill_bg else args.fill_color

    image_resized_with_padding = resize_with_padding(
        image_cropped, (256, 256), fill=fill)

    # ------------------------------------------------------------------------------------------------

    save_dir = f'../crop/{file}_cropped256_paddingbg'
    if not os.path.exists(f'{save_dir}'):
        os.makedirs(f'{save_dir}')
    # image_resized.save(f'{save_dir}/{f}_cropped.png')
    image_resized_with_padding.save(f'{save_dir}/{f}_cropped.png')

    print(
        f"{100*(index+1)/len(bboxes_df.dropna()):.2f}%, {save_dir}/{f}_cropped.png' saved", end="\r")

print(f"\n{save_dir} Cropping Finished\t")

# -----------------------------------------------------------------------------------------
# 處理bboxes缺值的資料 。處理策略:
# 抓出缺值index後，將圖片裁成左、右半邊，分別存入(蟲體位置有左有右)
# -----------------------------------------------------------------------------------------


assert file == 'CARS' and file_bboxes == 'moth_all_bboxes_CARS'

index_nan = bboxes_df[bboxes_df.bboxes.isnull()].index.values  # 　12筆
for index, row in bboxes_df.iloc[index_nan].iterrows():
    fpath = row.file
    f = os.path.split(fpath)[-1].split('.')[0]
    img = io.imread(fpath)          # 採用skimage讀入純陣列資料格式，避免讀取到EXIF資訊
    image = Image.fromarray(img)

    width, height = image.size
    # coordinate : (left, upper, right, lower)
    left_side, right_side = (
        0, 0, 0+width, 0+height), (width, 0, width, 0+height)
    # image_left, image_right = image.crop(left_side), image.crop(right_side)

    save_dir = f'../crop/{file}_halfcropped'
    if not os.path.exists(f'{save_dir}'):
        os.makedirs(f'{save_dir}')
    image.crop(left_side).save(f'{save_dir}/{f}_left_side.png')
    image.crop(right_side).save(f'{save_dir}/{f}_right_side.png')
print("NonBboxes data Cropping Finished")


# -----------------------------------------------------------------------------------------
# 刪除手動篩選有問題的資料
# -----------------------------------------------------------------------------------------

# prob_df = pd.read_csv(f'../crop/problemed.csv', index_col=0)
# path_SJTT = glob.glob('../crop/SJTT_cropped256_120up/**/*', recursive=True)
# path_CARS = glob.glob('../crop/CARS_cropped256_120up/**/*', recursive=True)
# pathes = path_SJTT + path_CARS
# len(path_CARS)
# len(path_SJTT)

# files = [path for path in pathes
#          if os.path.splitext(path)[1].lower() in ['.png', '.jpeg', '.png']]  # path.split('.')[-1]
# print('data size : ', len(files))  # 4554

# save_dir = f'../problem'
# if not os.path.exists(f'{save_dir}'):
#     os.makedirs(f'{save_dir}')

# len(list(prob_df.img)) # 12

# delete = []
# delete_img = []
# for path in files:
#     img_name = path.split(os.sep)[-1].split('.')[0].split('_cropped')[0]
#     if img_name in list(prob_df.img):
#         print(img_name)
#         delete.append(path)
#         delete_img.append(img_name)
#         img  = io.imread(path)
#         io.imsave(f'{save_dir}/{img_name}.png', img)


# =============================================================================================================

# img_names = [
#     'PYR02_SJTT1267_1_male',
#     'CARS1581',
#     'GEO062_CARS0383',
#     'GEO143_CARS1195',
#     'GEO156_CARS0658',
#     'GEO195_CARS1530',
#     'NOC072_CARS0456',
#     'NOC163_CARS0984',
#     'NOC136_CARS0391',
#     'NOC231_CARS2055',
#     'NOC265_CARS2011',
#     'Not_id_yet_CARS0796'
# ]

# img_names = [
#     'GEO195_CARS1530'
# ]
