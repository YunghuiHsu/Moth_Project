import os
import glob
import time
from pathlib import Path
from numpy.lib.function_base import percentile
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from skimage import io
from skimage.transform import resize
import argparse
from utils.utils import get_bgcolor, restrict_boundry, resize_with_padding


# ===============================================================================================
parser = argparse.ArgumentParser()

# parser.add_argument('--save_dir', default='model/Unsup_rmbg')
parser.add_argument("--file", default="SJRS",
                    type=str, help='"CATT", "CARS" or "SJTT", "SJRS" or "SJRS_halfcropped", "CARS_halfcropped"')
parser.add_argument("--fill_bg", action='store_true', default=False,
                    help="whether to fill padding with backgroung color")
parser.add_argument("--fill_color", default=(255, 255, 255),
                    type=tuple, help='color to be filled for padding')
parser.add_argument("--data_root", '-d', default='../data',
                    type=str, help='where the data directory store')
# parser.add_argument('--halfcrop', '-hc', action='store_true',
#                     help='whether to crop data if bboxes is null. when data ')
parser.add_argument("--basedir", '-dir', default='',
                    type=str, help='where the data directory to predict')
parser.add_argument('--start_idx', default=0, type=int,
                    help="Manual epoch number (useful on restarts)")
parser.add_argument('--end_idx', default=-1, type=int,
                    help="Manual epoch number (useful on restarts)")

args = parser.parse_args()
# ===============================================================================================

file = Path(args.file)
data_root = Path(args.data_root)


if args.basedir:
    basedir = args.basedir
    file = os.path.split(basedir)[-1]
else:
    if file.endswith("RS"):
        basedir = f'{data_root}/data_raw/{file}/'
    elif file.endswith("TT"):
        basedir = f'{data_root}/data_resize/{file}/'
    elif file.endswith("halfcropped"):
        basedir = f'{data_root}/data_resize_cropped/{file}/'

save_dir = Path(f'{data_root}/data_resize_cropped/{file}_cropped256_paddingbg')
if not os.path.exists(f'{save_dir}'):
    os.makedirs(f'{save_dir}')

files = [path for path in glob.glob(f'{basedir}**/*', recursive=True)
         if os.path.splitext(path)[1].lower() in ['.jpg', '.jpeg', '.png']]  # path.split('.')[-1]
print(f'Prepare data : {basedir}')
print(f'Data size in {args.basedir} : {len(files):,d}')

try:
    file_bboxes = f"moth_all_bboxes_{file}"
    bboxes_df = pd.read_csv(f'{file_bboxes}.csv', index_col=0)
    print(f'{file_bboxes}.csv loaded ')
except Exception as e:
    print(e)
    print('You should put bboxes file in here')

bboxes_df_drop = bboxes_df.dropna()
print(f'Data size after dropna in {file_bboxes}: {len(bboxes_df_drop):,d}')

start = args.start_idx
end = len(bboxes_df_drop) if args.end_idx == -1 else args.end_idx
print(f'Idx Process range from [{start} : {end}]')

# loading problemed file, it will not be output
try:
    problemed_df = pd.read_csv(f'{data_root}/problemed.csv', index_col=0)
    problemed_list = list(problemed_df.img)
    len(problemed_list)  # 12
except Exception as e:
    print(e)
    problemed_list = []


# ================================================================================================
# start crop by bboxes
# ================================================================================================


# for resize bbox range
scale_w = 0.1 if not file.endswith("RS") else 0.2
scale_ht, scale_hb = (0.1, 0.1) if not file.endswith("RS") else (0.2, 0.2)

for idx, row in bboxes_df_drop[start:end].iterrows():

    fpath = row.file
    fname = os.path.split(fpath)[-1].split('.')[0]
    if file.endswith("halfcropped"):
        fname = fname.split('right')[0].split('left')[0]

    # 跳過問題檔案名單
    if problemed_list:
        if fname in problemed_list:
            print(f"{fname} will not be cropped")
            continue

    # try:
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
    bboxes_reset = restrict_boundry(bboxes, img)
    image_cropped = image.crop(bboxes_reset)

    # resize and padding------------------------------------------------------------------------------------------------
    # ratio = (bboxes[3]-bboxes[1]) / (bboxes[2]-bboxes[0])   # H/W
    # width = 256
    # height = int(width*ratio)
    # image_resized = image_cropped.resize(
    #     (width, height), resample=Image.LANCZOS)  # resized

    bg_color = get_bgcolor(
        image_cropped, pixel_size=10) if args.fill_bg else args.fill_color

    image_resized_with_padding = resize_with_padding(
        image_cropped, (256, 256), fill=bg_color)

    image_resized_with_padding.save(f'{save_dir}/{fname}_cropped.png')

    print(
        f"[{idx}][{len(bboxes_df_drop):,d}] |{100*(idx+1)/len(bboxes_df_drop):.2f}%, {save_dir}/{fname}_cropped.png' saved", end="\r")

    # except FileNotFoundError as e:
    #     continue
    # except Exception as e:
    #     print(e)
    #     break


print(f"\n{save_dir} Cropping Finished\t")

# -----------------------------------------------------------------------------------------
# 處理bboxes缺值的資料 。
# 處理策略:
# 抓出缺值index後，將圖片裁成左、右半邊，分別存入(蟲體位置有左有右)
# -----------------------------------------------------------------------------------------
assert file.endswith('RS') and file_bboxes.endswith('RS'), 'Finished'
print('Starting halfcrop data which YOLO detect failure')

save_dir = f'{data_root}/data_resize_cropped/{file}_halfcropped'
if not os.path.exists(f'{save_dir}'):
    os.makedirs(f'{save_dir}')
    print(f'{save_dir} established')

index_nan = bboxes_df[bboxes_df.bboxes.isnull()].index.values
print(f'Number of data has no bboxes :{len(index_nan)}')

for idx, row in bboxes_df.iloc[index_nan].iterrows():
    fpath = row.file
    file_name = os.path.split(fpath)[-1].split('.')[0]
    img = io.imread(fpath)          # 採用skimage讀入純陣列資料格式，避免讀取到EXIF資訊
    image = Image.fromarray(img)

    width, height = image.size
    # coordinate : (left, upper, right, lower)
    left_side = (0, 0, 0 + width/2, 0 + height)
    right_side = (width/2, 0, width, 0 + height)
    # image_left, image_right = image.crop(left_side), image.crop(right_side)

    image.crop(left_side).save(f'{save_dir}/{file_name}_left_side.png')
    image.crop(right_side).save(f'{save_dir}/{file_name}_right_side.png')
    print(f'{idx:4d}, halfcrop of {file_name}, saved')

print("Bboxes_null data Cropping Finished")


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
