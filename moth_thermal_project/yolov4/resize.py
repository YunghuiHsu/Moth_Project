import numpy as np
import cv2
from skimage import io
import sys
from pathlib import Path
import os
import time
import datetime
import glob
import re
import argparse


# ===============================================================================================
parser = argparse.ArgumentParser(
    description='resize the images of moth specimen imgs'
)
parser.add_argument("--data_dir", default='../data/data_raw/CATT/',
                    type=str, help="where to load data to be resized")
# parser.add_argument("--data_resize", default='CATT', type=str,
#                     help="which data to resize: 'CATT', 'SJTT' or 'SJTT_Label'")
parser.add_argument("--data_start", default=0, type=int,
                    help="Manual iter number (useful on restarts)")
# parser.add_argument("--data_end", default='', type=int,
#                     help="Manual iter number (useful on restarts)")
parser.add_argument("--save_root", default='../data/data_resize',
                    type=str, help="where to save resized image")
parser.add_argument('--size', default="1200, 800",
                    type=str, help='image size to be resized')
parser.add_argument('--width', default=1200,
                    type=int, help='image width to be resized')
parser.add_argument('--ratio', default=0.5,
                    type=float, help='image size to be resized')
parser.add_argument('--img_type', '-t', dest='img_type',
                    metavar='TYPE', type=str, default='.jpg', help='image type: .png, .jpg ...')
args = parser.parse_args()

# ===============================================================================================

# data sourse:  'CARS': 成大馬來西亞標本、 'SJTT': 多樣性中心四川標本、 'CATT':多樣性中心馬來西亞標本
# 指定resize圖檔要儲存的位置
today = datetime.datetime.now().strftime('%Y%m%d')

data_dir = Path(args.data_dir)
save_dir = Path(args.save_root).joinpath(data_dir.name)
save_dir.mkdir(exist_ok=True, parents=True)

# 抓取所有檔案、遍歷資料夾、先抓i取資料夾路徑名稱+檔案名稱
# def resize 函式、批次resize後，替換檔名(加上科名)、另存新檔


# 抓取所有檔案
# ===========================================================================================================
# 先處理'moth_thermo_20210810'
# - 僅抓取確定科名的資料
# ===========================================================================================================
# 搭配**與 recursive=True取得所有子目錄的資料
# glob 內的regex 使用的是linux shell風格
# image_pathes_ = [path for path in glob.glob(data_dir+'**', recursive=True)
#                  if os.path.splitext(path)[1].lower() in ['.jpg', '.jpeg', '.png']]
# image_pathes_ = [path.replace("JPG", "jpg") for path in image_pathes_]

# image_pathes_ = list(data_dir.glob('**/*.jpg'))
# 多樣性中心副檔名小寫為標本照(_1.jpg)、大寫為標籤照('_2.JPG')
# '._'為隱藏檔
image_pathes_ = [path for path in data_dir.glob('**/*.jpg')
                 if not path.stem.startswith('._') and path.suffix == '.jpg']
print('Number of imgs :　', len(image_pathes_))

# image_SJTT_pathes = [path for path in image_pathes_ if (
#     "重複或problem" not in path) and ("SJTT" in path)]  # 排除"重複或problem"資料夾內的路徑
# print("Total SJTT Imgs not including '重複或problem' : ",
#       len(image_SJTT_pathes))  # 4908


# 將SJTT檔案按照標本、標籤分開
# image_SJTT_specimen, image_SJTT_label, search_none, search_other = [], [], [], []
# for i, path in enumerate(image_SJTT_pathes):
#     #　檔名後方加入_1_或未有任何標示為標本照，若_2_ 以後為標籤照
#     search = re.search(r"(_[0]{0,1}[\d]{1})", path.split(os.sep)[-1])
#     # search is None為檔名中沒有 _digit_ 文字的檔案。
#     if (search is None) or search.groups()[0] == "_1" or search.groups()[0] == "_01":
#         image_SJTT_specimen.append(path)
#     else:
#         image_SJTT_label.append(path)
#     print(i, end="\r")
# # 2465(2462+3) 內含兩個重複檔名; 2443 內含兩個重複檔名
# print(
#     f"SJTT specimen size : {len(image_SJTT_specimen)}, SJTT label size : {len(image_SJTT_label)}")
# # print(f"search_none : {len(search_none)}, search_other : {len(search_other)}")


# 檢視檔名是否有重複
# for img_path in [image_SJTT_specimen, image_SJTT_label, image_CARS_pathes]:
#     imgname = [path.split(os.sep)[-1] for path in img_path]
#     print(f"file name duplicated? {len(imgname) != len(set(imgname))}. ", len(imgname) , len(set(imgname)))
#     if len(imgname) != len(set(imgname)):
#         print([key for key, value in dict(Counter(imgname)).items() if value > 1])

# def resizeImg(imgPath, save_dir, size=(1200, 800)):
#     start_time=time.time()
#     for index in range(len(imgPath)):
#         # img = cv2.imread(image_list[index])  # 為避免讀取中文檔名錯誤，改用skimage，且能正確讀取exif資訊
#         img=io.imread(imgPath[index])
#         img_Name=imgPath[index].split(os.sep)[-1].split(".")[0]
#         img_resize=cv2.resize(img, size, interpolation=cv2.INTER_AREA)
#         path_save=f"{save_dir}{os.sep}{img_Name}.jpg"
#         # cv2.imwrite(path_save, img_resize)
#         io.imsave(path_save, img_resize)
#         time_passed=time.time()-start_time

#         print(f"i: {index:4d}, {100*(index+1)/len(imgPath):.2f}%, Img Size:{len(os.listdir(save_dir)):4d},\
#     | Time : {time_passed//60:.0f}m, {time_passed%60:.0f}s | {path_save:<40s}    ", end="\r")

# ===========================================================================================================
# reseze images
# ===========================================================================================================
# def str_to_tuple(x):
#     return tuple([int(x) for x in x.split(',')])

# size = str_to_tuple(args.size)  # if not dir_resize == 'SJTT' or  else (150, 100)
# path_file=image_SJTT_specimen if dir_resize == 'SJTT' else image_SJTT_label


img = io.imread(image_pathes_[0])
h, w, _ = img.shape
resize = args.width, round(h*args.width/w)

s = args.data_start

print(f"{save_dir} resizing... , Img size: {len(image_pathes_)}. Resize {w, h} to {resize}")

# resizeImg(image_pathes_[s:], save_dir, size=size)
start_time = time.time()
for idx, path in enumerate(image_pathes_[s:]):
    name = path.stem

    img = io.imread(path)
    h, w, _ = img.shape
    resize = args.width, round(h*args.width/w)

    img_resize = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
    path_save = save_dir.joinpath(name + args.img_type)
    io.imsave(path_save, img_resize)
    time_passed = time.time()-start_time

    print(f"i: {idx:4d}, {100*(idx+1)/len(image_pathes_):.2f}%, Img Size:{len(list(save_dir.glob('*.png'))):4d},\
    | Time : {time_passed//60:.0f}m, {time_passed%60:.0f}s | {str(name):<40s}   ", end="\r")


print(f"{save_dir} Img Resize Finished!")


# ----------------------------------------------------------------------------------------------

#　檔案讀取與顯示
# 使用matplotlib
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
# plt.imshow(img[...,::-1])  # BGR to RGB
# plt.show()

# 檢查檔名是否內含中文
# zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
# zh_list = []
# for index in range(len(image_SJTT_specimen)):
#     match = zhPattern.search(image_SJTT_specimen[index])
#     if match:
#         # zh_list.append(image_SJTT_specimen[index].split(os.sep)[-1])
#         zh_list.append(image_SJTT_specimen[index])
# for i, path in enumerate(zh_list):
#     print(i, path)
#     img = io.imread(path)
