'''
predict.py有几个注意点
-1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。-
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
4、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
'''
import pandas as pd
from PIL import Image
from skimage import io
from yolo import YOLO
import os
import glob
import time
import argparse


# ===============================================================================================
parser = argparse.ArgumentParser(
    description='predict bounding box by yolo'
)
parser.add_argument("--file", default='SJRS',
                    type=str, help='"CATT", "CARS" or "SJTT", "SJRS" or "SJRS_halfcropped", "CARS_halfcropped"')
parser.add_argument("--data_root", '-r', default='../data',
                    type=str, help='where the data directory store')
parser.add_argument("--basedir", '-dir', default='',
                    type=str, help='where the data directory to predict. If setted, setting of "--file" and "--data_root" will be override')
parser.add_argument('--start_idx', default=0, type=int,
                    help="Manual epoch number (useful on restarts)")
parser.add_argument('--end_idx', default=-1, type=int,
                    help="Manual epoch number (useful on restarts)")

args = parser.parse_args()

# ===============================================================================================

yolo = YOLO()

file = args.file
data_root = args.data_root

if args.basedir:
    basedir = args.basedir
    basedir_ = os.path.dirname(basedir)
    file = os.path.split(basedir_)[-1]
else:
    if file.endswith("RS"):
        basedir = f'{data_root}/data_raw/{file}/'
    elif file.endswith("TT"):
        basedir = f'{data_root}/data_resize/{file}/'
    elif file.endswith("halfcropped"):
        basedir = f'{data_root}/data_resize_cropped/{file}/'


files_ = [path for path in glob.glob(f'{basedir}/**/*', recursive=True)
          if os.path.splitext(path)[1].lower() in ['.jpg', '.jpeg', '.png']]  # path.split('.')[-1]
print(f'Prepare data :　{basedir}')
print('data size : ', len(files_))

# ----------------------------------------------------------------------------------------------
# 處理切半的資料: "*_halfcropped"
# ----------------------------------------------------------------------------------------------
# if file.endswith("halfcropped"):
#     path = f'{data_root}/data_resize_cropped/{file}_specimen.txt'
#     files_halfcrop = []
#     with open(path) as f:
#         for line in f.readlines():
#             path = basedir + line.strip('\n')+'.jpg'
#             files_halfcrop.append(path)
#     files_halfcrop
#     print('data size for files_halfcrop: ', len(files_halfcrop))
#     files_ = files_halfcrop
# ---------------------------------------------------------------------------------------------

save_dir = f'{data_root}/data_resize_cropped/yolo_crop/{file}'
if not os.path.exists(f'{save_dir}/cropped'):
    os.makedirs(f'{save_dir}/cropped')
print(f'save_dir :　{save_dir}')


start = args.start_idx
end = len(files_) if args.end_idx == -1 else args.end_idx
print(f'Process range from [{start} : {end}]')


files = []
bboxes_strs = []
start_time = time.time()
for index, fpath in enumerate(files_[start:end]):

    # fpath = basedir + f
    f = fpath.split('/')[-1]

    img = io.imread(fpath)          # 採用skimage讀入純陣列資料格式，避免讀取到EXIF資訊
    image = Image.fromarray(img)
    # image = Image.open(fpath)
    preview, cropped_imgs, bboxes_str = yolo.detect_image(image)
    preview.thumbnail((256, 256))
    # preview.save('./predicts640wHalfPenalty/%s' % f)
    preview.save(f'{save_dir}/{f}')

    for idx_, cim in enumerate(cropped_imgs):
        cim.thumbnail((256, 256))
        # cim.save('./predicts640wHalfPenalty/cropped/%s_%d.jpg' % ('.'.join(f.split('.')[:-1]), idx_))
        cim.convert('RGB').save(
            f"{save_dir}/cropped/{'.'.join(f.split('.')[:-1])}_{idx_}.jpg")

    files.append(fpath)
    bboxes_strs.append(bboxes_str)
    time_passed = time.time()-start_time
    print(f"i: {index:4d}, {100*(index)/len(files_):.2f}% \
        | Time : {time_passed//60:.0f}m, {time_passed%60:.0f}s | {fpath:<s}   ", end="\r")

pd.DataFrame({'file': files, 'bboxes': bboxes_strs}
             ).to_csv(f'moth_all_bboxes_{file}.csv', sep=',')
