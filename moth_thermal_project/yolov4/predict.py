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

yolo = YOLO()

# file = "CARS"  # "CARS" or "SJTT" or "CARS_halfcropped"
file = "CARS_halfcropped"  # "CARS" or "SJTT" or "CARS_halfcropped"

if file == "CARS":
    basedir = f'../data_raw/{file}/' 
elif file == "SJTT": 
    basedir = f'../data_resize/{file}/'
elif file == "CARS_halfcropped":
    basedir = f'../crop/{file}/'

files_ = [path for path in glob.glob(f'{basedir}**/*', recursive=True)
         if os.path.splitext(path)[1].lower() in ['.jpg', '.jpeg', '.png']]  # path.split('.')[-1]
print(f'Prepare data :　{basedir}')
print('data size : ', len(files_))  

# ----------------------------------------------------------------------------------------------
# 處理切半的資料 "CARS_halfcropped"
# ----------------------------------------------------------------------------------------------
if file == "CARS_halfcropped":
    path = '../crop/CARS_halfcropped_specimen.txt'
    files_halfcrop = []
    with open(path) as f:
        for line in f.readlines():
            path = basedir + line.strip('\n')+'.jpg'
            files_halfcrop.append(path)
    files_halfcrop
    print('data size for files_halfcrop: ', len(files_halfcrop))
    files_ = files_halfcrop
# ---------------------------------------------------------------------------------------------

save_dir = f'../crop/yolo_crop/{file}'
if not os.path.exists(f'{save_dir}/cropped'):
    os.makedirs(f'{save_dir}/cropped')
print(f'save_dir :　{save_dir}')

files = []
bboxes_strs = []
start_time = time.time()
for index, fpath in enumerate(files_):
    if not fpath.lower().endswith('.jpg'):
        continue
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
        cim.save(
            f"{save_dir}/cropped/{'.'.join(f.split('.')[:-1])}_{idx_}.jpg")

    files.append(fpath)
    bboxes_strs.append(bboxes_str)
    time_passed = time.time()-start_time
    print(f"i: {index+1:4d}, {100*(index+1)/len(files_):.2f}% \
        | Time : {time_passed//60:.0f}m, {time_passed%60:.0f}s | {fpath:<s}   ", end="\r")

pd.DataFrame({'file': files, 'bboxes': bboxes_strs}
             ).to_csv(f'moth_all_bboxes_{file}.csv', sep=',')

 