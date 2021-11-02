'''
predict.py有几个注意点
-1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。-
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
4、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
'''
from PIL import Image
from yolo import YOLO
import os

yolo = YOLO()

basedir = '/data/moth/data/downloaded_with_gbif/vott_flat_images_project20210322/vott_flat_images/'
files_ = os.listdir(basedir)
len(files_)

files = []
bboxes_strs = []
for f in files_:
    if not f.lower().endswith('.jpg'):
        continue
    fpath = basedir + f

    image = Image.open(fpath)
    preview, cropped_imgs, bboxes_str = yolo.detect_image(image)
    preview.thumbnail((256, 256))
    preview.save('./predicts640wHalfPenalty/%s' % f)

    for idx_, cim in enumerate(cropped_imgs):
        cim.thumbnail((256, 256))
        cim.save('./predicts640wHalfPenalty/cropped/%s_%d.jpg' % ('.'.join(f.split('.')[:-1]), idx_))

    files.append(fpath)
    bboxes_strs.append(bboxes_str)

import pandas as pd
pd.DataFrame({'file':files, 'bboxes':bboxes_strs}).to_csv('moth_all_bboxes.csv', sep='\t')