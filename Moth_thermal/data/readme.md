
#### data/data_for_Sup_train
 - 放置remove_bg中，Unet監督式訓練(Sup_train_pytorch.py)需要的訓練資料


#### data/tools/

- 小工具，包括:
  - 遮罩轉換、重複檢查、根據mask產出去背影像、
  - 根據標本翅膀部位的hsv做分群產出label(get_imgs_label_byHSV.py)
    - 計算畫面中翅膀位置的 
  - mask後處理(Postprocess.py)
    - 抓取最大輪廓線並進行填補
    - 去除mask最大輪廓線外的雜訊並將輪廓線內的畫素完全填補)
