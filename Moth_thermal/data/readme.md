
#### data/data_for_Sup_train
 - 放置remove_bg中，Unet監督式訓練(Sup_train_pytorch.py)需要的訓練資料


#### data/tools/

- 放置各種處理資料的小工具，包括:
  - 遮罩轉換、重複檢查、根據mask產出去背影像、
  - 花式取樣
    - 根據標本翅膀部位的hsv做分群產出label(get_imgs_label_byHSV.py)
  - mask後處理(Postprocess.py)

#### data/meta_thermal_rangesize
- meta data清理與對齊
