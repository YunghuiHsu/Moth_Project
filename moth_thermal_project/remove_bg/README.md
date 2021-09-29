# Background Remove for Moth Project
###### tags:`Tool` `Moth project` `image segmentation` 


### 鱗翅目去背工具參考流程: [github/colorful-moth](https://github.com/twcmchang/colorful-moth)


-------------------------------------------------------
### 資料io原則與注意事項：
- 中間影像的轉換格式均避免使用jpg格式壓縮導致細節損失及失真(採用png)
- 進入非監督、監督式去背的影像預先padding成(256, 256)，避免影像變形
- 影像存取採用skimage 直接讀取為numpy陣列，避免讀取exif檔時存取到變形、選轉的影像
-------------------------------------------------------
### 資料說明：
- 將資料按照來源區分，包含：
    - 成大(CARS)
    - 多樣性中心(SJTT)
- 檔案命名規則
    - 注意: 原檔名內含中文可能導致亂碼
        -　[透過 7-Zip 壓縮 ZIP 檔案時須注意中文檔名無法正確解壓縮的問題](https://blog.miniasp.com/post/2020/09/14/7-Zip-with-Chinese-filenames-should-use-UTF-8)
        -　[[分享]Ubuntu解壓縮zip檔亂碼問題快速解法](https://blog.impochun.com/ubuntu-unzip-traditional-chinese-garbled-text-solution/)
    


- 目視篩選過濾品質不佳樣本
    - 翅膀過度殘缺、展翅不全及其他例外
    - 可將檔名寫成.csv或.text檔，之後於批次crop-resize by Bounding Box的流程中排除掉 
--------------------------------------------------------
==========================================================

## 一、物件偵測(object detection)，裁切抓取出主體(蛾類標本)
- 使用針對鱗翅目訓練好的yolov4模型
- 影像讀取一率使用skimage.io
    - 最後的裁切圖可根據yolo跑出的Bounding Box座標點調整裁切範圍
        - yolo讀圖預設是使用PIL，要注意可能有些圖片在作業系統中被轉正過，使用PIL的話會讀取到轉正的EXIF資訊 

### 流程:
1. 先使用原資料跑一遍yolov4
    - 獲得裁切的圖片與Bounding Box座標點
2. 檢視裁切結果
    - 成大(CARS)
        - 由於圖片太小，較有切邊情形發生 
        - 部分標本過小偵測不到
    - 多樣性中心(SJTT)
        - 標籤與圖片分開，在前置作業時可以先用regex先分開
        - 圖片較不乾淨，內含有側面、生殖器、label名稱誤標為標本照等問題
        - 較容易裁切到比例尺與色卡 
3. 處理對策
    - 針對成大(CARS)過小的標本，直接將影像切成左右兩半再去跑yolo，取得Bounding Box
    - 獲得Bounding Box座標點後，直接依據座標點加大選取範圍，來進行裁切獲得影像
        - Bounding Box座標點分別代表: left, top, right, bottom
        - 縮放尺度 scale= 1.1 - 1.2
            - 例如左邊界：- 寬*0.95、右邊界： + 寬*1.05，合計為 1.1倍 
        - 高(上下邊界)的縮放範圍可以比左右再大一些
        - 使用Bounding Box座標點的中心絕對位置(x, y) +- 0.5*寬(高)*縮放比例  


##### Bounding Boxh裁切範圍code
:::spoiler
```python=

scale_w = 0.1  if file == "SJTT" else 0.1       
scale_ht, scale_hb = (0.0, 0.0) if file == "SJTT" else (0.1, 0.1)

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
# bboxes_reset = restrict_boundry(bboxes)
image_cropped = image.crop(bboxes)

```
:::

==========================================================
## 二、去背
- 使用 手動去背+非監督式去背(Unsup_train.py)，產生想要的mask(label)
    - 這邊的label是像素等級的，例如：
        - 標本主題: 類別0
        - 背景(想要去掉的)： 類別1 
- 訓練監督式去背模型(Sup_train.py)
    - 將前一步驟產生的mask(label)作為Y，標本照片做為X
    - 訓練模型至去背結果與輸入的Y(mask)一致
- 使用訓練好的去背模型(Sup_predict_rmbg.py)進行去背，獲得MASK
- 後處理(Postprocess.ipynb)
    - 採用前一步驟產出的MASK為基底，搭配其他傳統的電腦視覺技術如crf, find_cntr等優化MASK的輪廓
        - (當監督式模型尚未訓練完成時，需要先用手動去背+非監督式去背產出第一批MSAK)
    - 最後視覺化呈現決定採取哪個遮罩進行去背

- 按資料特性採用不同處理方案
    - 背景雜亂
        - 手動以小畫家去背製作mask 
    - 背景單純
        - 低對比度圖案
            - 先以程式篩出低對比的影像
            - 增加對比後再丟進其他背景單純的圖像一併處理 

### 流程

#### 1. 非監督式方法產生mask遮罩

* 目的是讓非監督式方法可以將背景跟主體影像區分開來產生鱗翅目影像的遮罩(mask)
* 取得主體255(白)，背景0(黑)的黑白mask 
* 使用技術 : [pytorch-unsupervised-segmentation](https://github.com/kanezaki/pytorch-unsupervised-segmentation) by kanezaki

#### 非監督+手動去背操作順序:
* 使用非監督方法直接取得部分mask
* 無法用非監督方法直接取得的部分，則利用非監督方法中間產出的rgb色塊圖
    * 使用小畫家填補工具將背景以後色填補後
    * 保留黑色背景(0)，將標本主體部分顏色全轉為白色(255)
    * 使用[Postprocess.ipynb](https://github.com/twcmchang/colorful-moth/blob/master/Postprocess.ipynb)工具，抓出最大的主體輪廓並填補，取得較好的mask效果
   
   ![Unsup流程示意](https://hackmd.io/_uploads/ryEPmze4K.png)


* 前兩者無法處理的，送去[remove.bg](https://www.remove.bg/)網站去背
    * 付費網站，每張成本約4-6ntw
    * 隱私模式下，實測可處理500張以上影像 
    * 取得去背影像後，再將圖片透明通道設為0，影像主體設為255即得到mask
* remove.bg網站無法去背的，則直接送去小畫家3D手動去背
    * 使用魔術選取工具
    * 去掉選取出來的主體保留背景，再轉換為mask

#### unsupervised-segmentation 參數設定
:::spoiler
##### 起始segmentation方法與參數選擇
- [Comparison of segmentation and superpixel algorithms](https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_segmentations.html)
- SLIC: 
    - conpactness 0.1-10。越小相近的色塊越容易被劃為一塊
    - sigma 接近0.0效果較好可以至0.5。 數字越大線條越平滑，對於色塊細微的差異越不敏感
    - lr=0.1
    - 背景填補後很難直接取得mask
    - 但適合取得rgb圖供後續用小畫家填補取得背景
- FELZENSZWALB
    - 1000起跳 -3000
    - sigma0.5起跳， 0.5-1
    - lr=0.05
    - 對於部分主題與背景分離的樣本可成功得到mask，但試驗成功率約10%
##### 對於不同類型影像
- 翅膀輪廓、對比明顯
- 翅膀透明或白色跟背景融合
    - 可以透過PCA(針對RGB三個色階使之差異最大化)、調整對比等方式先前處理影像，先加強樣本輪廓與背景的差異
    - minlab可以稍微設置大一點(建議10以下)，避免背景太難填補
:::


#### 3. 監督式方法訓練去背模型(核心採用UNET)
* 以`非監督+後處理`過的MASK作為標籤(以像素為單位)
    
#### 4. 使用訓練好的模型進行去背處理  


==========================================================

## 參考資料
#### Segmentation
- [Comparison of segmentation and superpixel algorithms](https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_segmentations.html)
- [kanezaki/pytorch-unsupervised-segmentation](https://github.com/kanezaki/pytorch-unsupervised-segmentation)
- [单张图片的无监督语义分割，理解算法，并改良原始代码(从30秒改到5秒)](https://zhuanlan.zhihu.com/p/68528056)

#### UNET
- https://github.com/milesial/Pytorch-UNet
