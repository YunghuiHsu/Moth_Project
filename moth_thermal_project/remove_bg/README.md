
## 2.去背
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
