
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
    

# 一、物件偵測(object detection)，裁切抓取出主體(蛾類標本)
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
