  - 遮罩轉換(convert_to_mask.py)
    - 將3通道(channel=3, 8x3=24bit)的mask，轉換為1通道(channel=1, 8x1=8bit)
    - 將各種來源的mask轉換為單通道mask，包括
      - remove.bg得到的去背影像
      - 小畫家修補去背
      - Unsup segmentation獲得的rgb色塊圖 
  - 重複檢查(check_mask_duplicated.py)
    - 刪除/保留重複  
  - 獲得去背影像(get_Image_rmbg.py)
    - 原圖檔 x mask 產出去背影像
    - 背景可指定顏色填補

  - 花式取樣(get_imgs_label_byHSV.py)
    - 根據標本翅膀部位的hsv做分群產出label
    - 系統化採樣 在每張照片固定位置取一個5x5或3x3的方塊，計算HSV平均值
      - 假設取12個點，得到36個值，然後做分群 然後按照群去分層取樣
      - 取樣點分布示意(翅膀部位取樣(3x3)，每個取樣點計算h,s,v平均)
      - 依據取樣點計算h,s,v後，降維成3個維度，並使用Birch分群視覺化
      - 最後產出imgs_label_byHSV.csv，提供Sup_train.py 在資料切分為train、valid時使用
      <pre><code>
      df = pd.read_csv('../data/imgs_label_byHSV.csv', index_col=0)
      X_train, X_valid, y_train, y_valid = train_test_split(
          img_paths, mask_paths, stratify=df.label)
      </code></pre>

  - mask後處理(Postprocess.py)
    - 原理:抓取最大輪廓線並進行填補
    - 執行效果:
      - 去除mask最大輪廓線外的雜訊，並將輪廓線內的畫素完全填補
