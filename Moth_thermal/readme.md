
### 研究目標:
- 探討不同緯度及海拔下，鱗翅目**適溫範圍與型態功能性變異**的關係

---  

### 資料夾說明(按工作順序)

#### data/
- 放置資料所在
- data/tools/
    - 放置一些小工具，包括
        - 遮罩轉換、重複檢查、根據mask產出去背影像、mask後處理
        - 根據標本翅膀部位的hsv做分群產出label等

#### yolov4/
- 內含針對鱗翅目標本訓練好的yolov4模型
- 使用yolo模型從標本照框選出標本所在位置(獲得BoundingBox)
- 根據產出的BoundingBox挑整框選範圍，獲得初步crop與padding過的標本影像(256 x 256)

#### removebg/
- 使用監督與非監督方法進行影像分割，取得去背的鱗翅目標本影像
    - Supervised : [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
    - Unsupervised : [Unsupervised Image Segmentation by Backpropagation](https://github.com/kanezaki/pytorch-unsupervised-segmentation)

#### vsc/
- 使用去背的鱗翅目標本影像作為材料，進行Autoencoder模型訓練
    - 使用的Autoencoder版本為 : [Variational Sparse Coding](https://github.com/ftonolini45/Variational_Sparse_Coding) 

- Training result(embeddings) exploring
    - `explore_LatentSpace.ipynb`
