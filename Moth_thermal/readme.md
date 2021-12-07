


## 資料夾說明
### yolov4/
- 從標本照框選出標本所在位置，獲得BoundingBox

### removebg/
- 使用監督與非監督方法進行影像分割，取得去背的標本影像
    - Supervised : [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
    - Unsupervised : [Unsupervised Image Segmentation by Backpropagation](https://github.com/kanezaki/pytorch-unsupervised-segmentation)

### vsc/
- 使用去背的蛾類標本影像作為材料，進行Autoencoder模型訓練
    - Autoencoder版本 : [Variational Sparse Coding](https://github.com/ftonolini45/Variational_Sparse_Coding) 

### data/
- 放置資料所在
