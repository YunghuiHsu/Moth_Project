## 2.2 監督式方法(Supervied Segmentation)
- 核心技術為[Unet](https://github.com/milesial/Pytorch-UNet)
- 訓練過程見　［Background Remove for Moth Project］（https://hackmd.io/@YungHuiHsu/Hkr3R3wV9） 
 
### 2.2.1 訓練
- [Sup_train_pytorch.py ](https://github.com/YunghuiHsu/Moth_Project/blob/92125e68f634816c04bc1bd9e0398605bc583b18/Moth_thermal/remove_bg/Sup_train_pytorch.py)
- 以`非監督+後處理`過的MASK作為標籤(以像素為單位)


```
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
 ``` 
