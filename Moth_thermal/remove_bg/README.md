## 2.2 監督式方法(Supervied Segmentation)
- Modified from [`milesial/Pytorch-UNet`](https://github.com/milesial/Pytorch-UNet)
- 訓練過程見　［Background Remove for Moth Project］（https://hackmd.io/@YungHuiHsu/Hkr3R3wV9） 
 
### Train
- [Sup_train_pytorch.py ](https://github.com/YunghuiHsu/Moth_Project/blob/92125e68f634816c04bc1bd9e0398605bc583b18/Moth_thermal/remove_bg/Sup_train_pytorch.py)
- 以`非監督+後處理`過的MASK作為標籤(以像素為單位)


```
> python Sup_train_pytorch.py -h
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


### Predict
After training your model and saving it to MODEL.pth, you can easily test the output masks on your images via the CLI.

To predict a single image and save it:

python predict.py -i image.jpg -o output.jpg

```
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] 
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Filenames of input images
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of output images
  --viz, -v             Visualize the images as they are processed
  --no-save, -n         Do not save the output masks
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel white
  --scale SCALE, -s SCALE
                        Scale factor for the input images
```
