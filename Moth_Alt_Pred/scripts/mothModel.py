import torch
from torch import  nn
from torchvision import models

class ResNet50Moth(nn.Module):
    '''使用 ResNet50 模型，並移除原本分類用的輸出層 (Dense 1024)，改為 Jimmy (張鈞閔) 用的架構:
    - Dense 1024
    - Batch Normalize 1d
    - ReLU
    - Dense 1
    
    Average pool 層的 2048 維輸出值即為用來預測海拔的影像特徵值，一併修改模型 foward function 的回傳值，
    訓練結束後將這些特徵值存起來 '''
    
    def __init__(self, pretrained=False, progress=True):
        super().__init__()
        resnet50 = models.resnet50(pretrained, progress)  # pretrained=True，載入已經訓練好的模型
        feature = nn.ModuleList(resnet50.children())[:-1]           # 取出除了最後一層以外的模型架構   
        self.resnet_feature = nn.Sequential(*feature)               # 將取出的層放入nn.Sequential()中
        self.fc = nn.Sequential(                                    # 加入自定義的、最後的全連接層
            nn.Linear(2048, 1024, bias=False),                      # BN層設置bias無用，可小省參數數量  
            nn.BatchNorm1d(1024), 
            nn.ReLU(inplace=True),                                  # 微分前後數值都一樣，關閉可節省運算時間
            nn.Linear(1024, 1))
    def forward(self, x):
        x = self.resnet_feature(x)
        feature = x.view(x.size(0),-1)  # ([batch_size, 2048, 1, 1]) > ([batch_size,2048])
        alt = self.fc(feature)
        return feature, alt                                         # 回傳特徵值與預測值(海拔)