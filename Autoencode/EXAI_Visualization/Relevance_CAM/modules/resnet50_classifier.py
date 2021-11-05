import torch
import torch.nn as nn
import antialiased_cnns
from torch.nn.parallel import data_parallel
from torch import Tensor
from typing import Tuple

class ResNet50_family_classifier(nn.Module):
    
    def __init__(self, out_planes: int = 90, parallel=False, getDeepFeature=True) -> None:
        super(ResNet50_family_classifier, self).__init__()
        resnet50 = antialiased_cnns.resnet50(pretrained=True)
        
        self.parallel = parallel
        self.getDeepFeature = getDeepFeature
        
        self.conv1 = nn.Sequential(*list(resnet50.children())[0:4])
        self.conv2_x = nn.Sequential(*list(resnet50.children())[4])
        self.conv3_x = nn.Sequential(*list(resnet50.children())[5])
        self.conv4_x = nn.Sequential(*list(resnet50.children())[6])
        self.conv5_x = nn.Sequential(*list(resnet50.children())[7])
        self.avgpool = nn.Sequential(list(resnet50.children())[8])
        self.fc = nn.Linear(2048, out_planes)
        
    def forward(self, x) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.parallel:
            f1 = data_parallel(self.conv1, x)
            f2 = data_parallel(self.conv2_x, f1)
            f3 = data_parallel(self.conv3_x, f2)
            f4 = data_parallel(self.conv4_x, f3)
            f5 = data_parallel(self.conv5_x, f4)
            favg = data_parallel(self.avgpool, f5)
            fc = data_parallel(self.fc, favg.reshape(favg.size(0), -1))
        else:
            f1 = self.conv1(x)
            f2 = self.conv2_x(f1)
            f3 = self.conv3_x(f2)
            f4 = self.conv4_x(f3)
            f5 = self.conv5_x(f4)
            fc = self.fc(self.avgpool(f5).reshape(f5.size(0), -1))
        
        if self.getDeepFeature:
            return fc, f2, f3, f4, f5
        else:
            return fc

class ResNet50_nonfc(nn.Module):
    
    def __init__(self, out_planes: int = 90, parallel=False) -> None:
        super(ResNet50_nonfc, self).__init__()
        resnet50 = antialiased_cnns.resnet50(pretrained=True)
        
        self.parallel = parallel
        
        self.conv1 = nn.Sequential(*list(resnet50.children())[0:4])
        self.conv2_x = nn.Sequential(*list(resnet50.children())[4])
        self.conv3_x = nn.Sequential(*list(resnet50.children())[5])
        self.conv4_x = nn.Sequential(*list(resnet50.children())[6])
        self.conv5_x = nn.Sequential(*list(resnet50.children())[7])
        self.avgpool = nn.Sequential(list(resnet50.children())[8])
        self.fc = nn.Linear(2048, out_planes)
        
    def forward(self, x) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.parallel:
            f1 = data_parallel(self.conv1, x)
            f2 = data_parallel(self.conv2_x, f1)
            f3 = data_parallel(self.conv3_x, f2)
            f4 = data_parallel(self.conv4_x, f3)
            f5 = data_parallel(self.conv5_x, f4)
#             favg = data_parallel(self.avgpool, f5)
#             family = data_parallel(self.fc, favg.reshape(favg.size(0), -1))
        else:
            f1 = self.conv1(x)
            f2 = self.conv2_x(f1)
            f3 = self.conv3_x(f2)
            f4 = self.conv4_x(f3)
            f5 = self.conv5_x(f4)
#             family = self.fc(self.avgpool(f5).reshape(f5.size(0), -1))
        return f2, f3, f4, f5

