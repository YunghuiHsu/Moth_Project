from layers import *
import antialiased_cnns_scripts
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class ResNet50_family_classifier(nn.Module):
    
    def __init__(self, out_planes: int = 90, getDeepFeature=True) -> None:
        super(ResNet50_family_classifier, self).__init__()
        resnet50 = antialiased_cnns_scripts.resnet50(pretrained=True)
        self.conv1 = Sequential(*list(resnet50.children())[0:4])
        self.conv2_x = Sequential(*list(resnet50.children())[4])
        self.conv3_x = Sequential(*list(resnet50.children())[5])
        self.conv4_x = Sequential(*list(resnet50.children())[6])
        self.conv5_x = Sequential(*list(resnet50.children())[7])
        self.avgpool = Sequential(list(resnet50.children())[8])
        self.fc = Linear(2048, out_planes)
        
    def forward(self, x) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        f1 = self.conv1(x)
        f2 = self.conv2_x(f1)
        f3 = self.conv3_x(f2)
        f4 = self.conv4_x(f3)
        f5 = self.conv5_x(f4)
        family = self.fc(self.avgpool(f5).view(f5.size(0), -1))
        
        if self.getDeepFeature:
            return family, f2, f3, f4, f5
        else:
            return family

    def CLRP(self,x):
        maxindex = torch.argmax(x)
        R = torch.ones(x.shape).cuda()
        R /= -1000
        R[:, maxindex] = 1

        return R

    def LRP(self,x):
        maxindex = torch.argmax(x)
        R = torch.zeros(x.shape).cuda()
        R[:, maxindex] = 1
        return R

    def m_relprop(self, R, pred, alpha):
        R = self.fc.m_relprop(R, pred, alpha)
        if torch.is_tensor(R) == False:
            for i in range(len(R)):
                R[i] = R[i].reshape_as(self.avgpool.Y)
        else:
            R = R.reshape_as(self.avgpool.Y)
        R = self.avgpool.m_relprop(R, pred, alpha)

        R = self.conv5_x.m_relprop(R, pred, alpha)
        R = self.conv4_x.m_relprop(R, pred, alpha)
        R = self.conv3_x.m_relprop(R, pred, alpha)
        R = self.conv2_x.m_relprop(R, pred, alpha)
        R = self.conv1.m_relprop(R, pred, alpha)

        return R

    def RAP_relprop(self, R):
        R = self.fc.RAP_relprop(R)
        R = R.reshape_as(self.avgpool.Y)
        R = self.avgpool.RAP_relprop(R)

        R = self.conv5_x.RAP_relprop(R)
        R = self.conv4_x.RAP_relprop(R)
        R = self.conv3_x.RAP_relprop(R)
        R = self.conv2_x.RAP_relprop(R)
        R = self.conv1.RAP_relprop(R)

        return R
