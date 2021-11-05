# This code is built from the PyTorch examples repository: https://github.com/pytorch/vision/tree/master/torchvision/models.
# Copyright (c) 2017 Torch Contributors.
# The Pytorch examples are available under the BSD 3-Clause License.
#
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2019 Adobe. All rights reserved.
# Adobe’s modifications are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License (CC-NC-SA-4.0). To view a copy of the license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.
#
# ==========================================================================================
#
# BSD-3 License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from layers import *
from blurpool import *

__all__ = ['ResNet', 'resnet50']

model_urls = {
    'resnet50_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet50_lpf4_finetune-cad66808.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, filter_size=1):
        super(Bottleneck, self).__init__()
        self.clone = Clone()

        norm_layer = BatchNorm2d

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)  # Conv(stride2)-Norm-Relu --> #Conv-Norm-Relu-BlurPool(stride2)
        self.bn2 = norm_layer(planes)
        if(stride==1):
            self.conv3 = conv1x1(planes, planes * self.expansion)
        else:
            self.conv3 = Sequential(BlurPool(planes, filt_size=filter_size, stride=stride),
                conv1x1(planes, planes * self.expansion))
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.relu1 = ReLU(inplace=True)
        self.relu2 = ReLU(inplace=True)
        self.relu3 = ReLU(inplace=True)

        self.add = Add()

        self.register_forward_hook(forward_hook)


    def forward(self, x):
        x1, x2 = self.clone(x, 2)

        #out = self.conv1(x)
        out = self.conv1(x1)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            #2 = self.downsample(x)
            x2 = self.downsample(x2)

        out = self.add([out, x2])
        #out = self.add([out, x])
        out = self.relu3(out)

        return out

    def relprop(self, R, alpha):
        out = self.relu2.relprop(R, alpha)
        out, x2 = self.add.relprop(out, alpha)

        if self.downsample is not None:
            x2 = self.downsample.relprop(x2, alpha)

        out = self.bn2.relprop(out, alpha)
        out = self.conv2.relprop(out, alpha)

        out = self.relu1.relprop(out, alpha)
        out = self.bn1.relprop(out, alpha)
        x1 = self.conv1.relprop(out, alpha)

        return self.clone.relprop([x1, x2], alpha)

    def m_relprop(self, R, pred, alpha):
        out = self.relu2.m_relprop(R, pred, alpha)
        out, x2 = self.add.m_relprop(out, pred, alpha)

        if self.downsample is not None:
            x2 = self.downsample.m_relprop(x2, pred, alpha)

        out = self.bn2.m_relprop(out, pred, alpha)
        out = self.conv2.m_relprop(out, pred, alpha)

        out = self.relu1.m_relprop(out, pred, alpha)
        out = self.bn1.m_relprop(out, pred, alpha)
        x1 = self.conv1.m_relprop(out, pred, alpha)

        return self.clone.m_relprop([x1, x2], pred, alpha)

    def RAP_relprop(self, R):
        out = self.relu2.RAP_relprop(R)
        out, x2 = self.add.RAP_relprop(out)

        if self.downsample is not None:
            x2 = self.downsample.RAP_relprop(x2)

        out = self.bn2.RAP_relprop(out)
        out = self.conv2.RAP_relprop(out)

        out = self.relu1.RAP_relprop(out)
        out = self.bn1.RAP_relprop(out)
        x1 = self.conv1.RAP_relprop(out)

        return self.clone.RAP_relprop([x1, x2])

# Simplified
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, norm_layer=None):
        super(ResNet, self).__init__()

        self._norm_layer = BatchNorm2d

        self.inplanes = 64

        self.conv1 = Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = ReLU(inplace=True)
        self.maxpool = Sequential(*[
                MaxPool2d(kernel_size=2, stride=1), 
                BlurPool(self.inplanes, filt_size=4, stride=2,)
            ])

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, Conv2d):
                if(m.in_channels!=m.out_channels or m.out_channels!=m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    print('Not initializing')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=2):
        norm_layer = self._norm_layer
        downsample = None

        # since this is just a conv1x1 layer (no nonlinearity),
        # conv1x1->blurpool is the same as blurpool->conv1x1; the latter is cheaper
        downsample = [BlurPool(filt_size=4, stride=stride, channels=self.inplanes),] if(stride !=1) else []
        downsample += [conv1x1(self.inplanes, planes * block.expansion, 1),
            norm_layer(planes * block.expansion)]
        downsample = Sequential(*downsample)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer, filter_size=4))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, filter_size=4))

        return Sequential(*layers)

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


    def m_relprop(self, R, pred, alpha):
        R = self.fc.m_relprop(R, pred, alpha)
        if torch.is_tensor(R) == False:
            for i in range(len(R)):
                R[i] = R[i].reshape_as(self.avgpool.Y)
        else:
            R = R.reshape_as(self.avgpool.Y)
        R = self.avgpool.m_relprop(R, pred, alpha)

        R = self.layer4.m_relprop(R, pred, alpha)
        R = self.layer3.m_relprop(R, pred, alpha)
        R = self.layer2.m_relprop(R, pred, alpha)
        R = self.layer1.m_relprop(R, pred, alpha)

        R = self.maxpool.m_relprop(R, pred, alpha)
        R = self.relu.m_relprop(R, pred, alpha)
        R = self.bn1.m_relprop(R, pred, alpha)
        R = self.conv1.m_relprop(R, pred, alpha)

        return R

    def RAP_relprop(self, R):
        R = self.fc.RAP_relprop(R)
        R = R.reshape_as(self.avgpool.Y)
        R = self.avgpool.RAP_relprop(R)

        R = self.layer4.RAP_relprop(R)
        R = self.layer3.RAP_relprop(R)
        R = self.layer2.RAP_relprop(R)
        R = self.layer1.RAP_relprop(R)

        R = self.maxpool.RAP_relprop(R)
        R = self.relu.RAP_relprop(R)
        R = self.bn1.RAP_relprop(R)
        R = self.conv1.RAP_relprop(R)

        return R

def resnet50(pretrained=False, filter_size=4, pool_only=True, _force_nonfinetuned=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        filter_size (int): Antialiasing filter size
        pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
        _force_nonfinetuned (bool): [False] If True, load the trained-from scratch pretrained model (if available)
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], filter_size=filter_size, pool_only=pool_only, **kwargs)
    if pretrained:
        if(filter_size==4 and not _force_nonfinetuned):
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50_lpf4_finetune'], map_location='cpu', check_hash=True)['state_dict'])
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50_lpf%i'%filter_size], map_location='cpu', check_hash=True)['state_dict'])
    return model
