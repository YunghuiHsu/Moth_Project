import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Tuple

class VGG_feat_extractor(nn.Module):
    
    def __init__(self) -> None:
        super(VGG_feat_extractor, self).__init__()
        vgg19 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
        self.fe1    = self.make_layers(vgg19, 0, 2)
        self.other1 = self.make_layers(vgg19, 2, 5)
        self.fe2    = self.make_layers(vgg19, 5, 7)
        self.other2 = self.make_layers(vgg19, 7, 10)
        self.fe3    = self.make_layers(vgg19, 10, 12)
        self.other3 = self.make_layers(vgg19, 12, 19)
        self.fe4    = self.make_layers(vgg19, 19, 21)
        self.other4 = self.make_layers(vgg19, 21, 28)
        self.fe5    = self.make_layers(vgg19, 28, 30)
        self.other5 = self.make_layers(vgg19, 30, 37)
        
    def make_layers(self, vgg, index_s: int, index_f: int) -> nn.Sequential:
        layers = list(vgg.children())[0][index_s:index_f]
        return nn.Sequential(*layers)
    
    def forward(self, x) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        f1 = self.fe1(x)
        f2 = self.fe2(self.other1(f1))
        f3 = self.fe3(self.other2(f2))
        f4 = self.fe4(self.other3(f3))
        f5 = self.fe5(self.other4(f4))
        return f1, f2, f3, f4, f5