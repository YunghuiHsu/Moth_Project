import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel
import torch.optim as optim
from torch.autograd import Variable
#from torchvision import datasets, transforms
import torch.nn.functional as F
#import torchvision.utils as vutils
import math
import time
import torch.multiprocessing as multiprocessing

class _Residual_Block(nn.Module): 
    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(_Residual_Block, self).__init__()
        
        midc = int(outc*scale)
        
        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        else:
            self.conv_expand = None
          
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x): 
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.relu2(self.bn2(torch.add(output,identity_data)))
        return output 


class VSCEncoder(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(VSCEncoder, self).__init__() 
        
        assert (2 ** len(channels)) * 4 == image_size
        
        self.hdim = hdim
        
        cc = channels[0]
        self.main = nn.Sequential(
                nn.InstanceNorm2d(cdim, affine=True),        # 对batch中的单个样本的每一层特征图抽出来一层层求mean和variance，与batch size无关
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),                
                nn.AvgPool2d(2),
              )
              
        sz = image_size//2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz//2), nn.AvgPool2d(2))
            cc, sz = ch, sz//2
        
        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))                    
        self.fc = nn.Linear((cc)*4*4, 3*hdim)
    
    def forward(self, x):        
        y = self.main(x).view(x.size(0), -1)
        y = self.fc(y)
        mu, logvar, logspike_ = y.chunk(3, dim=1)
        logspike = -F.relu(-logspike_)
        return mu, logvar, logspike

    
class Decoder(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(Decoder, self).__init__() 
        
        assert (2 ** len(channels)) * 4 == image_size
        
        cc = channels[-1]
        self.fc = nn.Sequential(
                      nn.Linear(hdim, cc*4*4),
                      nn.ReLU(True))
                  
        sz = 4
        self.main = nn.Sequential()
        for ch in channels[::-1]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz*2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz*2
       
        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))
                    
    def forward(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        y = y.view(z.size(0), -1, 4, 4)
        y = self.main(y)
        return y

class DFC_VSC_WGAN(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], d_channels=[256, 512, 1024, 1024], image_size=256, parallel=False, block=_Residual_Block):
        super(DFC_VSC_WGAN, self).__init__()         
        self.hdim = hdim
        self.parallel = parallel
        self.block = block
        self.encoder = VSCEncoder(cdim, hdim, channels, image_size)
        self.decoder = Decoder(cdim, hdim, channels, image_size)
        
        self.discriminator = nn.Sequential()
        sz = 64
        cc = d_channels[0]
        for ch in d_channels[1:]:
            self.discriminator.add_module(f"{sz}_to_{sz//2}", self.make_layers(cc, ch))
            sz, cc = sz//2, ch
        self.discriminator.add_module("conv", nn.Conv2d(1024, 1, kernel_size=4, stride=4, bias=False))
        self.discriminator.add_module("bn", nn.BatchNorm2d(1))
        self.discriminator.add_module("leakyrelu", nn.LeakyReLU(inplace=True))  #不能使用有負值的輸出
        self.alpha = 0.01
        self.c = 50
        self.c_delta = 0.01
    
    
    def forward(self, x, x_denoised, resnet50):
        #------VSC autoencoder-------------------
        mu, logvar, logspike = self.encode(x)
        z = self.reparameterize(mu, logvar, logspike)
        x_rec = self.decode(z)
        
        #------discriminator-------------------
        _, *f_targ = resnet50(x_denoised)
        _, *f_pred = resnet50(x_rec)
        
        if self.parallel:
            y =  data_parallel(self.discriminator, torch.cat((f_targ[0].detach(), f_pred[0].detach()), 0))
        else:
            y = self.discriminator(torch.cat((f_targ[0].detach(), f_pred[0].detach()), 0))
        
        return mu, logvar, logspike, z, x_rec, f_targ, f_pred, y
        
    def make_layers(self, in_planes: int, out_planes: int) -> nn.Sequential:
        layers = [
            nn.Conv2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(inplace=True),
            self.block(inc=out_planes, outc=out_planes)
        ]
        return nn.Sequential(*layers)
    
    def sample(self, z):
        y = self.decode(z)
        return y
    
    def encode(self, x):
        if self.parallel:
            mu, logvar, logspike = data_parallel(self.encoder, x)
        else:
            mu, logvar, logspike = self.encoder(x)
        return mu, logvar, logspike
        
    def decode(self, z):
        if self.parallel:
            x_rec = data_parallel(self.decoder, z)
        else:
            x_rec = self.decoder(z)
        return x_rec
    
#     def discriminator(self, f_targ, f_pred):
#         if self.parallel:
#             y =  data_parallel(self.discriminator, torch.cat((f_targ[0].detach(), f_pred[0].detach()), 0))
#         else:
#             y = self.discriminator(torch.cat((f_targ[0].detach(), f_pred[0].detach()), 0))
#         return y

    def reparameterize(self, mu, logvar, logspike):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        gaussian = eps.mul(std).add_(mu)
        eta = torch.rand_like(std)
        #selection = F.sigmoid(125 * (eta + logspike.exp() - 1))
        selection = F.sigmoid(self.c * (eta + logspike.exp() - 1))
        return selection.mul(gaussian)
    
    def kl_loss(self, mu, logvar, prior_mu=0):
        v_kl = mu.add(-prior_mu).pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        v_kl = v_kl.sum(dim=-1).mul_(-0.5) # (batch, 2)
        return v_kl
    
    def reconstruction_loss(self, f_targ, f_pred, size_average=False):
        MSE = 0
        for i in range(len(f_targ)):
            error = F.mse_loss(f_targ[i], f_pred[i], reduction="none").reshape(f_targ[i].size(0), -1).sum(dim=-1)
            if size_average:
                MSE += error.mean()
            else:
                MSE += error.sum()
        return MSE

    # Reconstruction + KL divergence losses summed over all elements of batch
    def prior_loss(self, mu, logvar, logspike):
        # see Appendix B from VSC paper / Formula 6
        spike = torch.clamp(logspike.exp(), 1e-6, 1.0 - 1e-6) 

        prior1 = -0.5 * torch.sum(spike.mul(1 + logvar - mu.pow(2) - logvar.exp()))
        prior21 = (1 - spike).mul(torch.log((1 - spike) / (1 - self.alpha)))
        prior22 = spike.mul(torch.log(spike / self.alpha))
        prior2 = torch.sum(prior21 + prior22)
        PRIOR = prior1 + prior2

        LOSS = 0.01 * PRIOR

        return LOSS
    
    def W_loss(self, y):
        return -y[:int(y.size(0)/2)].mean() + y[int(y.size(0)/2):].mean()