import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Tuple

class Flatten(nn.Module):
    def forward(self, input: Tensor):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input: Tensor, channel: int = 512, H: int = 4, W: int = 4):
        return input.view(input.size(0), channel, H, W)    


class ResidualBlock(nn.Module):

    def __init__(self, planes: int) -> None:
        super(ResidualBlock, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.resblock(x)
        out += identity
        return out


class VAE_WGAN(nn.Module):
    def __init__(self, 
                 in_planes: int = 3, 
                 block: Type[ResidualBlock] = ResidualBlock, 
                 z_dim: int = 100) -> None:
        super(VAE_WGAN, self).__init__()
        self.block = block
        self.encoder = nn.Sequential(
            self.make_layers(in_planes, 64, "encoder"),
            self.make_layers(64, 128, "encoder"),
            self.make_layers(128, 256, "encoder"),
            self.make_layers(256, 512, "encoder"),
            Flatten()
        )
        self.fc1 = nn.Linear(512*4*4, z_dim)
        self.fc2 = nn.Linear(512*4*4, z_dim)
        self.fc3 = nn.Linear(z_dim, 512*4*4)
        self.decoder = nn.Sequential(
            UnFlatten(),
            self.make_layers(512, 256, "decoder"),
            self.make_layers(256, 128, "decoder"),
            self.make_layers(128, 64, "decoder"),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReplicationPad2d(1),
            nn.Conv2d(64, in_planes, kernel_size=3)
        )
        self.discriminator = nn.Sequential(
            self.make_layers(64, 128, "discriminator"),
            self.make_layers(128, 256, "discriminator"),
            self.make_layers(256, 512, "discriminator"),
            self.make_layers(512, 512, "discriminator"),
            nn.Conv2d(512, 1, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True)
        )

    
    def make_layers(self, in_planes: int, out_planes: int, mode: str) -> nn.Sequential:
        if (mode == "encoder") or (mode == "discriminator"):
            layers = [
                nn.Conv2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(inplace=True),
                self.block(out_planes),
                nn.LeakyReLU(inplace=True)
            ]
            return nn.Sequential(*layers)
        elif mode == "decoder":
            layers = [
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ReplicationPad2d(1),
                nn.Conv2d(in_planes, out_planes, kernel_size=3, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(inplace=True),
                self.block(out_planes),
                nn.LeakyReLU(inplace=True)
            ]
            return nn.Sequential(*layers)
    
    def forward(self, x: Tensor, vgg, device_id=None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        h1 = self.encoder(x)
        
        # sampling codings in latent space
        mu, logvar = self.fc1(h1), self.fc2(h1)
        std = torch.exp(logvar/2) # std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(device_id)
        z = mu + std * esp
        
        h2 = self.fc3(z)
        recon_x = self.decoder(h2)
        fo1, *_ = vgg(x)
        fr1, *_ = vgg(recon_x)
        y = self.discriminator(torch.cat((fo1, fr1), 0))
        return y, recon_x, z, mu, std