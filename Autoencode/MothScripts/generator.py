import torch.nn as nn

class MothGenerator(nn.Module):
    def __init__(self, vae_wgan):
        super().__init__()
#         self.moth_generator = nn.Sequential(*list(vae_wgan.children())[3:5])
        self.moth_generator = nn.Sequential(*[module for name, module in vae_wgan.named_children() if name in ['fc3','decoder']])
    def forward(self, x):
        output = self.moth_generator(x)
        return output