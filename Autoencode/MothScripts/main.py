import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from vae_wgan import VAE_WGAN
from vgg import VGG_feat_extractor
from dataset import PokemonImageDataset

device_id = torch.device('cuda:0')
fnames = np.load(os.path.join("fnames.npy"), allow_pickle=True)
a
vgg = VGG_feat_extractor().to(device_id)

def loss_fn(recon_x, x, mu, std, vgg, y, labels):
    fo = vgg(x)
    fr = vgg(recon_x)
    MSE = 0
    c = [64, 128, 256, 512, 512]
    for i in range(5):
        MSE += F.mse_loss(fo[i], fr[i], reduction="sum")*100/c[i]**2
    #MSE = F.mse_loss(recon_x, x, reduction="sum")
    
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = 0.5 * torch.mean(std.pow(2) - (1 + std.pow(2).log()) + mu.pow(2))
    BCE = F.binary_cross_entropy_with_logits(y, labels, reduction="sum")
    loss_D = -y[:x.size(0)].mean() + y[x.size(0):].mean()
    
    return MSE + 500*KLD + 10*loss_D, MSE, KLD, 10*loss_D

def save_checkpoint(model, optimizer, epoch_f, pth_name):
    checkpoint = {
        "epoch": epoch_f, 
        "model_state": model.state_dict(), 
        "optim_state": optimizer.state_dict()
    }
    torch.save(checkpoint, pth_name)

model = VAE_WGAN().to(device_id)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
checkpoint = torch.load("cp1801.pth")
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optim_state'])
epoch_s = checkpoint['epoch']
#epoch_s = 0
epoch_f = epoch_s + 1
#optimizer.param_groups[0]['lr'] = 5e-5

model.train()

for epoch in range(epoch_s, epoch_f):
    loss_list = np.array([0, 0, 0, 0])
    for idx, images in enumerate(dloader):
        images = images.to(device_id)
        optimizer.zero_grad()
        y, recon_images, z, mu, std = model(images, vgg)
        labels = torch.tensor([10, -10]).repeat_interleave(z.size(0)).to(device_id)
        loss, mse, kld, bce = loss_fn(recon_images, images, mu, std, vgg, y.view(-1), labels.float())
        loss.backward()
        optimizer.step()
        loss_list = loss_list + np.array([loss.item(), mse.item(), kld.item(), bce.item()])
    print(f"Epoch[{epoch+1}] Loss: {round(loss_list[0]/idx, 3)} {round(loss_list[1]/idx, 3)} {round(loss_list[2]/idx, 3)} {round(loss_list[3]/idx, 3)}")

save_checkpoint(model, optimizer, epoch+1, f"cp{epoch+1}.pth")