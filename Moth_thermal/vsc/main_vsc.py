# python main_vsc.py
import argparse
import os
import glob
import random
import time
import numpy as np
# import pandas as pd
# from math import log10
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.utils as vutils
from torch.autograd import Variable
import torchvision
from torchvision.utils import make_grid, save_image
import wandb

from utils.dataset import ImageDatasetFromFile, ImageDatasetFromCache
from utils.networks import VSC
from utils.average_meter import AverageMeter
from utils.utils import str_to_list, load_model, is_image_file, save_log, log_rec_images, save_checkpoint
#from model import AAResNet50NoTop


# =======================================================================================================================

parser = argparse.ArgumentParser(
    description='Train VSC(Variational Sparse Coding) Autoencoder model for Moth diversity'
)

# data
parser.add_argument('--dataroot', type=str, default="data/moth_thermal_rmbg_padding_256.npz",
                    help='path to dataset "data/moth_thermal_rmbg_padding_256" \
                        or load image_catch "data/moth_thermal_rmbg_padding_256.npz"')

parser.add_argument('--input_height', type=int, default=256,
                    help='the height  of the input image to network')
parser.add_argument('--input_width', type=int, default=256,
                    help='the width  of the input image to network')
parser.add_argument('--output_height', type=int, default=256,
                    help='the height  of the output image to network')
parser.add_argument('--output_width', type=int, default=256,
                    help='the width  of the output image to network')

# checkpoint
parser.add_argument("--start_epoch", default=0, type=int,
                    help="Manual epoch number (useful on restarts)")
parser.add_argument('--outf', default='results',
                    help='folder to output images and model checkpoints')
parser.add_argument("--save_iter", type=int, default=2, help="Default=1")
parser.add_argument("--test_iter", type=int, default=2000, help="Default=2000")
parser.add_argument('--nrow', type=int,
                    help='the number of images in each row', default=8)
parser.add_argument("--pretrained", default='', type=str,
                    help="path to pretrained model (default: none)")

# model
parser.add_argument('--batchSize', '-b',  type=int,
                    default=16, help='input batch size')
parser.add_argument('--channels', default="32, 64, 128, 256, 512, 512",
                    type=str, help='the list of channel numbers')
parser.add_argument("--hdim", type=int, default=512,
                    help="dim of the latent code, Default=512")
# parser.add_argument('--trainsize', type=int,
#                     help='number of training data', default=-1)
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=2)
parser.add_argument("--nEpochs", type=int, default=50000,
                    help="number of epochs to train for")
# parser.add_argument("--num_vsc", type=int, default=0,
#                     help="number of epochs to for vsc training")
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam. default=0.5')
# parser.add_argument("--momentum", default=0.9, type=float,
#                     help="Momentum, Default: 0.9")

# set env
#parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gpu', '-g', dest='gpu', default='0')
parser.add_argument('--parallel', action='store_true', default=False,
                    help='for multiple GPUs')
parser.add_argument('--manualSeed', type=int, help='manual seed')
#parser.add_argument("--pretrained", default='../vsc_wwlep_glance/model/vsc/model_local_epoch_3000_iter_0.pth', type=str, help="path to pretrained model (default: none)")

# =======================================================================================================================


def main():

    global opt, vsc_model, epoch
    opt = parser.parse_args()
    print(opt)

    dir_save = Path(opt.outf)
    for subdir in ['vsc', 'vsc_valid']:
        dir_save.joinpath(subdir).mkdir(exist_ok=True, parents=True)
    log_save_path = './results/vsc_losses.log'

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    # if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    # if torch.cuda.is_available() and not opt.cuda:
    #     print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" if opt.parallel else opt.gpu

    # --------------build VSC models -------------------------
    if opt.parallel:
        vsc_model = VSC(cdim=3, hdim=opt.hdim, channels=str_to_list(
            opt.channels), image_size=opt.output_height, parallel=True).cuda()
    else:
        print(opt.hdim, str_to_list(opt.channels), opt.output_height)
        vsc_model = VSC(cdim=3, hdim=opt.hdim, channels=str_to_list(
            opt.channels), image_size=opt.output_height).cuda()

    dir_pretrained = Path('pretrained')
    dir_pretrained.mkdir(exist_ok=True, parents=True)
    pretrained_default = f'pretrained/vsc_epoch_{opt.start_epoch:d}.pth'
    #pretrained_default = '../vsc_wwlep_glance/model/vsc/model_local_epoch_2000_iter_0.pth'

    vsc_model.train()

    use_adam = True
    if use_adam:
        optimizerE = optim.Adam(
            vsc_model.encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#         optimizerE = optim.AdamW(vsc_model.encoder.parameters(), lr=opt.lr)
        optimizerG = optim.Adam(
            vsc_model.decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#         optimizerG = optim.AdamW(vsc_model.encoder.parameters(), lr=opt.lr)
        optimizers = [optimizerE, optimizerG]
    else:
        optimizerE = optim.RMSprop(vsc_model.encoder.parameters(), lr=opt.lr)
        optimizerG = optim.RMSprop(vsc_model.decoder.parameters(), lr=opt.lr)
        optimizers = []

    if os.path.isfile(pretrained_default):
        if opt.start_epoch > 0:
            print("Loading default pretrained {opt.start_epoch}...")
            load_model(vsc_model, pretrained_default, optimizers)
        else:
            print("Start training from scratch...")
    elif opt.pretrained:
        print("Loading pretrained from {opt.pretrained:s}...")
        load_model(vsc_model, pretrained_default, optimizers)

    # -----------------load dataset--------------------------
    if opt.dataroot.endswith('npz' or 'npy'):
        # load image_catch.npz
        path_preloads = Path(f'{opt.dataroot}')
        print(f'\tLoading image_catch from : {path_preloads}')

        image_cache = np.load(path_preloads)['image_catch']
        print(f'\tCheck image_cache.shape : {image_cache.shape}')

        train_set = ImageDatasetFromCache(image_cache, aug=True)
    else:
        # load image from path
        print(f'\tLoading image_list from : {opt.dataroot}')
        image_list = [x for x in glob.iglob(
            opt.dataroot + '/**/*', recursive=True) if is_image_file(x)]
        #train_list = image_list[:opt.trainsize]
        train_list = image_list[:]
        assert len(train_list) > 0
        train_set = ImageDatasetFromFile(train_list, aug=True)

    vsc_data_loader = torch.utils.data.DataLoader(
        train_set, batch_size=opt.batchSize, shuffle=True,
        num_workers=int(opt.workers), drop_last=False, pin_memory=True)

    # valid_list = ['./benchmarks/' +
    #               x for x in os.listdir('./benchmarks') if is_image_file(x)]
    if Path('data/benchmarks.npz').exists():
        path_preloads = Path(f'data/benchmarks.npz')
        print(f'\tLoading benchmarks_catch from : {path_preloads}')
        benchmarks_cache = np.load(path_preloads)['image_catch']
        print(f'\tCheck benchmarks_cache.shape : {benchmarks_cache.shape}')
        valid_set = ImageDatasetFromCache(benchmarks_cache, aug=False)
    else:
        dir_benchmarks = Path('data/benchmarks')
        valid_list = [str(path) for path in dir_benchmarks.glob('*.png')]
        valid_set = ImageDatasetFromFile(valid_list, aug=False)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=opt.batchSize, shuffle=False,
        num_workers=int(opt.workers), pin_memory=True)

    start_time = time.time()

    #cur_iter = 0
    #cur_iter = int(np.ceil(float(opt.trainsize) / float(opt.batchSize)) * opt.start_epoch)
#     cur_iter = len(train_data_loader) * (opt.start_epoch - 1)
#     if opt.test_iter > len(train_data_loader) - 1:
#         opt.test_iter = len(train_data_loader) - 1

    # ----------------Train func
    def train_vsc(epoch, iteration, batch, denoised_batch, cur_iter):

        batch_size = batch.size(0)

        real = Variable(batch).cuda()
        denoised = Variable(denoised_batch).cuda()

        noise = Variable(torch.zeros(
            batch_size, opt.hdim).normal_(0, 1)).cuda()
        fake = vsc_model.sample(noise)

        time_cost = time.time()-start_time
        info = f"====> Cur_iter: [{cur_iter}]| Epoch[{epoch}]({iteration}/{len(data_loader)})"
        info += f"| time: {time_cost//(60*60):3.0f}h{time_cost//60%60:2.0f}m{time_cost%60:2.0f}s"
        loss_info = '[loss_rec, loss_margin, lossE_real_kl, lossE_rec_kl, lossE_fake_kl, lossG_rec_kl, lossG_fake_kl,]'

        # =========== Calc Losses and Update Optimizer ================

        real_mu, real_logvar, real_logspike, z, rec = vsc_model(real)

        loss_rec = vsc_model.reconstruction_loss(rec, denoised, True)
        #loss_rec =  0

        loss_prior = vsc_model.prior_loss(real_mu, real_logvar, real_logspike)

        loss = loss_rec + 2 * loss_prior

        optimizerG.zero_grad()
        optimizerE.zero_grad()
        loss.backward()
        optimizerE.step()
        optimizerG.step()

        try:
            am_rec.update(loss_rec.item())
        except:
            am_rec.update(0)

        am_prior.update(loss_prior.item())

        info += f'| Rec: {am_rec.val:.4f}({am_rec.avg:.4f}), Prior: {am_prior.val:.4f}({am_prior.avg:.4f})'
        print(info, end='\r')

        # save rec_imgs
        if epoch % 5 == 0:
            rec_imgs = log_rec_images(
                epoch, real, rec, fake, save=False, return_=True)
            experiment.log({
                'images': wandb.Image(rec_imgs, caption="1st row: Real, 2nd row: Rec, 3nd row: Fake"),
            }, commit=False)

        if epoch < 100 and epoch % 10 == 0:
            log_rec_images(epoch, real, rec, fake, outf=dir_save)
        elif epoch >= 100 and epoch % 100 == 0:
            log_rec_images(epoch, real, rec, fake, outf=dir_save)

    # ----------------Train func

    def valid_vsc(epoch, batch):
        with torch.no_grad():
            real = Variable(batch).cuda()
            *_, rec = vsc_model(real)

            if epoch % 5 == 0:
                rec_imgs = log_rec_images(
                    epoch, real, rec, flag='valid', save=False, return_=True)
                experiment.log({
                    'benchmarks': wandb.Image(rec_imgs, caption="1st row: Real, 2nd row: Rec, 3nd row: Fake"),
                })

            if epoch < 100 and epoch % 10 == 0:
                log_rec_images(
                    epoch, real, rec, flag='valid', outf=dir_save)
            elif epoch >= 100 and epoch % 100 == 0:
                log_rec_images(epoch, real, rec, flag='valid', outf=dir_save)

    # =======================================================================================================================

    # --------------Initialize logging------------
    experiment = wandb.init(
        project='Autoencoder_VSC_Moth_ThermalRangeSize', resume='allow')
    argparse_log = vars(opt)    # save argparse.Namespace into dictionary
    experiment.config.update(argparse_log)

    # ----------------Train by epochs--------------------------
    prev_checkpoint = None
    current_checkpoint = None

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        # save models
        save_epoch = (epoch//opt.save_iter)*opt.save_iter

        if epoch == save_epoch:
            # if epoch % opt.save_iter==0:
            current_checkpoint = save_checkpoint(
                vsc_model, optimizers, save_epoch)

            if prev_checkpoint is not None:
                try:
                    os.remove(prev_checkpoint)
                except:
                    print(f'\tCan\'t find {prev_checkpoint}. keep training')

        prev_checkpoint = current_checkpoint

        vsc_model.c = 50 + epoch * vsc_model.c_delta

        am_rec = AverageMeter()
        am_prior = AverageMeter()
        # # am_levelN = AverageMeter()

        data_loader = vsc_data_loader
        if opt.test_iter > len(data_loader) - 1:
            # opt.test_iter = len(data_loader) - 1
            opt.test_iter = int(len(data_loader)/2) + 1

        cur_iter = 0

        # --------------train vsc------------
        vsc_model.train()
        for iteration, (batch, denoised_batch, filenames) in enumerate(data_loader, 0):

            train_vsc(epoch, iteration, batch, denoised_batch, cur_iter)
            cur_iter += 1
        save_log(log_save_path, epoch, am_rec, am_prior)
        print(f'\n=========> Train round finished. Starting evaluation round ===========')

        # --------------store parameters to wandb histograms-----
        if epoch % 10 == 0:
            histograms = {}
            for tag, value in vsc_model.named_parameters():
                tag = tag.replace('/', '.')
                histograms['Weights/' +
                           tag] = wandb.Histogram(value.data.cpu())
                histograms['Gradients/' +
                           tag] = wandb.Histogram(value.grad.data.cpu())
            experiment.log({**histograms}, commit=False)

        # --------------valid------------
        vsc_model.eval()
        for iteration, (batch, denoised_batch, filenames) in enumerate(valid_data_loader, 0):

            print('=========> Validating...')
            valid_vsc(epoch, batch)

        print('=========> wandb logging...')
        # save logs
        experiment.log({
            'epoch': epoch,
            'loss_rec':  am_rec.avg,
            'loss_prior': am_prior.avg,
            'learning rate_Encoder': optimizerE.param_groups[0]['lr'],
            'learning rate_Generator': optimizerG.param_groups[0]['lr'],
            # 'images': wandb.Image(rec_imgs_train, caption="1st row: Real, 2nd row: Rec, 3nd row: Fake"),
            # 'Benchmarks': wandb.Image(rec_imgs_valid, caption="Upper row: Real; Lower row: Rec"),
            # **histograms
        })

# =======================================================================================================================


if __name__ == "__main__":
    main()
