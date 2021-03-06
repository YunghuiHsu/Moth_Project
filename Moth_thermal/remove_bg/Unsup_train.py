# coding: utf-8

# Package
from func.tool import get_fname
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.nn.init
from skimage.segmentation import mark_boundaries  # show SLIC result
from skimage.transform import resize
from skimage import segmentation
from skimage import io
from PIL import Image
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import argparse
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
# get_ipython().run_line_magic('matplotlib', 'inline')

# =====================================================================================================
# # Parser
parser = argparse.ArgumentParser(
    description='PyTorch Unsupervised Segmentation')
parser.add_argument('--gpu_id', '-g', default=0, type=int)

parser.add_argument('--SAVEDIR', default='data/Unsup_rmbg')
parser.add_argument('--XX_DIR', default='data/data_for_Unsup_rmbg')

parser.add_argument('--nChannel', metavar='N', default=100,
                    type=int, help='number of channels')
parser.add_argument('--inputDimension', '-d', metavar='C', default=3,
                    type=int, help='number of input Dimension(channels) of imgs')
parser.add_argument('--maxIter', '-t', metavar='T', default=1000,
                    type=int, help='number of maximum iterations')
parser.add_argument('--minLabels', '-la', default=3, type=int,
                    help='minimum number of labels')
parser.add_argument('--lr', '-lr', metavar='LR', default=0.1,
                    type=float, help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2,
                    type=int, help='number of convolutional layers')

# segmentation and superpixel algorithm-------------------------------------------------------------------------------------------------------------
parser.add_argument('--segmentation', '-seg', metavar='Segmentation', default='SLIC', type=str,
                    help='segmentation and superpixel algorithm : "SLIC", "FEL"(Felzenszwalb) ')
# ??????parameter for "SLIC"
parser.add_argument('--compactness', '-c',  metavar='Compactness', default=10,
                    type=float, help='compactness of superpixels')
parser.add_argument('--num_superpixels', '-p', metavar='num_superpixels',
                    default=2000, type=int, help='number of superpixels')

# ??????parameter for "Felzenszwalb"
parser.add_argument('--sca', metavar='Scale', default=1000,
                    type=float, help='Higher means larger clusters')
parser.add_argument('--sig', metavar='Sigma', default=0.0, type=float,
                    help='Width (standard deviation) of Gaussian kernel used in preprocessing')
parser.add_argument('--size', metavar='Min_size', default=20000, type=int,
                    help='Minimum component size. Enforced using postprocessing.')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
use_cuda = torch.cuda.is_available()

# =====================================================================================================

method_ = (f'{args.segmentation}_mlabel{args.minLabels}_lr{args.lr}_compact{args.compactness}_sigma{args.sig}'
           if args.segmentation == 'SLIC'
           else f'{args.segmentation}_mlabel{args.minLabels}_lr{args.lr}_scale{args.sca}_sigma{args.sig}')

imgdir = Path(args.XX_DIR)

moths_path = list(imgdir.rglob('*.png'))
# moths_path = [os.path.join(imgdir, i) for i in moths if os.path.splitext(
#     i)[-1].lower() in ['.jpg', '.png', '.jpeg']]
print('Total Number of images : ', len(moths_path))


# # Model
# CNN model
class MyNet(nn.Module):
    def __init__(self, input_dim):
        print(input_dim)
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = []
        self.bn2 = []
        for i in range(args.nConv-1):
            self.conv2.append(
                nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(args.nChannel))
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel,
                               kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = self.bn2[i](x)
            x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


# Train
def visul():
    im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
    im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
    rgb = np.fliplr(im_target_rgb.reshape(-1, 3)).reshape(im_target_rgb.shape)
    cost = time.time() - s_time

    fig = plt.figure(figsize=(20, 20), dpi=400)
    ax = fig.add_subplot(151)
    ax.set_title('Input Image\n{}'.format(fname))
    ax.imshow(inpt)

    ax = fig.add_subplot(152)
    ax.set_title('pred\n n_label:{} \n iter:{} \n time:{}' .format(
        nLabels, batch_idx+1, cost))
    ax.imshow(rgb)

    ax = fig.add_subplot(153)

    points = [str(rgb[25, 128, :]),  # up-mid
              str(rgb[25, 25, :]),  # up-left
              str(rgb[25, 225, :]),  # up-right
              str(rgb[225, 25, :]),  # down-left
              str(rgb[225, 225, :])]  # down-right
    p_max = max(points, key=points.count)
    vote = points.count(p_max)
    ax.set_title('keep_center_pixel\n vote:{}/5' .format(vote))

    for pts in ([rgb[25, 128, :],
                 rgb[25, 25, :],
                 rgb[25, 225, :],
                 rgb[225, 25, :],
                 rgb[225, 225, :]]):
        if str(pts) == p_max:
            rgb_Nto_kp = pts

    tmp = rgb.copy()
    for i in range(256):
        for j in range(256):
            if np.all(tmp[j, i, :] == rgb_Nto_kp):
                tmp[j, i, :] = 1
            else:
                tmp[j, i, :] = 0

    ax.imshow(tmp*255)

    ax = fig.add_subplot(154)
    tmp2 = rgb.copy()
    for i in range(256):
        for j in range(256):
            if np.all(tmp2[j, i, :] == rgb_Nto_kp):
                tmp2[j, i, :] = 0
            else:
                tmp2[j, i, :] = 1
    ax.set_title('mask')
    ax.imshow(tmp2*255)

    ax = fig.add_subplot(155)
    ax.set_title('rmbg')
    ax.imshow(inpt*tmp2 + tmp)

    # plt.show()

    save_to_check = os.path.join(model_dir, 'checking')
    if not os.path.exists(save_to_check):
        os.makedirs(save_to_check)
    fig.savefig(os.path.join(save_to_check, fname + f'_checking_{method_}.jpg'),
                dpi=100, format='jpg', bbox_inches='tight')

    save_to_rmbg = os.path.join(model_dir, 'moth_rmbg')
    if not os.path.exists(save_to_rmbg):
        os.makedirs(save_to_rmbg)
    io.imsave(os.path.join(save_to_rmbg, fname +
              f'_rmbg_{method_}.jpg'), inpt*tmp2 + tmp)

    save_to_mask = os.path.join(model_dir, 'moth_rmbg_mask')
    if not os.path.exists(save_to_mask):
        os.makedirs(save_to_mask)
    io.imsave(os.path.join(save_to_mask, fname +
              f'_mask_{method_}.png'), tmp2*255)

    save_to_rmbg = os.path.join(model_dir, 'pred')
    if not os.path.exists(save_to_rmbg):
        os.makedirs(save_to_rmbg)
    io.imsave(os.path.join(save_to_rmbg, fname +
              f'_rgb_{method_}.png'), rgb)

    plt.close(fig)


# Save DIR
save_dir = Path(args.SAVEDIR).joinpath(args.segmentation)
save_dir.mkdir(parents=True, exist_ok=True)

# Model naming
log_time_str = time.strftime("%y%m%d%H%M%S")
model_dir = save_dir.joinpath(method_)
model_dir.mkdir(parents=True, exist_ok=True)

# init & loss, optimizer setting
# model = MyNet(3)
# if use_cuda:
#     model.cuda()
#     for i in range(args.nConv-1):
#         model.conv2[i].cuda()
#         model.bn2[i].cuda()
# model.train()
loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)


label_colours = np.random.randint(255, size=(100, 3))


for index, img_path in enumerate(moths_path):
    s_time = time.time()

    # init & loss, optimizer setting ----------------------------------------------------------------------
    model = MyNet(args.inputDimension)
    if use_cuda:
        model.cuda()
        for i in range(args.nConv-1):
            model.conv2[i].cuda()
            model.bn2[i].cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # ----------------------------------------------------------------------

    # input: load and process
    print(index, img_path)
    im = cv2.imread(str(img_path))
    im = resize(im, (256, 256))
    fname = get_fname(img_path)
    print(index, fname)
    data = torch.from_numpy(
        np.array([im.transpose((2, 0, 1)).astype('float32')/255.]))
    if use_cuda:
        data = data.cuda()
    data = Variable(data)
    inpt = np.fliplr(im.reshape(-1, 3)).reshape(im.shape)

    # segmentation and superpixel algorithms??
    # slic
    # The compactness parameter trades off color-similarity and proximity, as in the case of Quickshift,
    # while n_segments chooses the number of centers for kmeans.

    if args.segmentation == 'SLIC':
        labels = segmentation.slic(
            im, compactness=args.compactness, sigma=args.sca, n_segments=args.num_superpixels)
    elif args.segmentation == 'FEL':
        labels = segmentation.felzenszwalb(
            im, scale=args.sca, sigma=args.sca, min_size=args.size)

#     print('SLIC number of segments: {}'.format(len(np.unique(labels))))
#     plt.imshow(mark_boundaries(im, labels))
#     plt.show()
    labels = labels.reshape(im.shape[0]*im.shape[1])  # flatten
    u_labels = np.unique(labels)
    # put place_index with same SLIC group in a same sublist
    # n_sublist = number of unique labels
    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append(np.where(labels == u_labels[i])[0])
    print(f'{args.segmentation} n_LABELS: ', len(u_labels))

    for batch_idx in range(args.maxIter):
        # forwarding
        optimizer.zero_grad()

        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
        ignore, target = torch.max(output, 1)

        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
#         print('MODEL OUTPUT n_LABELS: ', nLabels)

        visul()

        # superpixel refinement
        # TODO: use Torch Variable instead of numpy for faster calculation

        for i in range(len(l_inds)):
            labels_per_sp = im_target[l_inds[i]]
            u_labels_per_sp = np.unique(labels_per_sp)
            hist = np.zeros(len(u_labels_per_sp))
            for j in range(len(hist)):
                hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
            im_target[l_inds[i]] = u_labels_per_sp[np.argmax(hist)]
        target = torch.from_numpy(im_target)
        if use_cuda:
            target = target.cuda()
        target = Variable(target)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # , 'loss:',loss.data[0], end='\r')
        print(batch_idx, '/', args.maxIter, 'n_label:', nLabels, end='\r')
        if nLabels <= args.minLabels:
            print("nLabels", nLabels, "reached minLabels",
                  args.minLabels, "\t" * 10)
            torch.save(model.state_dict(), model_dir.joinpath('model.pkl'))
            break


# # SAVE_LOG

# Save log
summary_save = '%s/training_summary.csv' % (args.SAVEDIR)
# save into dictionary
sav = vars(args)
sav['model_dir'] = model_dir


# Append into summary files
dnew = pd.DataFrame(sav, index=[0])
if os.path.exists(summary_save):
    dori = pd.read_csv(summary_save)
    dori = pd.concat([dori, dnew])
    dori.to_csv(summary_save, index=False)
else:
    dnew.to_csv(summary_save, index=False)

print(summary_save)
