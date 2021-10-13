import argparse
import logging
import pathlib
import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
import skimage.io as io

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.data_loading_moth import MothDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
from utils.utils import early_stop, plt_learning_curve, get_data_attribute

# =======================================================================================================
# def get_args():
parser = argparse.ArgumentParser(
    description='Train the UNet on images and target masks')

# enviroment
parser.add_argument('--gpu', '-g', dest='gpu', default='0')
parser.add_argument("--save_epoch", '-save_e', type=int,
                    default=1, help="save checkpoint per epoch")

# data
parser.add_argument('--XX_DIR', dest='XX_DIR',
                    default='data/data_for_Sup_train/imgs/')
parser.add_argument('--YY_DIR', dest='YY_DIR',
                    default='data/data_for_Sup_train/masks/')
parser.add_argument('--SAVEDIR', dest='SAVEDIR',
                    default='model/Unet_rmbg')  # Unet_rmbg

parser.add_argument('--image_input_size', '-s_in', dest='size_in',
                    type=str, default='256,256', help='image size input')
parser.add_argument('--image_output_size', '-s_out', dest='size_out',
                    type=str, default='256,256', help='image size output')
parser.add_argument('--image-c', dest='im_c', default=3, type=int)
parser.add_argument('--img_type', '-t', dest='img_type',
                    metavar='TYPE', type=str, default='.png', help='image type: .png, .jpg ...')

# model
parser.add_argument('--epochs', '-e', metavar='E',
                    type=int, default=100, help='Number of epochs')
parser.add_argument('--batch-size', '-b', dest='batch_size',
                    metavar='B', type=int, default=8, help='Batch size')
parser.add_argument('--learning-rate', '-lr', metavar='LR', type=float, default=1e-3,
                    help='Learning rate', dest='lr')
parser.add_argument('--load', '-f', type=str,
                    default=False, help='Load model from a .pth file')
parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                    help='Percent of the data that is used as validation (0-100)')
parser.add_argument('--amp', action='store_true',
                    default=False, help='Use mixed precision')


# return parser.parse_args()
args = parser.parse_args()
# =======================================================================================================


def size_str_tuple(input):
    str_ = input.replace(' ', '').split(',')
    h, w = int(str_[0]), int(str_[1])
    return h, w


input_size = size_str_tuple(args.size_in)
output_size = size_str_tuple(args.size_out)

dir_img = Path(args.XX_DIR)
dir_mask = Path(args.YY_DIR)

time_ = datetime.now().strftime("%y%m%d_%H%M")
dir_checkpoint = Path(f'{args.SAVEDIR}/{time_}')
# dir_checkpoint = Path(f'./checkpoints/{time_}')
dir_checkpoint.mkdir(parents=True, exist_ok=True)

dir_benchmarks = Path('./data/data_for_Sup_train/benchmarks/')

# ----------------------------------------------------------------------------
# convert masks rgb(w,h,c)  to grey(w,h)
# dir_mask_tmp = Path('./data/data_for_Sup_train/masks_tmp')
# dir_mask_tmp.mkdir(parents=True, exist_ok=True)
# for i, path in enumerate(dir_mask.glob('*.png')):
#     mask_name = path.stem
#     mask_ = io.imread(path, as_gray=True)  # (h,w) uint8
#     mask_int8 = (mask_*255).astype(np.uint8)
#     mask_bi = np.where(mask_int8 == 0, 0, 255).astype('uint8')
#     save_path = dir_mask_tmp.joinpath(mask_name + '.png')
#     io.imsave(save_path, mask_bi)
#     print(i, mask_name, 'saved' )


# ## prepare dir_img by  dir_mask
# dir_ori = Path('../crop/origin/')
# for i, path in enumerate(dir_mask.iterdir()):
#     if not path.name.endswith('.png'):
#         continue
#     img_name = path.stem
#     origin_file = dir_ori.joinpath(img_name + '.png')
#     img = io.imread(origin_file)
#     save_path = dir_img.joinpath(img_name  + '.png')
#     io.imsave(save_path, img)
#     print(i, img_name, 'saved' )

# # ----------------------------------------------------------------------------

def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 8,
              learning_rate: float = 0.001,
              val_percent: float = 0.2,
              save_checkpoint: bool = True,
              input_size: tuple = (256, 256),
              output_size: tuple = (256, 256),
              img_type: str = '.png',
              amp: bool = False
              ):

    # Prepare dataset, Split into train / validation partitions
    img_paths = list(dir_img.glob('*' + img_type))
    mask_paths = list(dir_mask.glob('*' + img_type))

    assert len(img_paths) == len(
        mask_paths), f'number imgs: {len(img_paths)} and masks: {len(mask_paths)} need equal '

    df = get_data_attribute(img_paths)
    assert len(df.Source_Family) == len(img_paths) , f'number of imgs: {len(img_paths)} and data_attribute.csv: {len(df.Source_Family)} need equal '

    X_train, X_valid, y_train, y_valid = train_test_split(
        img_paths, mask_paths, test_size=val_percent, random_state=1, stratify=df.Source_Family)
    train_set = MothDataset(
        X_train, y_train, input_size=input_size, output_size=output_size, img_aug=True)
    val_set = MothDataset(
        X_valid, y_valid, input_size=input_size, output_size=output_size, img_aug=True)

    n_val = len(val_set)
    n_train = len(train_set)

    # Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False,
                            drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, input_size=input_size, output_size=output_size,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device}
        input_size:      {input_size}
        output_size:     {output_size}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(
        net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=20)  # goal: maximize Dice score > 'max' / minimize Valid Loss > 'min'
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # Parameters that should be stored.
    global params
    params = {}
    params['epoch'] = []
    params['step'] = []
    # params['time(h)'] =[]
    params['time(m)'] = []
    params['learning_rate'] = []
    params['train_loss'] = []
    params['valid_loss'] = []
    params['valid_dice'] = []

    # 5. Begin training
    best_loss_init = 1e3
    best_dice_init = 0
    patience = 50
    trigger_times = 0
    metric = 'max'
    start_time = time.time()
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            # ------------------------------------------------------------------------------------------
            # Train round
            # ------------------------------------------------------------------------------------------
            train_loss_collector = []
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # with torch.cuda.amp.autocast(enabled=amp):
                masks_pred = net(images)
                print('masks_pred : ', masks_pred.shape)
                print('true_masks : ', true_masks.shape)
                train_loss_batch = criterion(masks_pred, true_masks) \
                    + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, net.n_classes).permute(
                        0, 3, 1, 2).float(),
                    multiclass=True)

                # optimizer.zero_grad(set_to_none=True)
                # grad_scaler.scale(loss_train).backward()
                # grad_scaler.step(optimizer)
                # grad_scaler.update()
                optimizer.zero_grad()
                train_loss_batch.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                # epoch_loss += loss_train.item()
                train_loss_collector = np.append(
                    train_loss_collector, train_loss_batch.cpu().detach().numpy())

                pbar.set_postfix(**{'loss (batch)': train_loss_batch.item()})

            # ------------------------------------------------------------------------------------------
            # Evaluation round
            # ------------------------------------------------------------------------------------------
            # if global_step % (n_train // (10 * batch_size)) == 0:
            histograms = {}
            for tag, value in net.named_parameters():
                tag = tag.replace('/', '.')
                histograms['Weights/' +
                           tag] = wandb.Histogram(value.data.cpu())
                histograms['Gradients/' +
                           tag] = wandb.Histogram(value.grad.data.cpu())

            net.eval()
            val_score, valid_loss = evaluate(net, val_loader, device)
#             scheduler.step(val_score.cpu().numpy())

            # ---------------------------------------------------------------------------------------------------------------------------
            # Update and log parameters
            # ---------------------------------------------------------------------------------------------------------------------------
            # training log
            time_cost = time.time()-start_time
            train_loss = np.mean(train_loss_collector)
            curr_lr = optimizer.param_groups[0]['lr']
            params['epoch'] += [epoch]
            params['step'] += [global_step]
            # params['time(h)'] += [int(time_cost//(60*60))]
            params['time(m)'] += [time_cost/60]
            params['learning_rate'] += [curr_lr]
            params['train_loss'] += [train_loss]
            params['valid_loss'] += [valid_loss]
            params['valid_dice'] += [val_score.cpu().numpy()]
            pd.DataFrame.from_dict(params).to_csv(
                dir_checkpoint.joinpath('losses.csv'))

            global best_dice_score
            best_dice_score = max(params['valid_dice'])
            global best_val_loss
            best_val_loss = min(params['valid_loss'])
            global best_epoch
            best_epoch = np.argmin(params['valid_loss'])

            # learning curve
            if (epoch+1) % 5 == 0:
                sub = f'min val_loss {best_val_loss:.4f}, at epoch {best_epoch:2d}'
                fig = plt_learning_curve(
                    params['train_loss'], params['valid_loss'], title='Loss', sub=f'{dir_checkpoint} | {sub:s}')
                fig.savefig(dir_checkpoint.joinpath('loss.jpg'))
                print(f'\nLoss fig saved : {dir_checkpoint}')

            # get mask_pred
            threshold = 0.5
            full_mask_ = torch.softmax(masks_pred, dim=1)[
                0].float().clone().detach().cpu()
            full_mask = torch.tensor(full_mask_ > 0.5).float()

            logging.info(
                f'Valid - Dice_score: {val_score:.4f}, Loss:{valid_loss:.4f}')

            experiment.log({
                'learning rate': curr_lr,
                'train loss':  train_loss,
                'validation Dice': val_score,
                'validation Loss': valid_loss,
                'images': wandb.Image(images[0].cpu()),
                'masks': {
                    'true': wandb.Image(true_masks[0].float().cpu()),
                    # 'pred': wandb.Image(torch.softmax(masks_pred, dim=1)[0].float().cpu()),
                    'pred': wandb.Image(full_mask),
                },
                'step': global_step,
                'epoch': epoch,
                **histograms
            })

            # ---------------------------------------------------------------------------------------------------------------------------
            # get best model depends on valid_loss or val_score
            save_model_switch = False
            if metric == 'max':           # for maximize accuracy or val_score
                valid_value = val_score.cpu().numpy()
                if epoch == 0:
                    best_value = 0
                    cond =True
                else:
                    cond = best_value < valid_value
            elif metric == 'min':        # for minimize loss
                valid_value = valid_loss
                if epoch == 0:
                    best_value = 1e4
                    cond =True
                else:
                    cond = best_value > valid_value
            if cond:
                best_value = valid_value
                torch.save(net.state_dict(),
                           dir_checkpoint.joinpath('checkpoint.pth'))
                logging.info(f'Checkpoint saved! \
                    Best value updated: {best_value:,.4f},  Model Saved!')

            # ---------------------------------------------------------------------------------------------------------------------------

        # ---------------------------------------------------------------------------------------------------------------------------
        # Early stopping
        # ---------------------------------------------------------------------------------------------------------------------------
        trigger_times = early_stop(
            val_score.cpu().numpy(), best_dice_score, trigger_times, patience, metric=metric)
        if trigger_times >= patience:
            print('\nTrigger Early stopping!')
            break

# Save log


def save_log():
    summary_save = f'{args.SAVEDIR}/training_summary_pytorch.csv'

    # save into dictionary
    sav = vars(args)
    # sav['test_loss'] = test_loss
    # sav['Dice loss'] = mIoU
    sav['dir_checkpoint'] = dir_checkpoint
    sav['validation Dice'] = best_dice_score
    sav['best_val_loss'] = best_val_loss
    sav['best_epoch'] = best_epoch

    # Append into summary files
    dnew = pd.DataFrame(sav, index=[0])
    if os.path.exists(summary_save):
        dori = pd.read_csv(summary_save)
        dori = pd.concat([dori, dnew])
        dori.to_csv(summary_save, index=False)
    else:
        dnew.to_csv(summary_save, index=False)

    print(f'\n{summary_save} saved!')

# ============================================================================================================================


if __name__ == '__main__':
    # return parser
    # args = get_args()
    # set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=2, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  input_size=input_size,
                  output_size=output_size,
                  val_percent=args.val / 100,
                  img_type=args.img_type,
                  amp=args.amp
                  )
        save_log()
    except KeyboardInterrupt:
        torch.save(net.state_dict(),
                   dir_checkpoint.joinpath('INTERRUPTED.pth'))
        logging.info('Saved interrupt')
        save_log()
        sys.exit(0)


print('Training Ended')
