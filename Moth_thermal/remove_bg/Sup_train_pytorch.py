import argparse
import logging
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
import torchvision.utils as vutils
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.data_loading_moth import MothDataset
from utils.dice_score import dice_loss, loss_contour_weighted
from utils.evaluate import evaluate
from unet import UNet
from utils.utils import early_stop, plt_learning_curve
from utils.sampler_moth import ImgBatchAugmentSampler

# =======================================================================================================
# def get_args():
parser = argparse.ArgumentParser(
    description='Train the UNet on images and target masks')

# enviroment
parser.add_argument('--gpu', '-g', dest='gpu', default='0')
parser.add_argument("--save_epoch", '-save_e', type=int,
                    default=1, help="save checkpoint per epoch")

# data
parser.add_argument('--XX_DIR', '-x_dir', dest='XX_DIR', type=str,
                    default='../data/data_for_Sup_train/imgs')
parser.add_argument('--YY_DIR', '-y_dir', dest='YY_DIR', type=str,
                    default='../data/data_for_Sup_train/masks')
parser.add_argument('--SAVEDIR', '-save', dest='SAVEDIR', type=str,
                    default='model/Unet_rmbg')  # Unet_rmbg

parser.add_argument('--input_size', '-s_in', dest='size_in',
                    type=str, default='256,256', help='image size input')
parser.add_argument('--output_size', '-s_out', dest='size_out',
                    type=str, default='256,256', help='image size output')
parser.add_argument('--image_channel', '-c', dest='image_channel',
                    metavar='Channel', default=3, type=int, help='channel of image input')
parser.add_argument('--img_type', '-t', dest='img_type',
                    metavar='TYPE', type=str, default='.png', help='image type: .png, .jpg ...')
parser.add_argument('--stratify', action='store_true',
                    default=False, help='whether to stratified sampling')

# model
parser.add_argument('--epochs', '-e', metavar='E',
                    type=int, default=100, help='Number of epochs')
parser.add_argument('--batch-size', '-b', dest='batch_size',
                    metavar='B', type=int, default=8, help='Batch size')
parser.add_argument('--learning-rate', '-lr', dest='lr', metavar='LR', type=float, default=1e-3,
                    help='Learning rate')
parser.add_argument('--load', '-f', type=str,
                    default=False, help='Load model from a .pth file')
parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                    help='Percent of the data that is used as validation (0-100)')
parser.add_argument('--amp', action='store_true',
                    default=False, help='Use mixed precision')
parser.add_argument('--contour_weight', '-w', dest='contour_weight',
                    metavar='Contour_Weight', type=float, default=0, help='loss weight for contour area. must be positive')
parser.add_argument('--loss_metric', '-m', dest='loss_metric',
                    metavar='Metric', type=str, default='max',
                    help="Loss fuction goal: maximize Dice score > 'max' / minimize Valid Loss > 'min'")
parser.add_argument("--pretrained", '-p', default="",
                    type=str, help="path to pretrained model (default: none)")
parser.add_argument("--imgBatchArgMode", '-arg', default='', type=str,
                    help='Image Batch Argmentaion mode: "random", "mix" in ImgBatchAugmentSampler. \
                        \nIf you launch imgBatchArgMode, you need prepared data which you want batchArgmentation in \
                        data_for_Sup_train/imgs_batch_arg, masks_batch_arg ')


# return parser.parse_args()
args = parser.parse_args()

# =======================================================================================================

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


def size_str_tuple(input):
    str_ = input.replace(' ', '').split(',')
    h, w = int(str_[0]), int(str_[1])
    return h, w


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 8,
              learning_rate: float = 0.001,
              val_percent: float = 0.2,
              save_checkpoint: bool = True,
              amp: bool = False,
              metric: str = 'max',
              contour_weight: int = 3,
              arg_flag: str = ''
              ):

    # Prepare dataset, Split into train / validation partitions

    dir_img = Path(args.XX_DIR)
    dir_mask = Path(args.YY_DIR)
    img_paths = list(dir_img.glob('**/*' + args.img_type))
    mask_paths = list(dir_mask.glob('**/*' + args.img_type))

    assert len(img_paths) == len(
        mask_paths), f'number imgs: {len(img_paths)} and masks: {len(mask_paths)} need equal '

    if args.stratify:
        try:
            df = pd.read_csv('../data/imgs_label_byHSV.csv', index_col=0)
        except:
            print('You need provide label of imgs at "../data/imgs_label_byHSV.csv".')
        assert len(df.Name) == len(
            img_paths), f'number of imgs: {len(img_paths)} and imgs_label_byHSV.csv: {len(df.label)} need equal '
        print(
            f'Stratified sampling by "imgs_label_byHSV.csv", clustering: {np.unique(df.label).size}')

    X_train, X_valid, y_train, y_valid = train_test_split(
        img_paths, mask_paths, test_size=val_percent, random_state=1,
        stratify=df.label if args.stratify else None)

    train_set = MothDataset(
        X_train, y_train, input_size=args.size_in, output_size=args.size_out, img_aug=True)
    val_set = MothDataset(
        X_valid, y_valid, input_size=args.size_in, output_size=args.size_out, img_aug=False)

    n_val = len(val_set)
    n_train = len(train_set)

    # Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=2,
                       pin_memory=True, drop_last=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    # ------------------------------------------------------
    # AddBatchArgmentation(optional)

    if arg_flag:
        dir_imgs_arg = Path('../data/data_for_Sup_train/imgs_batch_arg')
        imgs_arg_paths = list(dir_imgs_arg.glob('**/*' + args.img_type))
        dir_masks_arg = Path('../data/data_for_Sup_train/masks_batch_arg')
        masks_arg_paths = list(dir_masks_arg.glob('**/*' + args.img_type))

        assert dir_masks_arg.exists() and dir_imgs_arg.exists(), f'\
            if you launch imgBatchArgMode, you need prepared data which you want batchArgmentation in {str(dir_imgs_arg)} and {str(dir_masks_arg)}'
        assert len(imgs_arg_paths) == len(
            masks_arg_paths), f'number of imgs_arg: {len(imgs_arg_paths)} and masks_arg: {len(masks_arg_paths)} need equal'

        val_name = [path.stem for path in X_valid]
        imgs_arg_paths = [
            path for path in imgs_arg_paths if path.stem not in val_name]
        masks_arg_paths = [
            path for path in masks_arg_paths if path.stem not in val_name]

        X_train_arg = X_train + imgs_arg_paths
        y_train_arg = y_train + masks_arg_paths
        size_X_train = len(X_train)

        # Sampler : single, multi, random, mix
        batchsampler = ImgBatchAugmentSampler(
            X_train_arg, size_X_train, batch_size, flag=arg_flag, sample_factor=1.0 if arg_flag == 'random' else 3.0)

        train_set = MothDataset(
            X_train_arg, y_train_arg, input_size=args.size_in, output_size=args.size_out, img_aug=True)
        train_loader = DataLoader(
            train_set, batch_sampler=batchsampler, num_workers=2, pin_memory=True)
        n_train = len(train_set)

    n_iter = len(train_loader)*batch_size

    dir_save_Argmentation = Path(f'tmp/Check_Argmentation_{arg_flag}')
    dir_save_Argmentation.mkdir(exist_ok=True, parents=True)

    # ------------------------------------------------------

    # (Initialize logging)
    experiment = wandb.init(project='U-Net_MothThermal_AddBatchArgmentationTest',
                            resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, input_size=args.size_in, output_size=args.size_out,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device}
        input_size:      {args.size_in}
        output_size:     {args.size_out}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
#     optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-3)
#     optimizer = optim.Adam(net.parameters(), lr=learning_rate)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, metric, patience=5)  # goal: maximize Dice score > 'max' / minimize Valid Loss > 'min'

    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    criterion = nn.CrossEntropyLoss()
    # shape = (b, h, w) if reduction ='none'
    # criterion_3d = nn.CrossEntropyLoss(reduction='none')
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
    patience = 11                # early_stop
    trigger_times = 0            # early_stop
    metric = metric
    warmup_epochs = 5
    start_time = time.time()
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0

        # ------------------------------------------------------------------------------------------
        # learning rate warmup(optional)

        if (epoch == warmup_epochs and not args.pretrained) or (epoch == 0 and args.pretrained):
            optimizer = optim.AdamW(
                net.parameters(), lr=learning_rate, weight_decay=1e-3)
            # goal: maximize Dice score > 'max' / minimize Valid Loss > 'min'
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, metric, patience=(patience//2))
        elif (epoch < warmup_epochs and not args.pretrained):
            print(f'\n lr waruping')
            warmup_percent_done = epoch/warmup_epochs
            # gradual warmup_lr
            warmup_learning_rate = learning_rate ** (1 /
                                               (warmup_percent_done + 1e-10))
            optimizer = optim.AdamW(net.parameters(), lr=warmup_learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, metric, patience=(patience//2))

        # ------------------------------------------------------------------------------------------

        with tqdm(total=n_iter, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            # ==========================================================================================
            # Train round
            # ==========================================================================================
            train_loss_collector = []
            for idx, batch in enumerate(train_loader):
                images = batch['image']
                masks_true = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                masks_true = masks_true.to(device=device, dtype=torch.long)

                # with torch.cuda.amp.autocast(enabled=amp):
                masks_pred = net(images)
                # (b, class, h, w)
                print('masks_pred : ', masks_pred.shape)
                # (b, h, w), torch.int64, [0, 1].
                print('masks_true : ', masks_true.shape)

                # --------------------------------------------------------------------------------------------

                # Choose loose function
                # 1. cross_entrophy + dice_loss
                if args.contour_weight == 0:
                    train_loss = criterion(masks_pred, masks_true) \
                        + dice_loss(F.softmax(masks_pred, dim=1).float(),             # (b, class, h, w)
                                    F.one_hot(masks_true, net.n_classes).permute(     # (b, class, h, w)
                            0, 3, 1, 2).float(),
                        multiclass=True)

                # 2. (cross_entrophy + dice_loss) weighted by countour
                else:
                    # get mask_contour and weight it for loss calculation (optional)
                    cross_entrophy_weighted, dice_loss_AllArea, dice_loss_Contour = loss_contour_weighted(
                        masks_true, masks_pred, net, device, contour_weight
                    )
                    train_loss = cross_entrophy_weighted + \
                        dice_loss_AllArea + dice_loss_Contour*contour_weight

                # optimizer.zero_grad(set_to_none=True)
                # grad_scaler.scale(loss_train).backward()
                # grad_scaler.step(optimizer)
                # grad_scaler.update()
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                # epoch_loss += loss_train.item()
                train_loss_collector = np.append(
                    train_loss_collector, train_loss.cpu().detach().numpy())

                pbar.set_postfix(
                    **{'loss (batch)': train_loss.item()})

            # ==========================================================================================
            # Evaluation round
            # ==========================================================================================
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
            scheduler.step(val_score.cpu().numpy())

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

            global best_dice_score, best_val_loss, best_epoch
            best_dice_score = max(params['valid_dice'])
            best_val_loss = min(params['valid_loss'])
            best_epoch = np.argmin(params['valid_loss'])

            # learning curve
            if (epoch+1) % 5 == 0:
                sub = f'min val_loss {best_val_loss:.4f}, at epoch {best_epoch:2d}'
                fig = plt_learning_curve(
                    params['train_loss'], params['valid_loss'], title='Loss', sub=f'{dir_checkpoint} | {sub:s}')
                fig.savefig(dir_checkpoint.joinpath('loss.jpg'))
                print(f'\nLoss fig saved : {dir_checkpoint}')

            # get mask_pred
            # threshold = 0.5
            full_mask = torch.softmax(
                masks_pred, dim=1).float().clone()
            full_mask = torch.tensor(full_mask > 0.5).float()[
                :, 1, ::]  # (b, c, h, w)

            logging.info(
                f'Valid - Dice_score: {val_score:.4f}, Loss:{valid_loss:.4f}')

            experiment.log({
                'learning rate': curr_lr,
                'train loss':  train_loss,
                'validation Dice': val_score,
                'validation Loss': valid_loss,
                'images': wandb.Image(images[0].cpu()),
                'masks': {
                    'true': wandb.Image(masks_true[0].float().cpu()),
                    # 'pred': wandb.Image(torch.softmax(masks_pred, dim=1)[0].float().cpu()),
                    'pred': wandb.Image(full_mask[0].detach().cpu()),
                },
                'step': global_step,
                'epoch': epoch,
                **histograms
            })

            # ------------------------------------------------------
            # Check Argmentaion and  Predict satatus
            if epoch % 10 == 0:
                vutils.save_image(
                    torch.cat([
                        images,
                        torch.stack([masks_true.float()]*3, dim=1),
                        torch.stack([full_mask]*3, dim=1)
                    ], dim=0).data.cpu(),
                    dir_save_Argmentation.joinpath(f'Epoch_{epoch}.jpg'),
                    nrow=batch_size, pad_value=1)
                print(f'Epoch_{epoch}.jpg saved')
            # ------------------------------------------------------

            # ---------------------------------------------------------------------------------------------------------------------------
            # get best model depends on valid_loss or val_score
            save_model_switch = False
            if metric == 'max':           # for maximize accuracy or val_score
                valid_value = val_score.cpu().numpy()
                if epoch == 0:
                    best_value = 1e-4
                    cond = True
                else:
                    cond = best_value < valid_value
            elif metric == 'min':        # for minimize loss
                valid_value = valid_loss
                if epoch == 0:
                    best_value = 1e4
                    cond = True
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


# ============================================================================================================================

if __name__ == '__main__':
    # return parser
    # args = get_args()
    # set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    input_size, output_size = size_str_tuple(
        args.size_in), size_str_tuple(args.size_out)

    time_ = datetime.now().strftime("%y%m%d_%H%M")
    dir_checkpoint = Path(f'{args.SAVEDIR}/{time_}')
    dir_checkpoint.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=args.image_channel, n_classes=2, bilinear=True)
    net.to(device=device)

    # whether train from scratch
    if args.pretrained:  # '' : False
        logging.info(f'Model loaded from {args.pretrained}')
        weights = torch.load(args.pretrained, map_location=device)
        net.load_state_dict(weights)
    else:
        logging.info('train from scratch')

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  val_percent=args.val / 100,
                  amp=args.amp,
                  metric=args.loss_metric,
                  contour_weight=args.contour_weight,
                  arg_flag=args.imgBatchArgMode
                  )
        save_log()
    except KeyboardInterrupt:
        torch.save(net.state_dict(),
                   dir_checkpoint.joinpath('INTERRUPTED.pth'))
        logging.info('Saved interrupt')
        save_log()
        sys.exit(0)


print('Training Ended')
