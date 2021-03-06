import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from utils.dice_score import multiclass_dice_coeff, dice_coeff, dice_loss

criterion = nn.CrossEntropyLoss()


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    epoch_loss = 0

    # iterate over the validation set
    valid_loss_collector = []
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']

        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true_ = F.one_hot(mask_true, net.n_classes).permute(
            0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred,
                                         mask_true, reduce_batch_first=False)

                # cpmpute the validation loss
                val_loss_batch = criterion(mask_pred, mask_true_) \
                    + dice_loss(F.softmax(mask_pred, dim=1).float(),
                                F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(), multiclass=False)

            else:
                mask_pred = F.one_hot(mask_pred.argmax(
                    dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(
                    mask_pred[:, 1:, ...], mask_true_[:, 1:, ...], reduce_batch_first=False)

                # cpmpute the validation loss
                val_loss_batch = criterion(mask_pred, mask_true) \
                    + dice_loss(F.softmax(mask_pred, dim=1).float(),
                                F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(), multiclass=True)

            # epoch_loss += loss.item()
            valid_loss_collector = np.append(
                valid_loss_collector, val_loss_batch.cpu().detach().numpy())
            valid_loss = np.mean(valid_loss_collector)

    net.train()
    return (dice_score / num_val_batches), valid_loss
