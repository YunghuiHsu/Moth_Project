import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)



# -----------------------------------------------------------------------------------
# get mask_contour and weight it for loss calculation 
# -----------------------------------------------------------------------------------
def get_masks_contour(masks: np.ndarray, iterations: int = 5) -> torch.Tensor:
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    assert masks.ndim == 3, f'Shape of mask must be (h,w) or (batch, h, w), mask.shape : {masks.shape}'
    assert masks.dtype == 'uint8', f'dtype of mask need to be "uint8", masks.dtype : {masks.dtype}.\nuse .astype("uint8") convert dtype'

    kernel_ELLIPSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3, 3))

    masks_contour = []
    for mask_ in masks:
        mask_dilate = cv2.dilate(mask_, kernel_ELLIPSE, iterations=6)
        mask_erode = cv2.erode(mask_, kernel_ELLIPSE, iterations=iterations)
        mask_contour = mask_dilate - mask_erode
        masks_contour.append(mask_contour)

    masks_contour_tensor = torch.tensor(masks_contour, dtype=torch.long)

    return masks_contour_tensor


def loss_contour_weighted(
    masks_true:torch.TensorType, masks_pred:torch.TensorType, net, device, contour_weight:int = 3
    ) -> tuple():
    '''
    masks_true :: torch.Tensor.(b, h, w), torch.int64, [0, 1]
    masks_pred :: torch.Tensor.(b, class, h, w)
    net :: UNet(n_channels=args.image_channel, n_classes=2, bilinear=True)
    contour_weight :: int. 'loss weight for contour area'
    '''

    criterion_3d = nn.CrossEntropyLoss(reduction='none')

    masks_contour = get_masks_contour(
        masks_true.cpu().numpy().astype(np.uint8)).to(device)  # (b, h, w), torch.int64, [0, 1]

    y = torch.ones(1, dtype=torch.int64).to(device)
    masks_weighted = torch.where(
        masks_contour == 1, contour_weight*y, y)  # (b, h, w)

    # caculate weighted loss based on countour
    masks_pred_one_hot = F.softmax(
        masks_pred, dim=1, dtype=torch.float32)                 # (b, class, h, w)
    masks_true_one_hot = F.one_hot(
        masks_true, net.n_classes).float().permute(0, 3, 1, 2)  # (b, h, w) > (b, h, w, class) > (b, class, h, w)

    masks_contour_pred_one_hot = masks_pred_one_hot * \
        torch.stack((masks_contour, masks_contour), dim=1)
    masks_contour_true_one_hot = masks_true_one_hot * \
        torch.stack((masks_contour, masks_contour), dim=1)

    cross_entrophy_weighted = (criterion_3d(                                        # nn.CrossEntropyLoss(reduction='none') : (b, h, w)
        masks_pred, masks_true) * masks_weighted).mean()
    dice_loss_AllArea = dice_loss(
        masks_pred_one_hot, masks_true_one_hot, multiclass=True)
    dice_loss_Contour = dice_loss(
        masks_contour_pred_one_hot, masks_contour_true_one_hot, multiclass=True)

    return cross_entrophy_weighted, dice_loss_AllArea, dice_loss_Contour