import argparse
import logging
import os
from pathlib import Path
from datetime import datetime
from matplotlib.pyplot import axis

import numpy as np
import torch
from torch._C import dtype
import torch.nn.functional as F
from PIL import Image
import skimage.io as io
from skimage.transform import resize
from torchvision import transforms

from utils.data_loading_moth import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask, plt_result
# ==================================================================================


def get_args():
    parser = argparse.ArgumentParser(
        description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='model/Unet_rmbg/checkpoint.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT',
                        nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true',
                        help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')
    parser.add_argument('--gpu', '-g', dest='gpu', default='0')
    parser.add_argument('--mask_suffix', '-suffix', type=str, default='_MaskUnet',
                        help='suffix name of mask predicted')
    return parser.parse_args()

# ===================================================================================


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(
        full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()

# def get_output_filenames(args):
#     def _generate_name(fn):
#         split = os.path.splitext(fn)
#         return f'{split[0]}{args.mask_suffix}{split[1]}'
#     return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        mask_ = (mask * 255).astype(np.uint8)
        # return Image.fromarray(mask_)
        return mask_
    elif mask.ndim == 3:
        # (n, h ,w) > (h, w)
        mask_ = (np.argmax(mask, axis=0) * 255 /
                 mask.shape[0]).astype(np.uint8)
        # return Image.fromarray(mask_)
        return mask_

# ==========================================================================================================================


if __name__ == '__main__':
    time_ = datetime.now().strftime("%y%m%d_%H%M")
    args = get_args()
    in_files = args.input

    # mkdir
    submodel_path = Path(in_files[0]).parent
    submodel_dir = os.path.split(submodel_path)[-1]
    model_dir = Path(args.model).parent.joinpath(time_ + '_' + submodel_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_to_check = model_dir.joinpath('Predict_checking')
    save_to_check.mkdir(exist_ok=True)
    save_to_mask = model_dir.joinpath('Predict_mask')
    save_to_mask.mkdir(exist_ok=True)
    save_to_rmbg = model_dir.joinpath('Predict_rmbg')
    save_to_rmbg.mkdir(exist_ok=True)

    net = UNet(n_channels=3, n_classes=2)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    for idx, filename in enumerate(in_files):
        logging.info(f'\t{idx:4d}, Predicting image {filename} ...')

        # img_pil = Image.open(filename)
        print(f'{idx:4d} loading {filename}')
        img_ndarray = io.imread(filename)
        img_pil = Image.fromarray(img_ndarray)

        mask = predict_img(net=net,
                           full_img=img_pil,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        # print('mask.shape: ', mask.shape) # (2, 256, 256), [0, 255], uint8

        if not args.no_save:

            # out_filename = out_files[i]
            fname = Path(filename).stem
            logging.info(f'Mask saved to {fname}')

            # loading original img
            # (h,w,c).  [0, 255], uint8 > [0, 1], float64
            ori_img = img_ndarray/255
            # ori_img = resize(ori_img, output_shape=(256, 256, 3))

            # transform mask to black / white
            # (n, h ,w), [0, 1], float64 > (h, w), [0, 255], uint8
            m_mask = mask_to_image(mask)
            norm_mask = (m_mask-m_mask.min()) / (m_mask.max() -
                                                 m_mask.min())  # (h ,w, 3), [0,1], flaot64
            norm_mask3 = np.stack(
                [norm_mask]*3, axis=2)   # (h ,w, 3)
            # transform mask to black(0) / white(1) depends on threshold
            # (h ,w, 3), [0/1], flaot64
            bin_mask3 = np.where(norm_mask3 > 0.5, 1.0, 0.0)

            # make color mask
            white_mask = 1-bin_mask3
            blue_mask = np.zeros_like(bin_mask3)
            # assign blue channel as 1.0 by w_mask
            blue_mask[..., 2] = white_mask[..., 2]
            color_mask = blue_mask
            # (h ,w, 3), [0, 1], flaot64
            img_rmbg = (bin_mask3 * ori_img) + color_mask

            # ploting checking fig
            img_list = [ori_img, bin_mask3, img_rmbg]
            title_list = ['Original image', 'U-Net mask', 'img_rmbg']
            fig = plt_result(img_list, title_list)
            path_fig_save = save_to_check.joinpath(
                fname + '__checking_Unet.jpg')
            fig.savefig(path_fig_save, dpi=100, bbox_inches='tight')

            # save mask
            # (h, w), [0/255], uint8
            binary_mask = (bin_mask3[:, :, 0]*255).astype('uint8')
            path_mask_save = save_to_mask.joinpath(fname + '_mask_Unet.png')
            io.imsave(path_mask_save, binary_mask)

            # save img_rmbg
            # (h, w, c). [0,1] float64 > [0,255], uint8
            img_rmbg_uint8 = (img_rmbg*255).astype('uint8')
            path_img_rmbg_save = save_to_rmbg.joinpath(
                fname + '_rmbg_Unet.png')
            io.imsave(path_img_rmbg_save, img_rmbg_uint8)

            print(f'{idx:4d} pred_mask of {fname} saved')

        if args.viz:
            logging.info(
                f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img_pil, mask)
