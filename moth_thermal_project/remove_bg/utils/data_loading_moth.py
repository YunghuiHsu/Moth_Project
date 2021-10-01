import logging
import os
from os import listdir
from os.path import splitext
import random
from pathlib import Path
from imgaug.augmenters.geometric import Rotate, TranslateX, TranslateY

import numpy as np
import torch
from PIL import Image
import skimage.io as io
from torch._C import dtype
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms.functional as F
from imgaug import augmenters as iaa

# =========================================================================================
# img augmentation


def img_aug_shape(img, mask):
    '''
    img.shape == (h, w, c) or (h,w) and img.dtype : uint8
    '''
    # flip horizontal
    aug_flip = iaa.Fliplr(1)

    # shape transform
    aug_seq = iaa.Sequential([
        # rotate by -10 to +10 degrees
        iaa.Rotate(random.uniform(-1, 1)*15),
        iaa.Affine(shear=random.uniform(-1, 1)*15),
        # scale images to 90-110% of their size, individually per axis
        iaa.Affine(scale=random.uniform(*[0.95, 1.05])),
        iaa.Affine(translate_percent=random.uniform(*[-0.05, 0.05]))
    ])
    if random.random() > 0.5:
        img = aug_flip.augment_image(img)
        mask = aug_flip.augment_image(mask)

    img = aug_seq.augment_image(img)
    mask = aug_seq.augment_image(mask)
    return img, mask


def img_aug_color_noise(img):
    '''
    img.shape == (n, h, w, c) or (h, w) and img.dtype : uint8
    '''
    # add coarse(rectangle shape) noise
    # size_percent : drop them on an image with min - max% percent of the original size
    aug_noise = iaa.CoarseDropout(p=(0.005, 0.05), size_percent=(.01, .5), per_channel=0.5)

    aug_color = iaa.Sequential([
        iaa.Multiply((0.9, 1.1), per_channel=0.05),
        iaa.LinearContrast((0.9, 1.1), per_channel=0.1)
    ])
    # img_ = img.transpose((1, 2, 0))  # (c, h, w) > (h, w, c)
    # img_ = (img*255).astype('uint8')

    img = aug_noise.augment_image(img)
    img = aug_color.augment_image(img)

    # img_ = img_.transpose((2, 0, 1))  # (h, w, c) > (c, h, w)
    return img
# =========================================================================================


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.names = [splitext(file)[0] for file in listdir(
            images_dir) if not file.startswith('.')]
        if not self.names:
            raise RuntimeError(
                f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.names)} examples')

    def __len__(self):
        return len(self.names)

    @ classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH))
        img_ndarray = np.asarray(pil_img)  # (w, h, c))

        if img_ndarray.ndim == 2 and not is_mask:       # (w, h) > (1, w, h)
            img_ndarray = img_ndarray[np.newaxis, ...]
        # (w, h, c) >  (c, w, h)
        elif not is_mask:
            img_ndarray = img_ndarray.transpose(
                (2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255  # dtype: 'uint8' > 'float64'

        return img_ndarray

    @ classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.names[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(
            mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(
            img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])
        img = img.convert('RGB')

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        img = torch.as_tensor(img.copy()).float().contiguous()
        mask = self.preprocess(mask, self.scale, is_mask=True)
        mask = torch.as_tensor(mask.copy()).long().contiguous()

        return {
            'image': img,
            'mask': mask
        }

class MothDataset(Dataset):
    def __init__(self, imgs_paths: list, masks_paths: list,
                 input_size: tuple = (256, 256), output_size: tuple = (256, 256),
                 mask_suffix: str = '', img_aug: bool = True):
        self.imgs_paths = imgs_paths
        self.masks_paths = masks_paths
        self.input_size = input_size
        self.output_size = output_size
        self.img_aug = img_aug

        self.names = [os.path.basename(path).split('.')[0] for path in imgs_paths
                      if not os.path.basename(path).startswith('.')]
        if not self.names:
            raise RuntimeError(
                f'No input file found in {imgs_paths}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.names)} examples')

        assert len(imgs_paths) == len(
            masks_paths), f'number of masks: {len(masks_paths)} need to be same as images: {len(imgs_paths)}'

    def __len__(self):
        return len(self.names)

    @ classmethod
    def preprocess(cls, img_ndarray, input_size, output_size, is_mask):
        '''
        ndarray to tensor

        '''
        # resize
        if input_size != output_size:
            img_ndarray = np.asarray(
                Image.fromarray(img_ndarray).resize(output_size), dtype=np.uint8)

        # (w, h) > (1, w, h)
        # (w, h, c) >  (c, w, h)
        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose(
                (2, 0, 1))

        if not is_mask:    # dtype: np.uint8 > tensor.float32
            img_ndarray = img_ndarray / 255
            img_tensor = torch.as_tensor(
                img_ndarray.copy()).float().contiguous()
        elif is_mask:              # dtype: np.uint8 > tensor.int64
            img_tensor = torch.as_tensor(
                img_ndarray.copy()).long().contiguous()
        return img_tensor

    def __getitem__(self, idx):
        name = self.names[idx]
        img_ = io.imread(self.imgs_paths[idx])   # (h, w, c)
        # (h, w),  dtype = float64
        mask_ = io.imread(self.masks_paths[idx], as_gray=True)   # as_gray=True > load as (h, w),  [0, 1], dtype = float64 
        mask_ = (mask_*255).astype('uint8')                      #  [0, 1], dtype = float64  >　[0, 255], dtype = uint8 

        if self.img_aug:
            # img.shape == (h, w, c) or (h, w) and img.dtype : uint8
            img_, mask_ = img_aug_shape(img_, mask_)
            img_ = img_aug_color_noise(img_)

        # ndarray to tensor
        img = self.preprocess(img_, self.input_size, self.output_size, False)
        mask = self.preprocess(mask_, self.input_size, self.output_size, True)

        return {
            'image': img,
            'mask': mask,
        }

# ===================================================================================================
# class MouseDataset(Dataset):
#     """
#     Required directory configuration:
#         data_dir
#         ├─ img/
#         └─ mask/

#     Requiring the same name of image and mask.
#     """
#     # augmentation config
#     angle = 90
#     translate = 0.2
#     scale = [0.7, 1.3]
#     shear = 20
#     brightness = 0.3
#     contrast = 0.3

#     def __init__(self, img_list, config, is_train=True):
#         self.img_list = img_list
#         self.mask_list = [p.replace('/img/', '/mask/') for p in img_list]
#         self.is_train = is_train
#         self.input_size = config['input_size']

#         self._color_jitter = transforms.ColorJitter(
#             brightness=self.brightness,
#             contrast=self.contrast
#         )

#     def _random_affine(self, img, mask):
#         a = random.uniform(-1, 1)*self.angle
#         w = int(random.uniform(0, self.translate)*self.input_size[0])
#         h = int(random.uniform(0, self.translate)*self.input_size[1])
#         b = random.uniform(*self.scale)
#         c = random.uniform(-1, 1)*self.shear

#         img = transforms.functional.affine(
#             img,
#             angle=a,
#             translate=[w, h],
#             scale=b,
#             shear=c
#         )
#         mask = transforms.functional.affine(
#             mask,
#             angle=a,
#             translate=[w, h],
#             scale=b,
#             shear=c
#         )

#         if random.random() > 0.5:
#             img = transforms.functional.hflip(img)
#             mask = transforms.functional.hflip(mask)

#         if random.random() > 0.5:
#             img = transforms.functional.vflip(img)
#             mask = transforms.functional.vflip(mask)

#         return img, mask

#     def __getitem__(self, index):
#         img = Image.open(self.img_list[index]).resize(self.input_size)
#         img = img.convert('RGB')
#         mask = Image.open(self.mask_list[index]).resize(self.input_size)

#         if self.is_train:
#             img, mask = self._random_affine(img, mask)
#             img = self._color_jitter(img)

#         img = to_tensor(img)
#         mask = to_tensor(mask)

#         return img, mask

#     def __len__(self):
#         return len(self.img_list)
