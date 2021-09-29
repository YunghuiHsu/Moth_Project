import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from imgaug import augmenters as iaa


def sometimes(aug): return iaa.Sometimes(0.5, aug)
def most_of_the_time(aug): return iaa.Sometimes(0.9, aug)
def usually(aug): return iaa.Sometimes(0.75, aug)
def always(aug): return iaa.Sometimes(1, aug)
def charm(aug): return iaa.Sometimes(0.33, aug)
def seldom(aug): return iaa.Sometimes(0.2, aug)


augseq_shape = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        most_of_the_time(
            iaa.Affine(
                # scale images to 90-110% of their size, individually per axis
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-15, 15),  # rotate by -10 to +10 degrees
                shear=(-10, 10),
                mode='constant',
            )
        ),
    ]
)


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(
            images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(
                f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH))
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
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


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1.0):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')


class TransformDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1.0, mask_suffix='', aug=True):
        # super().__init__(images_dir, masks_dir, scale, mask_suffix='')
        super(TransformDataset, self).__init__()
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.aug = aug
        print('__init__ self.aug :', self.aug)
    
    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(
            mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(
            img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        pil_mask = self.load(mask_file[0])
        pil_img = self.load(img_file[0])
        pil_img = pil_img.convert('RGB')

        assert pil_img.size == pil_mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img_ = self.preprocess(pil_img, self.scale, is_mask=False)
        mask_ = self.preprocess(pil_mask, self.scale, is_mask=True)

        print('__getitem__ self.aug :', self.aug)

        if self.aug:
            # img = augseq_all.augment_images(img_)
            # images_aug = seq(images=images)
            img = augseq_shape(img_)
            mask = augseq_shape(mask_)

            # img , mask = shape(img_, mask_)
            # img = color(img)
            # img = noise(img)

        else:
            img = img_
            mask = mask_

        # ndarray to tensor
        img = torch.as_tensor(img.copy()).float().contiguous()
        mask = torch.as_tensor(mask.copy()).long().contiguous()

        return {
            'image': img,
            'mask': mask
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
