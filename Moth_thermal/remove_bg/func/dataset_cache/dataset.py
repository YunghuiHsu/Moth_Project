import torch
import torch.utils.data as data

#import random
import numpy as np
import torchvision.transforms as transforms

import skimage.io as io
from imgaug import augmenters as iaa

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
most_of_the_time = lambda aug: iaa.Sometimes(0.9, aug)
usually = lambda aug: iaa.Sometimes(0.75, aug)
always = lambda aug: iaa.Sometimes(1, aug)
charm = lambda aug: iaa.Sometimes(0.33, aug)
seldom = lambda aug: iaa.Sometimes(0.2, aug)

augseq_all = iaa.Sequential([
    iaa.Fliplr(0.5),
    most_of_the_time(iaa.Affine(
            #translate_px=(0,5), # translate by -10 to +10 percent (per axis)
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 90-110% of their size, individually per axis
            mode='symmetric',
        )),
    iaa.Multiply((0.9, 1.1), per_channel=0.3),
    iaa.ContrastNormalization((0.9, 1.1), per_channel=0.3),
])

def rotation_ratio():
    thickness = 0.4
    rand = np.random.rand()
    if rand >= 0 and rand < thickness:
        return augseq_noise_1
    elif rand >= thickness and rand < (thickness * 2):
        return augseq_noise_2
    else:
        return augseq_noise


augseq_noise = iaa.Sequential([
    usually(iaa.Affine(
            #scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 90-110% of their size, individually per axis
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -10 to +10 percent (per axis)
            rotate=(-90,90), # rotate by -10 to +10 degrees
            mode='symmetric',
        )),
    charm(iaa.CoarseDropout(p=0.11, size_percent=0.033)),
    charm(iaa.CoarseDropout(p=0.11, size_percent=0.067)),
    charm(iaa.CoarseDropout(p=0.11, size_percent=0.1))
])

augseq_noise_1 = iaa.Sequential([
    usually(iaa.Affine(
            #scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 90-110% of their size, individually per axis
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -10 to +10 percent (per axis)
            rotate=(-180,-90), # rotate by -10 to +10 degrees
            mode='symmetric',
        )),
    charm(iaa.CoarseDropout(p=0.11, size_percent=0.033)),
    charm(iaa.CoarseDropout(p=0.11, size_percent=0.067)),
    charm(iaa.CoarseDropout(p=0.11, size_percent=0.1))
])

augseq_noise_2 = iaa.Sequential([
    usually(iaa.Affine(
            #scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 90-110% of their size, individually per axis
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -10 to +10 percent (per axis)
            rotate=(90,180), # rotate by -10 to +10 degrees
            mode='symmetric',
        )),
    charm(iaa.CoarseDropout(p=0.11, size_percent=0.033)),
    charm(iaa.CoarseDropout(p=0.11, size_percent=0.067)),
    charm(iaa.CoarseDropout(p=0.11, size_percent=0.1))
])


class ImageDatasetFromFile(data.Dataset):
    def __init__(self, image_list,
                input_height=256, input_width=256, output_height=256, output_width=256,
                aug=True):
        
        super(ImageDatasetFromFile, self).__init__()
                
        self.image_filenames = image_list 
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.aug = aug
                       
        self.input_transform = transforms.Compose([ 
                                   transforms.ToTensor()                                                                      
                               ])

    def __getitem__(self, index):
          
        img_ = io.imread(self.image_filenames[index])
        if self.aug:
            denoised_img = augseq_all.augment_images([img_])
            img = rotation_ratio().augment_images(denoised_img)
        else:
            img = [img_]
            denoised_img = img

        img = self.input_transform(img[0])
        denoised_img = self.input_transform(denoised_img[0])
        
        return img, denoised_img, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)


class ImageDatasetFromCache(data.Dataset):
    def __init__(self, image_cache,
                input_height=256, input_width=256, output_height=256, output_width=256,
                aug=True):
        
        super(ImageDatasetFromCache, self).__init__()
                
        #self.image_filenames = image_list 
        self.image_cache = image_cache
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.aug = aug
                       
        self.input_transform = transforms.Compose([ 
                                   transforms.ToTensor()                                                                      
                               ])

    def __getitem__(self, index):
          
        img_ = self.image_cache[index]
        if self.aug:
            denoised_img = augseq_all.augment_image(img_)
            img = rotation_ratio().augment_image(denoised_img)
        else:
            img = img_
            denoised_img = img

        img = self.input_transform(img)
        denoised_img = self.input_transform(denoised_img)
        
        return img, denoised_img, ''

    def __len__(self):
        return len(self.image_cache)

