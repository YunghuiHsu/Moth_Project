import random
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class ImgBatchAugmentSampler(Sampler):
    '''
    X_train_arg:: X_train + img_arg

    flag::
        -'single':  same image(from img_arg)  in a batch with different data argmentation.\n
        -'multi': similar group but dirrerent images(from img_arg) in a batch with different data argmentation.\n
        -'random': random sampled from X_train and img_arg in a batch with different data argmentation.\n
        -'mix': mix with SingleImgBatch and MultiImgBatch.\n

    sample_factor:: len(img_arg) * sample_factor
    '''

    def __init__(self, X_train_arg: list, size_X_train: int, batch_size: int, sample_factor: float = 1.0, flag: str = 'single'):

        self.batch_size = batch_size
        self.split_point = size_X_train

        self.X_train_indices = list(range(self.split_point))
        random.shuffle(self.X_train_indices)

        self.img_arg_indices = list(range(self.split_point, len(X_train_arg)))

        self.n_sample = int(
            sample_factor * (len(self.img_arg_indices) // self.batch_size))
        self.flag = flag

        # 'single' : sample n from img_arg_indices, and repeat data be sampled by batch_size
        self.indices_sample_s = np.random.choice(
            self.img_arg_indices, self.n_sample)
        self.indices_sample_s = np.repeat(
            self.indices_sample_s, self.batch_size)



        # 'multi': sample n from data_indices
        self.indices_sample_m = np.random.choice(
            self.img_arg_indices, self.n_sample*self.batch_size)

        if self.flag == 'single':
            self.indices_sample = self.indices_sample_s
        elif self.flag == 'multi':
            self.indices_sample = self.indices_sample_m
        elif self.flag == 'mix':
            # mix with 'single' and 'multi'
            slice = self.batch_size * round((self.n_sample*self.batch_size)*0.5)
            self.indices_sample = self.indices_sample_s[:int(slice)] + \
                self.indices_sample_m[:int(slice)]

        elif self.flag == 'Random':
            all = self.img_arg_indices + self.X_train_indices
            self.indices_sample = np.random.choice(
                all, self.n_sample*self.batch_size
            )

    def __iter__(self):
        def chunk(indices, chunk_size):
            return torch.split(torch.tensor(indices), chunk_size)

        # put data as batchs
        X_train_indices_batches = chunk(self.X_train_indices, self.batch_size)
        indices_sample_batches = chunk(self.indices_sample, self.batch_size)
        batches = list(X_train_indices_batches + indices_sample_batches)
        batches = [batch.tolist() for batch in batches]
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return len(self.X_train_indices)//self.batch_size + len(self.indices_sample)//self.batch_size


# class MultiImgBatchAugmentSampler(Sampler):
#     '''
#     SingleImgBatch:  same image in a batch with different data argmentation.
#     MultiImgBatch:   similar kind but dirrerent images in a batch with different data argmentation.
#     '''

#     def __init__(self, X_train_arg: list, size_X_train: int, batch_size: int, sample_factor: float = 1.0):
#         self.batch_size = batch_size
#         self.split_point = size_X_train
#         self.sample_factor = sample_factor

#         self.X_train_indices = list(range(self.split_point))
#         random.shuffle(self.X_train_indices)

#         self.img_arg_indices = list(range(self.split_point, len(X_train_arg)))
#         self.n_sample = int(size_X_train*0.01)
#         assert self.n_sample*self.batch_size <= len(
#             self.img_arg_indices), f'n_sample*batch_size({self.n_sample}*{self.batch_size}) is large than img_arg dataset({len(self.img_arg_indices)})'

#     def __iter__(self):
#         def chunk(indices, chunk_size):
#             return torch.split(torch.tensor(indices), chunk_size)

#         # sample n from data_indices
#         self.img_arg_indices_sample = random.sample(
#             self.img_arg_indices, self.n_sample*self.batch_size)
#         # self.img_arg_indices_sample.sort()

#         # put data as batchs
#         X_train_indices_batches = chunk(self.X_train_indices, self.batch_size)
#         img_arg_indices_sample_batches = chunk(
#             self.img_arg_indices_sample, self.batch_size)
#         batches = list(X_train_indices_batches +
#                        img_arg_indices_sample_batches)
#         batches = [batch.tolist() for batch in batches]
#         random.shuffle(batches)
#         return iter(batches)

#     def __len__(self):
#         return len(self.X_train_indices)//self.batch_size + self.n_sample


# class SingleImgBatchAugmentSampler(Sampler):
#     '''
#     SingleImgBatch:  same image in a batch with different data argmentation.
#     MultiImgBatch:   similar kind but dirrerent images in a batch with different data argmentation.
#     '''

#     def __init__(self, X_train_arg: list, size_X_train: int, batch_size: int, flag: str = 'SingleImg', sample_factor: float = 1.0):
#         self.flag = flag
#         self.batch_size = batch_size
#         self.split_point = size_X_train

#         self.X_train_indices = list(range(self.split_point))
#         random.shuffle(self.X_train_indices)

#         self.img_arg_indices = list(range(self.split_point, len(X_train_arg)))
#         self.n_sample = int(
#             sample_factor * len(self.img_arg_indices) // self.batch_size
#         )

#     def __iter__(self):
#         def chunk(indices, chunk_size):
#             return torch.split(torch.tensor(indices), chunk_size)

#         # sample n from img_arg_indices, and repeat data be sampled by batch_size
#         self.img_arg_indices_sample = random.sample(
#             self.img_arg_indices, self.n_sample)
#         img_arg_indices_sample = np.repeat(
#             self.img_arg_indices_sample, self.batch_size)

#         # put data as batchs
#         X_train_indices_batches = chunk(self.X_train_indices, self.batch_size)
#         img_arg_indices_batches = chunk(
#             img_arg_indices_sample, self.batch_size)
#         batches = list(X_train_indices_batches + img_arg_indices_batches)
#         batches = [batch.tolist() for batch in batches]
#         random.shuffle(batches)
#         return iter(batches)

#     def __len__(self):
#         return len(self.X_train_indices)//self.batch_size + self.n_sample


# class MultiImgBatchAugmentSampler(Sampler):
#     '''
#     SingleImgBatch:  same image in a batch with different data argmentation.
#     MultiImgBatch:   similar kind but dirrerent images in a batch with different data argmentation.
#     '''

#     def __init__(self, X_train_arg: list, size_X_train: int, batch_size: int, sample_factor: float = 1.0):
#         self.batch_size = batch_size
#         self.split_point = size_X_train
#         self.sample_factor = sample_factor

#         self.X_train_indices = list(range(self.split_point))
#         random.shuffle(self.X_train_indices)

#         self.img_arg_indices = list(range(self.split_point, len(X_train_arg)))
#         self.n_sample = int(size_X_train*0.01)
#         assert self.n_sample*self.batch_size <= len(
#             self.img_arg_indices), f'n_sample*batch_size({self.n_sample}*{self.batch_size}) is large than img_arg dataset({len(self.img_arg_indices)})'

#     def __iter__(self):
#         def chunk(indices, chunk_size):
#             return torch.split(torch.tensor(indices), chunk_size)

#         # sample n from data_indices
#         self.img_arg_indices_sample = random.sample(
#             self.img_arg_indices, self.n_sample*self.batch_size)
#         # self.img_arg_indices_sample.sort()

#         # put data as batchs
#         X_train_indices_batches = chunk(self.X_train_indices, self.batch_size)
#         img_arg_indices_sample_batches = chunk(
#             self.img_arg_indices_sample, self.batch_size)
#         batches = list(X_train_indices_batches +
#                        img_arg_indices_sample_batches)
#         batches = [batch.tolist() for batch in batches]
#         random.shuffle(batches)
#         return iter(batches)

#     def __len__(self):
#         return len(self.X_train_indices)//self.batch_size + self.n_sample


# class RandomImgBatchAugmentSampler(Sampler):
#     def __init__(self, X_train_arg: list, size_X_train: int, batch_size: int, sample_factor: float = 1.0):
#         self.batch_size = batch_size
#         self.split_point = size_X_train
#         self.sample_factor = sample_factor

#         self.X_train_arg_indices = list(range(len(self.X_train_arg)))
#         random.shuffle(self.X_train_arg_indices)

#         self.n_sample = int(size_X_train*self.sample_factor)

#     def __iter__(self):
#         def chunk(indices, chunk_size):
#             return torch.split(torch.tensor(indices), chunk_size)

#         # sample n from data_indices
#         self.X_train_arg_indices_sample = random.sample(
#             self.X_train_arg_indices_indices, self.n_sample*self.batch_size)
#         # self.img_arg_indices_sample.sort()

#         # put data as batchs
#         X_train_indices_batches = chunk(self.X_train_indices, self.batch_size)
#         X_train_arg_indices_sample_batches = chunk(
#             self.X_train_arg_indices_sample, self.batch_size)
#         batches = list(X_train_indices_batches +
#                        X_train_arg_indices_sample_batches)
#         batches = [batch.tolist() for batch in batches]
#         random.shuffle(batches)
#         return iter(batches)

#     def __len__(self):
#         return len(self.X_train_indices)//self.batch_size + self.n_sample


# class MixImgBatchAugmentSampler(Sampler):
#     '''
#     SingleImgBatch:  same image(from img_arg)  in a batch with different data argmentation.
#     MultiImgBatch: similar kind but dirrerent images(from img_arg) in a batch with different data argmentation.
#     RandomImgBatch:  Random sampled from X_train and img_arg in a batch with different data argmentation.
#     MixImgBatch: mix SingleImgBatch and MultiImgBatch.
#     '''

#     def __init__(self, X_train_arg: list, size_X_train: int, batch_size: int, sample_factor: float = 1.0):
#         self.batch_size = batch_size
#         self.split_point = size_X_train
#         self.sample_factor = sample_factor

#         self.X_train_indices = list(range(self.split_point))
#         random.shuffle(self.X_train_indices)

#         self.img_arg_indices = list(range(self.split_point, len(X_train_arg)))
#         self.n_sample_half = int(size_X_train*0.01*0.5)

#     def __iter__(self):
#         def chunk(indices, chunk_size):
#             return torch.split(torch.tensor(indices), chunk_size)

#         # sample n from img_arg_indices, and repeat data be sampled by batch_size
#         self.indices_sample_same = np.repeat(
#             random.sample(self.img_arg_indices, self.n_sample_half), self.batch_size)
#         self.indices_sample_differ = random.sample(
#             self.img_arg_indices, self.n_sample_half*self.batch_size
#         )
#         img_arg_indices_sample = self.indices_sample_same + self.indices_sample_differ

#         # put data as batchs
#         X_train_indices_batches = chunk(self.X_train_indices, self.batch_size)
#         img_arg_indices_batches = chunk(
#             img_arg_indices_sample, self.batch_size)
#         batches = list(X_train_indices_batches + img_arg_indices_batches)
#         batches = [batch.tolist() for batch in batches]
#         # random.shuffle(batches)
#         return iter(batches)

#     def __len__(self):
#         return len(self.X_train_indices)//self.batch_size + self.n_sample_half*2
