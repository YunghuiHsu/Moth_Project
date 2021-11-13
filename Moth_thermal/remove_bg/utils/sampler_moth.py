import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class AddBatchAugmentSampler(Sampler):
    def __init__(self, X_train_arg: list, size_X_train: int, batch_size: int):
        self.batch_size = batch_size
        self.split_point = size_X_train

        self.X_train_indices = list(range(self.split_point))
        random.shuffle(self.X_train_indices)

        self.img_arg_indices = list(range(self.split_point, len(X_train_arg)))
        self.n_sample = int(size_X_train*0.01)

    def __iter__(self):
        def chunk(indices, chunk_size):
            return torch.split(torch.tensor(indices), chunk_size)

        # sample n from img_arg_indices, and repeat data be sampled by batch_size
        img_arg_indices_sample = random.sample(
            self.img_arg_indices, self.n_sample)*self.batch_size
        img_arg_indices_sample.sort()

        # put data as batchs
        X_train_indices_batches = chunk(self.X_train_indices, self.batch_size)
        img_arg_indices_batches = chunk(
            img_arg_indices_sample, self.batch_size)
        batches = list(X_train_indices_batches + img_arg_indices_batches)
        batches = [batch.tolist() for batch in batches]
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return len(self.X_train_indices)//self.batch_size + self.n_sample


class NonBatchAugmentSampler(Sampler):
    def __init__(self, X_train_arg: list, size_X_train: int, batch_size: int):
        self.batch_size = batch_size
        self.split_point = size_X_train

        self.X_train_indices = list(range(self.split_point))
        random.shuffle(self.X_train_indices)

        self.img_arg_indices = list(range(self.split_point, len(X_train_arg)))
        self.n_sample = int(size_X_train*0.01)
        assert self.n_sample*self.batch_size <= len(
            self.img_arg_indices), f'n_sample*batch_size({self.n_sample}*{self.batch_size}) is large than img_arg dataset({len(self.img_arg_indices)})'

    def __iter__(self):
        def chunk(indices, chunk_size):
            return torch.split(torch.tensor(indices), chunk_size)

        # sample n from img_arg_indices, and repeat data be sampled by batch_size
        self.img_arg_indices_sample = random.sample(
            self.img_arg_indices, self.n_sample*self.batch_size)
        # self.img_arg_indices_sample.sort()

        # put data as batchs
        X_train_indices_batches = chunk(self.X_train_indices, self.batch_size)
        img_arg_indices_sample_batches = chunk(
            self.img_arg_indices_sample, self.batch_size)
        batches = list(X_train_indices_batches +
                       img_arg_indices_sample_batches)
        batches = [batch.tolist() for batch in batches]
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return len(self.X_train_indices)//self.batch_size + self.n_sample
