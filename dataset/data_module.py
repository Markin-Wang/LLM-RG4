from lightning.pytorch import LightningDataModule
from torch.utils.data import ConcatDataset, WeightedRandomSampler, DataLoader
from dataset.data_helper_sn import create_datasets_sn
from dataset.data_helper_sw import create_datasets_sw
from dataset.data_helper_mn import create_datasets_mn
from dataset.data_helper_mw import create_datasets_mw
from config.config import parser
import torch
import numpy as np
from torch.utils.data.sampler import RandomSampler
import math
import random

class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset.meta) for cur_dataset in dataset.datasets])

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)


class DataModule(LightningDataModule):

    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings.

        download

        tokenize

        etc…
        :return:
        """

    def setup(self, stage: str):
        if self.args.test_mode =='train_2':
            train_sn, dev_sn, test_sn = create_datasets_sn(self.args)
            train_sw, dev_sw, test_sw = create_datasets_sw(self.args)
            train_mn, dev_mn, test_mn = create_datasets_mn(self.args)
            train_mw, dev_mw, test_mw = create_datasets_mw(self.args)
            train_dataset = ConcatDataset([train_sn, train_sw, train_mn, train_mw])
            dev_dataset = ConcatDataset([dev_sn, dev_sw, dev_mn, dev_mw])
            test_dataset = ConcatDataset([test_sn, test_sw, test_mn, test_mw])
            self.dataset = {
                "train": train_dataset, "validation": dev_dataset, "test": test_dataset
            }
        if self.args.test_mode =='sn' or self.args.test_mode =='train_1':
            train_sn, dev_sn, test_sn = create_datasets_sn(self.args)
            self.dataset = {
                "train": train_sn, "validation": dev_sn, "test": test_sn
            }
        if self.args.test_mode =='sw':
            train_sw, dev_sw, test_sw = create_datasets_sw(self.args)
            self.dataset = {
                "train": train_sw, "validation": dev_sw, "test": test_sw
            }
        if self.args.test_mode =='mn':
            train_mn, dev_mn, test_mn = create_datasets_mn(self.args)
            self.dataset = {
                "train": train_mn, "validation": dev_mn, "test": test_mn
            }
        if self.args.test_mode =='mw':
            train_mw, dev_mw, test_mw = create_datasets_mw(self.args)
            self.dataset = {
                "train": train_mw, "validation": dev_mw, "test": test_mw
            }

    def train_dataloader(self):
        """
        Use this method to generate the train dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
        if self.args.test_mode == 'train_2':
            loader = DataLoader(self.dataset["train"],
                                sampler=BatchSchedulerSampler(dataset=self.dataset["train"],
                                                              batch_size=self.args.batch_size),
                                batch_size=self.args.batch_size, drop_last=True, pin_memory=True, shuffle=False,
                            num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
            return loader
        else:
            loader = DataLoader(self.dataset["train"], batch_size=self.args.batch_size, drop_last=True, pin_memory=False,shuffle=True,
                            num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
            return loader

    def val_dataloader(self):
        """
        Use this method to generate the val dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
        if self.args.test_mode == 'train_2':
            loader = DataLoader(self.dataset["validation"],
                                sampler=BatchSchedulerSampler(dataset=self.dataset["validation"],
                                                              batch_size=self.args.val_batch_size),
                                batch_size=self.args.val_batch_size, drop_last=False, pin_memory=True,shuffle=False,
                                num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
            return loader
        else:
            loader = DataLoader(self.dataset["validation"], batch_size=self.args.batch_size, drop_last=True, pin_memory=False,
                                shuffle=False,
                                num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
            return loader


    def test_dataloader(self):
        loader = DataLoader(self.dataset["test"], batch_size=self.args.test_batch_size, drop_last=False, pin_memory=False,
                        num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader



