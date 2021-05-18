"""
Test sampler for recurrent datasets.
"""

from unittest import TestCase

import torch
from torch.utils.data import Dataset, DataLoader

from data.recurrent_sampler import RecurrentSampler, get_ids_for_sampler


class DummyDataset(Dataset):
    def __init__(self, num_samples=100, recurrent_length=10):
        self.flat = list(range(num_samples))
        self.ids = [self.flat[i:i+recurrent_length] for i in range(0, num_samples, recurrent_length)]

        self.flat = [item for sublist in self.ids for item in sublist]

    def __getitem__(self, item):
        return self.flat[item]

    def __len__(self):
        return len(self.flat)


class TestRecurrentSampler(TestCase):
    def test_recurrent_sampler(self, num_samples=100, recurrent_length=7, batch_size=5, shuffle_ids=True):
        dataset = DummyDataset(num_samples=num_samples, recurrent_length=recurrent_length)
        ids = get_ids_for_sampler(num_samples=len(dataset), recurrent_length=recurrent_length)
        sampler = RecurrentSampler(data_source=(ids, batch_size, shuffle_ids))
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

        samples = torch.stack([sample for _, sample in enumerate(loader)])

        # check drop last
        self.assertTrue(samples.shape[0] % recurrent_length == 0)

        # batch size for insanity
        self.assertTrue(samples.shape[1] % batch_size == 0)

        # check consecutive intervals
        for sample in samples.transpose(1, 0).flatten().chunk(chunks=int(samples.shape[0]*batch_size/recurrent_length)):
            # all elements should be the same when dividing by recurrent_length
            self.assertTrue((sample / recurrent_length).ndim == 1)

        # also print for visual check
        print(samples)
