"""
Dataset for recurrent motion segmentation.
Expects the following folder structure:
root
|--train
|  |--<urandom-path>
|  |  |--0.hdf5
|  |  |--1.hdf5
|  |  |--...
|--val
|  |--<urandom-path>
|  |  |--0.hdf5
|  |  |--1.hdf5
|  |  |--...

Overwrite this class if you want to train on your custom dataset for segmenting from motion with recurrent layers.
"""

import os
from pathlib import Path
import numpy as np

import torchvision.transforms as transforms

from data.motion_dataset import MotionDataset
from data.custom_transforms import *
from utils.train_utils import load_hdf5, str_to_tens


class MotionDatasetRecurrent(MotionDataset):
    def __init__(self, cfg, split='train'):
        assert split in ['train', 'val']
        self.split = split
        self.cfg = cfg

        assert os.path.exists(cfg.ROOT), f"Invalid data root: {cfg.ROOT}"

        self.height = cfg.get("HEIGHT", 414)
        self.width = cfg.get("WIDTH", 736)
        _folders = os.listdir(cfg.ROOT)
        _paths = []
        for f in _folders:
            _paths.append(sorted([p for p in Path(os.path.join(cfg.ROOT, f)).rglob('*.hdf5')]))

        _paths = _paths
        if split == 'train':
            _paths = _paths[:int(len(_paths) * 0.9)]
        elif split == 'val':
            _paths = _paths[int(len(_paths) * 0.9):]

        _first_paths, _second_paths = self._generate_second_paths(_paths)

        self._flat_first_paths = [item for sublist in _first_paths for item in sublist]
        self._flat_second_paths = [item for sublist in _second_paths for item in sublist]

        print(f"Loaded {len(self._flat_first_paths)} recurrent paths")

        self.transforms = transforms.Compose([
                transforms.RandomApply([ChannelShuffle()], p=0.3),
                transforms.RandomApply([GaussianBlur(radius=2)], p=0.3),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.5, hue=0.2)]),
            ])

        self.gaussian_noise = transforms.Compose([transforms.RandomApply([AddGaussianNoise(mean=0., std=0.2)], p=0.5)])

    def _generate_second_paths(self, paths):
        # shifts nested lists and pads with continuous paths to simulate stopping motion
        first_paths, second_paths = [], []
        for path_list in paths:
            shifted_list = path_list[1:]
            cut = np.random.randint(1, 10)
            path_list[cut:] = [path_list[cut] for _ in range(cut, len(path_list))]
            shifted_list[cut:] = [path_list[cut] for _ in range(cut, len(path_list))]
            first_paths.append(path_list)
            second_paths.append(shifted_list)
        return first_paths, second_paths

    def __getitem__(self, idx):
        d1 = load_hdf5(self._flat_first_paths[idx], keys=['colors', 'segmap'])
        d2 = load_hdf5(self._flat_second_paths[idx], keys=['colors', 'segmap'])
        im1 = self._process_im(d1['colors'])
        im2 = self._process_im(d2['colors'])
        gt1 = self._process_segmap(d1['segmap'])

        sample = {
            'im1': im1,
            'im2': im2,
            'gt': gt1,
            'path': str_to_tens(str(self._flat_first_paths[idx]).split('/')[-2])
        }

        return sample

    def __len__(self):
        return len(self._flat_first_paths)
