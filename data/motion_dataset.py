"""
Dataset for motion segmentation.
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

Overwrite this class if you want to train on your custom dataset for segmenting from motion.
"""

import os
from pathlib import Path
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as ttf

from data.custom_transforms import *
from utils.train_utils import load_hdf5, str_to_tens


class MotionDataset(Dataset):
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
            paths = sorted([p for p in Path(os.path.join(cfg.ROOT, f)).rglob('*.hdf5')])
            if len(paths) == 10:
                for i in range(9):
                    _paths.append([paths[i], paths[i + 1]])
                _paths.append([paths[-1], paths[0]])

        if split == 'train':
            self._paths = _paths[:int(len(_paths) * 0.9)]
        elif split == 'val':
            self._paths = _paths[int(len(_paths) * 0.9):]

        print(f"Loaded {len(self._paths)} double-paths")

        self.transforms = transforms.Compose([
                transforms.RandomApply([ChannelShuffle()], p=0.3),
                transforms.RandomApply([GaussianBlur(radius=2)], p=0.3),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.5, hue=0.2)]),
            ])

        self.gaussian_noise = transforms.Compose([transforms.RandomApply([AddGaussianNoise(mean=0., std=0.2)], p=0.5)])

    def _process_im(self, im):
        im = ttf.to_pil_image(im)
        im = ttf.resize(im, size=(self.height, self.width), interpolation=Image.LINEAR)

        if self.split == 'train':
            im = self.transforms(im)

        im = ttf.to_tensor(im)
        im = ttf.normalize(im, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.split == 'train':
            im = self.gaussian_noise(im)

        return im

    def _process_segmap(self, segmap):
        segmap[segmap != 0] = 1
        segmap = ttf.to_pil_image(segmap)
        segmap = ttf.resize(segmap, size=(self.height, self.width), interpolation=Image.NEAREST)
        return torch.from_numpy(np.array(segmap)).to(dtype=torch.long)

    def __getitem__(self, idx):
        paths = random.sample(self._paths[idx], 2)
        d1 = load_hdf5(paths[0], keys=['colors', 'segmap'])
        d2 = load_hdf5(paths[1], keys=['colors', 'segmap'])
        im1 = self._process_im(d1['colors'])
        im2 = self._process_im(d2['colors'])
        gt1 = self._process_segmap(d1['segmap'])

        sample = {
            'im1': im1,
            'im2': im2,
            'gt': gt1,
            'path': str_to_tens('foo.bar')
        }

        return sample

    def __len__(self):
        return len(self._paths)
