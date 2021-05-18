"""
Dataset for recurrent semantic finetuning.
Expects the following folder structure:
root
|--rgb1
|  |--000000.png
|  |--000001.png
|  |--...
|--rgb2
|  |--000000.png
|  |--000001.png
|  |--...
|--rgb3
|...
|--gt1
|  |--000000.png
|  |--000001.png
|  |--...
|--gt2
|  |--000000.png
|  |--000001.png
|  |--...
|--gt3
|...
Overwrite this class if you want to train on your custom dataset for recurrent semantic finetuning.
"""

import os
import numpy as np
import glob
from imgaug import augmenters as iaa
from pathlib import Path

import torchvision.transforms as transforms
import torchvision.transforms.functional as ttf

from data.custom_transforms import *
from data.semantic_dataset import SemanticDataset
from utils.train_utils import str_to_tens


class SemanticDatasetRecurrent(SemanticDataset):
    def __init__(self, cfg, split='train'):
        assert split in ['train', 'val']
        self.split = split
        self.cfg = cfg

        assert os.path.exists(cfg.ROOT), f"Invalid data root: {cfg.ROOT}"
        folders = os.listdir(cfg.ROOT)
        rgb_folders = sorted([f for f in folders if f.startswith('rgb')])
        gt_folders = sorted([f for f in folders if f.startswith('gt')])
        assert len(rgb_folders) == len(gt_folders), f"Unequal length of rgb and gt sequences: {len(rgb_folders)} != {len(gt_folders)}"
        self.recurrent_length = len(rgb_folders)

        self.height = cfg.get("HEIGHT", 414)
        self.width = cfg.get("WIDTH", 736)
        self.with_distractor = cfg.get("WITH_DISTRACTOR_OBJECTS", False)

        _rgb_paths, _gt_paths = [], []
        for rgb_folder in rgb_folders:
            _rgb_paths.append(sorted([p for p in Path(os.path.join(cfg.ROOT, rgb_folder)).rglob('*.png')]))
        for gt_folder in gt_folders:
            _gt_paths.append(sorted([p for p in Path(os.path.join(cfg.ROOT, gt_folder)).rglob('*.png')]))

        _rgb_paths = list(map(list, zip(*_rgb_paths)))
        _gt_paths = list(map(list, zip(*_gt_paths)))

        if split == 'train':
            _rgb_paths = _rgb_paths[:int(len(_rgb_paths) * 0.9)]
            _gt_paths = _gt_paths[:int(len(_gt_paths) * 0.9)]
        elif split == 'val':
            _rgb_paths = _rgb_paths[int(len(_rgb_paths) * 0.9):]
            _gt_paths = _gt_paths[int(len(_gt_paths) * 0.9):]
        
        _first_rgb_paths, _second_rgb_paths, _first_gt_paths, _second_gt_paths = self._generate_second_paths(_rgb_paths, _gt_paths)

        self._flat_first_rgb_paths = [item for sublist in _first_rgb_paths for item in sublist]
        self._flat_second_rgb_paths = [item for sublist in _second_rgb_paths for item in sublist]
        self._flat_first_gt_paths = [item for sublist in _first_gt_paths for item in sublist]
        self._flat_second_gt_paths = [item for sublist in _second_gt_paths for item in sublist]

        print(f"Loaded {len(self._flat_first_rgb_paths)} recurrent paths")

        self.gaussian_noise = transforms.Compose([transforms.RandomApply([AddGaussianNoise(mean=0., std=0.2)], p=0.9)])
        self.colaug = transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.5, hue=0.2)

    def _generate_second_paths(self, rgb_paths, gt_paths):

        # shifts nested lists and pads with continuous paths to simulate stopping motion
        first_rgb_paths, second_rgb_paths = [], []
        first_gt_paths, second_gt_paths = [], []
        for rgb_path_list, gt_path_list in zip(rgb_paths, gt_paths):
            shifted_rgb_list = rgb_path_list[1:]
            shifted_gt_list = gt_path_list[1:]
            cut = np.random.randint(1, 5)
            rgb_path_list[cut:] = [rgb_path_list[cut] for _ in range(cut, len(rgb_path_list))]
            gt_path_list[cut:] = [gt_path_list[cut] for _ in range(cut, len(gt_path_list))]
            shifted_rgb_list[cut:] = [rgb_path_list[cut] for _ in range(cut, len(rgb_path_list))]
            shifted_gt_list[cut:] = [gt_path_list[cut] for _ in range(cut, len(gt_path_list))]
            first_rgb_paths.append(rgb_path_list)
            second_rgb_paths.append(shifted_rgb_list)
            first_gt_paths.append(gt_path_list)
            second_gt_paths.append(shifted_gt_list)
        return first_rgb_paths, second_rgb_paths, first_gt_paths, second_gt_paths

    def __getitem__(self, idx):
        # first load rgb and masks
        rgb1 = np.array(Image.open(self._flat_first_rgb_paths[idx]))
        rgb2 = np.array(Image.open(self._flat_second_rgb_paths[idx]))
        gt1 = np.array(Image.open(self._flat_first_gt_paths[idx]))
        gt2 = np.array(Image.open(self._flat_second_gt_paths[idx]))

        if self.with_distractor:  # treat distractor objects as separate class
            gt1[gt1 > 2] = 3
            gt2[gt2 > 2] = 3
            rgb1, rgb2 = self._augment_object(image=[rgb1, rgb2], mask=[gt1, gt2], cls_id=3)
        else:
            gt1[gt1 > 2] = 0
            gt2[gt2 > 2] = 0

        # augment manipulator and object
        if self.split == 'train':
            rgb1, rgb2 = self._augment_manipulator(image=[rgb1, rgb2], mask=[gt1, gt2])
            rgb1, rgb2 = self._augment_object(image=[rgb1, rgb2], mask=[gt1, gt2])

        rgb1 = self._process_im(rgb1)
        rgb2 = self._process_im(rgb2)

        # classes:
        # 0: background
        # 1: foreground
        # 2: manipulator
        # 3: object
        # 4: distractor objects (optional)
        gt = np.zeros((5 if self.with_distractor else 4, 414, 736), dtype=np.uint8)
        gt[0, :, :][gt1 == 0] = 1
        gt[1, :, :][gt1 != 0] = 1
        gt[2, :, :][gt1 == 1] = 1
        gt[3, :, :][gt1 == 2] = 1
        if self.with_distractor:
            gt[4, :, :][gt1 == 3] = 1

        sample = {
            'im1': rgb1,
            'im2': rgb2,
            'gt': torch.from_numpy(gt).to(dtype=torch.float32),
            'path': str_to_tens(str(self._flat_first_rgb_paths[idx]).split('/')[-1])
        }

        return sample

    def __len__(self):
        return len(self._flat_first_rgb_paths)
