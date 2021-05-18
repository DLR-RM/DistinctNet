"""
Dataset for semantic finetuning.
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
|--gt
|  |--000000.png
|  |--000001.png
|  |--...

Overwrite this class if you want to train on your custom dataset for semantic finetuning.
"""

import os
import numpy as np
import glob
from imgaug import augmenters as iaa

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as ttf

from data.custom_transforms import *
from utils.train_utils import str_to_tens


class SemanticDataset(Dataset):
    def __init__(self, cfg, split='train'):
        assert split in ['train', 'val']
        self.split = split
        self.cfg = cfg

        assert os.path.exists(cfg.ROOT), f"Invalid data root: {cfg.ROOT}"
        folders = os.listdir(cfg.ROOT)
        assert ('rgb1' in folders) and ('rgb2' in folders) and ('gt1' in folders) and ('gt2' in folders), f"Data root ({cfg.ROOT}) does not contain valid folders 'rgb1', 'rgb2', 'gt1' and 'gt2'"

        self.height = cfg.get("HEIGHT", 414)
        self.width = cfg.get("WIDTH", 736)
        self.with_distractor = cfg.get("WITH_DISTRACTOR_OBJECTS", False)

        rgb1_paths = sorted(glob.glob(os.path.join(cfg.ROOT, 'rgb1/*.png')))
        rgb2_paths = sorted(glob.glob(os.path.join(cfg.ROOT, 'rgb2/*.png')))
        gt1_paths = sorted(glob.glob(os.path.join(cfg.ROOT, 'gt1/*.png')))
        gt2_paths = sorted(glob.glob(os.path.join(cfg.ROOT, 'gt2/*.png')))

        assert len(rgb1_paths) == len(rgb2_paths) == len(gt1_paths) == len(gt2_paths), f"Unequal amount of rgb1/rgb2/gt1/gt2 paths: {len(rgb1_paths)}/{len(rgb2_paths)}/{len(gt1_paths)}/{len(gt2_paths)}"

        # take a 90/10 split ratio
        if split == 'train':
            self._rgb1_paths = rgb1_paths[:int(0.9*len(rgb1_paths))]
            self._rgb2_paths = rgb2_paths[:int(0.9*len(rgb2_paths))]
            self._gt1_paths = gt1_paths[:int(0.9*len(gt1_paths))]
            self._gt2_paths = gt2_paths[:int(0.9*len(gt2_paths))]
        else:
            self._rgb1_paths = rgb1_paths[int(0.9 * len(rgb1_paths)):]
            self._rgb2_paths = rgb2_paths[int(0.9 * len(rgb2_paths)):]
            self._gt1_paths = gt1_paths[int(0.9 * len(gt1_paths)):]
            self._gt2_paths = gt2_paths[int(0.9 * len(gt2_paths)):]

        print(f"Loaded {len(self._rgb1_paths)} double-paths")

        self.gaussian_noise = transforms.Compose([transforms.RandomApply([AddGaussianNoise(mean=0., std=0.2)], p=0.9)])
        self.colaug = transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.5, hue=0.2)

    def _process_segmap(self, segmap):
        segmap[segmap != 0] = 1
        segmap = ttf.to_pil_image(segmap)
        segmap = ttf.resize(segmap, size=(self.height, self.width), interpolation=Image.NEAREST)
        return torch.from_numpy(np.array(segmap)).to(dtype=torch.long)

    def __getitem__(self, idx):
        # first load rgb and masks
        rgb1 = np.array(Image.open(self._rgb1_paths[idx]))
        rgb2 = np.array(Image.open(self._rgb2_paths[idx]))
        gt1 = np.array(Image.open(self._gt1_paths[idx]))
        gt2 = np.array(Image.open(self._gt2_paths[idx]))

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
            'path': str_to_tens("foo.bar")
        }

        return sample

    def __len__(self):
        return len(self._rgb1_paths)

    def _augment_manipulator(self, image, mask):
        man_aug = iaa.BlendAlphaSegMapClassIds(class_ids=[1], foreground=iaa.Sequential([
            iaa.BlendAlphaSomeColors(iaa.AddToHueAndSaturation(value_hue=(-255, 255), value_saturation=(-50, 50))),
            iaa.AddToBrightness(add=(-75, 75))]))
        if isinstance(image, list):
            augmented = []
            man_aug = man_aug.to_deterministic()
            for im, ma in zip(image, mask):
                man_mask = (ma == 1).astype(np.uint8)[np.newaxis, ...][..., np.newaxis]
                augmented.append(man_aug(image=im, segmentation_maps=man_mask)[0])
            return augmented
        else:
            man_mask = (mask == 1).astype(np.uint8)[np.newaxis, ...][..., np.newaxis]
            return man_aug(image=image, segmentation_maps=man_mask)[0]

    def _augment_object(self, image, mask, cls_id=2):
        obj_aug = iaa.BlendAlphaSegMapClassIds(class_ids=[1], foreground=iaa.Sequential([
            iaa.BlendAlphaSomeColors(iaa.AddToHueAndSaturation(value_hue=(-255, 255), value_saturation=(-50, 50))),
            iaa.AddToBrightness(add=(-75, 75))]))
        if isinstance(image, list):
            augmented = []
            obj_aug = obj_aug.to_deterministic()
            for im, ma in zip(image, mask):
                obj_mask = (ma == cls_id).astype(np.uint8)[np.newaxis, ...][..., np.newaxis]
                augmented.append(obj_aug(image=im, segmentation_maps=obj_mask)[0])
            return augmented
        else:
            obj_mask = (mask == cls_id).astype(np.uint8)[np.newaxis, ...][..., np.newaxis]
            return obj_aug(image=image, segmentation_maps=obj_mask)[0]

    def _process_im(self, im):
        if self.split == 'train':
            im = self._augment_image(np.array(im))
        im = Image.fromarray(im)
        im = ttf.resize(im, size=(self.height, self.width), interpolation=Image.BILINEAR)
        im = ttf.to_tensor(im)
        im = ttf.normalize(im, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.split == "train":
            im = self.gaussian_noise(im)
        return im

    def _augment_image(self, image):
        mean = iaa.BlendAlpha(
            factor=(0.6, 0.8),
            foreground=iaa.MedianBlur(7),
            per_channel=False
        )

        return mean(image=image)
