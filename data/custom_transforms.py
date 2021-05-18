"""
Various custom transforms.
"""

import random
from PIL import Image, ImageFilter

import torch


class ChannelShuffle:
    """
    Randomly shuffles RGB channels.
    """
    def __call__(self, img):
        rgb = img.split()

        rgb = list(rgb)
        random.shuffle(rgb)
        return Image.merge(img.mode, rgb)


class GaussianBlur:
    """
    Applies Gaussian blur on an image.
    """
    def __init__(self, radius=1):
        self._rad = radius

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self._rad))

    def __repr__(self):
        return self.__class__.__name__ + '(rad={0})'.format(self._rad)


class AddGaussianNoise:
    """
    Adds Gaussian noise on an image.
    """
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * torch.rand(1) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
