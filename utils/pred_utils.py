"""
Prediction utilities.
"""

import numpy as np
import cv2
import os
import yaml
from yacs.config import CfgNode
from PIL import Image

import torch
import torchvision.transforms.functional as ttf

from networks.distinctnet import DistinctNet


def cfg_and_net_from_path(state_dict_path):
    assert os.path.isfile(state_dict_path) and state_dict_path.endswith('.pth'), f"Invalid state dict file: {state_dict_path}"

    cfg_path = '/'.join(state_dict_path.split('/')[:-2]) + '/config.yaml'
    with open(cfg_path, 'r') as f:
        cfg = CfgNode(yaml.load(f, Loader=yaml.FullLoader))

    net = DistinctNet(cfg.MODEL)
    rets = net.load_state_dict(torch.load(state_dict_path), strict=False)

    print(f"Loaded state dict from {state_dict_path}:")
    print(f"{rets}")

    return cfg, net


def process_im(im):
    """
    Converts given numpy.array (dtype uint8) to a normalized tensor of shape (1, 3, 414, 736)
    """

    assert im.ndim == 3, f"Wrong dims for image: {im.ndim}; expected 3 dims"
    assert im.shape[-1] == 3, f"Expects a three channel RGB image"
    im = ttf.to_pil_image(im)

    im = ttf.resize(im, (414, 736), interpolation=Image.LINEAR)
    im = ttf.to_tensor(im)
    im = ttf.normalize(im, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return im.unsqueeze(0)


def process_pred(pred):
    """
    Processes the raw network prediction.
    """

    if pred.shape[1] == 2:
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        return pred.squeeze().cpu().numpy()
    else:
        pred = torch.sigmoid(pred)
        return pred.squeeze().cpu().numpy()


def overlay_im_with_mask(im, ma, alpha=0.3):
    """
    Overlays an image with corresponding annotations.
    """

    im_col = im.copy()
    if ma.ndim == 2:
        a, b = np.where(ma == 1)
        if a != []:
            im_col[a, b, :] = np.array([0, 255, 0])
    else:
        a, b = np.where(ma[2] >= 0.5)
        if a != []:
            im_col[a, b, :] = np.array([0, 165, 255])
        a, b = np.where(ma[3] >= 0.5)
        if a != []:
            im_col[a, b, :] = np.array([0, 255, 0])
        if ma.shape[0] == 5:
            a, b = np.where(ma[4] >= 0.5)
            if a != []:
                im_col[a, b, :] = np.array([0, 0, 255])
    im_overlay = im.copy()
    im_overlay = cv2.addWeighted(im_overlay, alpha, im_col, 1 - alpha, 0.0)

    return im_overlay
