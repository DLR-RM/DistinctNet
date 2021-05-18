"""
Default Predictor class.
"""

import numpy as np
import cv2

import torch

from networks.distinctnet import DistinctNet
from utils.pred_utils import cfg_and_net_from_path, process_im, process_pred


class Predictor:
    def __init__(self, state_dict_path, device=torch.device('cuda')):

        cfg, net = cfg_and_net_from_path(state_dict_path=state_dict_path)

        self.net = net.eval().to(device)
        self.cfg = cfg
        self.device = device

        self.tim1 = None

    def predict(self, im1, im2):
        """
        Prediction forward pass.
        Expects im to be a numpy.array of dtype uint8.
        """

        if self.tim1 is None:
            self.tim1 = process_im(im1)
        tim2 = process_im(im2)

        with torch.no_grad():
            out = self.net({'im1': self.tim1.to(self.device), 'im2': tim2.to(self.device), 'path': torch.zeros(2, 10)})

        pred = process_pred(out['pred'])

        self.tim1 = tim2

        return pred
