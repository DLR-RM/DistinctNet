"""
Correlation layer.
"""

import torch
import torch.nn as nn


class FeatureL2Norm(nn.Module):
    """
    Adapted from https://github.com/ignacio-rocco/cnngeometric_pytorch/blob/master/model/cnn_geometric_model.py
    """
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class CorrelationVolume(nn.Module):
    """
    Adapted from https://github.com/ignacio-rocco/cnngeometric_pytorch/blob/master/model/cnn_geometric_model.py
    """
    def __init__(self, normalize=True):
        super(CorrelationVolume, self).__init__()
        self.normalize = normalize

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()

        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)  # shape (b,c,h*w)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)  # shape (b,h*w,c)
        feature_mul = torch.bmm(feature_B, feature_A)  # shape (b,h*w,h*w)
        correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        return correlation_tensor  # shape (b,h*w,h,w)


class CorrelationLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.corr = CorrelationVolume(normalize=False)
        self.norm = FeatureL2Norm()
        self.act = nn.ReLU()

    def forward(self, x1, x2):
        corr = self.corr(self.norm(x1), self.norm(x2))

        if self.act is not None:
            corr = self.act(corr)
        return self.norm(corr)
