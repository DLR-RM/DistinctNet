"""
DeepLabv3+ decoder.
Adapted from https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/decoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabDecoder(nn.Module):
    def __init__(self, num_classes, norm=nn.BatchNorm2d, low_level_reduction=48):
        super(DeepLabDecoder, self).__init__()
        low_level_inplanes = 256

        self.conv1 = nn.Conv2d(low_level_inplanes, low_level_reduction, 1, bias=False)
        self.bn1 = norm(low_level_reduction)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(256+low_level_reduction, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat, *args, **kwargs):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
