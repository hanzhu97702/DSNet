import torch
import torch.nn as nn
from torch.nn import init

import numpy as np
import math

class Conv_Classifier(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def __init__(self, input_channels, num_classes, patch_size=7, n_planes=64):
        super(Conv_Classifier, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        self.conv1 = nn.Conv2d(input_channels, n_planes, (3, 3), stride=(1, 1))
        self.conv2 = nn.Conv2d(n_planes, 100, (3, 3), stride=(1, 1))
        self.relu = nn.ReLU()

        self.feature_size = self._get_final_flattened_size()
        self.fc1 = nn.Linear(self.feature_size, 100)
        self.fc2 = nn.Linear(100, num_classes)
        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size))
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            _, c, w, h = x.size()
            return c * w * h

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, self.feature_size)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DSNet(nn.Module):
    """
    dual-branch subpixel-guided network for hyperspectral image classification
    """
    def __init__(self, band, num_classes, patch_size, basic_cls_name):
        super(DSNet, self).__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.basic_cls_name = basic_cls_name
        # unmixing module
        self.unmix_encoder = nn.Sequential(
            nn.Conv2d(band, band//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(band//2),
            nn.ReLU(),
            nn.Conv2d(band//2, band//4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(band//4),
            nn.ReLU(),
            nn.Conv2d(band//4, num_classes, kernel_size=1, stride=1, padding=0)
        )
        self.unmix_decoder = nn.Sequential(
            nn.Conv2d(num_classes, band*2, kernel_size=1, stride=1, bias=False),
            nn.ReLU()
        )
        self.unmix_decoder_nonlinear = nn.Sequential(
            nn.Conv2d(band*2, band, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(band, band, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )

        # basic classification backbone module
        if 'conv2d' in basic_cls_name:
            self.cls = Conv_Classifier(band, num_classes, patch_size, 64)
        else:
            raise KeyError("{} model is unknown.".format(basic_cls_name))
        # fusion module
        self.conv = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
        )
        self.feature_size = self._get_final_flattened_size()
        self.fc = nn.Linear(self.feature_size, num_classes)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.num_classes, self.patch_size, self.patch_size))
            x = self.conv(x)
            _, c, w, h = x.size()
            return c * w * h + self.num_classes

    def forward(self, x):
        abu = self.unmix_encoder(x)
        re_unmix = self.unmix_decoder(abu)
        re_unmix_nonlinear = self.unmix_decoder_nonlinear(re_unmix)
        feature_cls = self.cls(x) # cls token
        # abu sum-to-one and nonnegative constraint
        abu = abu.abs()
        abu = abu / abu.sum(1).unsqueeze(1)
        # reshape abu
        feature_abu = self.conv(abu)
        abu_v = feature_abu.reshape(x.shape[0], -1)
        # fuse abu features and cls token
        feature_fuse = torch.cat([abu_v, feature_cls], dim=1)
        output_cls = self.fc(feature_fuse)
        return re_unmix_nonlinear, re_unmix, output_cls
