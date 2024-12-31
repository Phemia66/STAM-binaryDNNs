import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import scipy.io as io
import time
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from collections import OrderedDict


class CNN(nn.Module):
    def __init__(self, input_size):
        super(CNN, self).__init__()
        img_dim = input_size[0]
        self.layers = nn.Sequential(
            nn.Conv2d(img_dim, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.final_fc = nn.Linear(input_size[1] * input_size[2] // 16 * 20, 10)

    def forward(self, x):
        x = self.layers(x)
        return self.final_fc(x.view(x.size(0), -1))






cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 100)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        in_channels = 3
        x = cfg[0]
        layers = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, x, kernel_size=3, padding=1)),
            ('norm0', nn.BatchNorm2d(x)),
            #('relu0', nn.ReLU(inplace=True))
            ('relu0', nn.Tanh())
        ]))
        in_channels = x

        index_pool = 0
        index_block = 1
        for x in cfg[1:]:
            if x == 'M':
                layers.add_module('pool%d' % index_pool,
                                  nn.MaxPool2d(kernel_size=2, stride=2))
                index_pool += 1
            else:
                layers.add_module('conv%d' % index_block,
                                  nn.Conv2d(in_channels, x, kernel_size=3, padding=1)),
                layers.add_module('norm%d' % index_block,
                                  nn.BatchNorm2d(x)),
                layers.add_module('relu%d' % index_block,
                                  nn.ReLU(inplace=True))
                in_channels = x
                index_block += 1
                #layers.add_module('avg_pool%d' % index_pool,
                #                   nn.AvgPool2d(kernel_size=1, stride=1))
        return layers

