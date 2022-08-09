from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.quantization import (QuantStub, DeQuantStub)
import torchvision.models as models

import numpy as np
from dataloader import *

IN_PLACE = True
BIAS = True

class Probe(nn.Module):
    def __init__(self, sequential: nn.Sequential):
        super(Probe, self).__init__()
        self.sequential = sequential

    def forward(self, x: torch.Tensor):
        for idx, sub_module in self.sequential._modules.items():
            # pdb.set_trace()
            # print(torch.int_repr(x[0, 0, :4, :4]))
            x = sub_module(x)
            # print(x[0, 0, :4, :4])
            # print(torch.int_repr(x[0, 0, :4, :4]))
            # pdb.set_trace()
        return x


class VGG11(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(VGG11, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1, bias=BIAS),
            nn.ReLU(inplace=IN_PLACE),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=BIAS),
            nn.ReLU(inplace=IN_PLACE),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=BIAS),
            nn.ReLU(inplace=IN_PLACE),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=BIAS),
            nn.ReLU(inplace=IN_PLACE),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=BIAS),
            nn.ReLU(inplace=IN_PLACE),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=BIAS),
            nn.ReLU(inplace=IN_PLACE),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=BIAS),
            nn.ReLU(inplace=IN_PLACE),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=BIAS),
            nn.ReLU(inplace=IN_PLACE),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            # nn.Linear(in_features=512*7*7, out_features=4096),
            # nn.ReLU(),
            # nn.Dropout2d(0.2),
            # nn.Linear(in_features=4096, out_features=4096),
            # nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.Linear(in_features=512*7*7, out_features=self.num_classes),
            nn.Dropout(0.2)
        )

        # self.q = QuantStub()
        # self.deq = DeQuantStub()

    def forward(self, x: torch.Tensor):
        # x = self.q(x)

        # probe = Probe(self.features)
        # x = probe(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)

        # x = self.deq(x)
        return x


if __name__ == "__main__":
    from train import train, test

    exit()
