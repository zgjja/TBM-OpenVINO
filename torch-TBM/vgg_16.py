import pickle, sys, os, pdb
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import Stone
from train import *

EPOCH = 400
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
CHECK_POINT = 20
CLASSES = 3
IN_PLACE = False


class VGG16(nn.Module):
    def __init__(self, num_classes:int = CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(in_channels=num_classes, out_channels=64, padding=1, kernel_size=3, stride=1), 
            nn.ReLU(inplace=IN_PLACE), 
            nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1), 
            nn.ReLU(inplace=IN_PLACE), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=IN_PLACE), 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=IN_PLACE), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1), 
            nn.ReLU(inplace=IN_PLACE), 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1), 
            nn.ReLU(inplace=IN_PLACE), 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1), 
            nn.ReLU(inplace=IN_PLACE), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # block 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1), 
            nn.ReLU(inplace=IN_PLACE), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1), 
            nn.ReLU(inplace=IN_PLACE), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1), 
            nn.ReLU(inplace=IN_PLACE), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # block 5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1), 
            nn.ReLU(inplace=IN_PLACE), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1), 
            nn.ReLU(inplace=IN_PLACE), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1), 
            nn.ReLU(inplace=IN_PLACE), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=num_classes)
        )
        

    def forward(self, x):
        feature = self.features(x)
        feature = feature.view(x.size(0), -1)
        y = self.classifier(feature)
        return y

if __name__ == "__main__":
    pass
