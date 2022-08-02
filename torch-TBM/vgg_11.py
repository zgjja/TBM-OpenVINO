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
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=IN_PLACE),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=IN_PLACE),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=IN_PLACE),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=IN_PLACE),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=IN_PLACE),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=IN_PLACE),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=IN_PLACE),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
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
        )

        # self.q = QuantStub()
        # self.deq = DeQuantStub()

    def forward(self, x: torch.Tensor):
        # x = self.q(x)

        # probe = Probe(self.features)
        # x = probe(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # x = torch.flatten(x, 1)
        x = self.classifier(x)

        # x = self.deq(x)
        return x


if __name__ == "__main__":
    from train import train, test

    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.001
    CLASSES = 3
    DEVICE = torch.device('cuda:0') # 'cpu'

    # with ToTensor()
    mean = [0.4548, 0.4811, 0.4541]
    std = [0.2276, 0.2212, 0.2236]
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean, std, inplace=True)
    ])

    stone_dataset = Stone(num_classes=CLASSES, transform=transform)
    stone_dataloader = DataLoader(stone_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    # testdataset = Stone(num_classes=CLASSES, prefix="../stone_enhanced_zjq_3/train", transform=transform)
    # stone_dataloader = DataLoader(testdataset, batch_size=32, num_workers=4, shuffle=False)

    model = VGG11(3, CLASSES)
    # torch.quantization.fuse_modules(model, [['conv', 'relu']])
    

    # model_dict = model.state_dict()
    # vgg11 = models.vgg11(pretrained=True)  # https://download.pytorch.org/models/vgg11-8a719046.pth
    # pretrained_dict = vgg11.state_dict()
    # for k, v in model_dict.items():
    #     if k in pretrained_dict and "classi" not in k:
    #         model_dict[k] = pretrained_dict[k]
    # model.load_state_dict(model_dict)

    # model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    # model_prepare = torch.quantization.prepare_qat(model)
    
    # model_int8 = torch.quantization.convert(model_prepare)
    # print(model_int8.state_dict().keys())
    # pdb.set_trace()

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.ConstantLR(optimizer, 0.5, 2500)
    loss_fn = nn.CrossEntropyLoss()
    
    path = None
    # train(model, transform, stone_dataloader, optimizer, loss_fn, lr_scheduler)
    train(model, None, stone_dataloader, optimizer, loss_fn, None, path=path, use_fp16=False)
    

    # print(a['state_dict'].keys())
    # print(torch.int_repr(a['state_dict']['features.0.weight'])) # int8
    # print(torch.int_repr(a['state_dict']['features.0.bias'])) # fp32
    # print(a['state_dict']['features.0.bias'])

    # print(quantized.features[0].scale.dtype)
    # print(type(quantized.features[0]))


    """two ways to print int8 quantized value"""
    # print(quantized.state_dict()['features.0.weight'][0, ...])
    # print(quantized.features[0].weight().int_repr().data[0, ...])
    # print(quantized.features[0].scale)
    # print(quantized.features[0].zero_point)
        # b = VGG11(3,3)
        # b.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        # b_prepare = torch.quantization.prepare_qat(b)
        # quantized = torch.quantization.convert(b_prepare)
        # quantized.load_state_dict(a["state_dict"])
        # im = torch.tensor(np.zeros((1, 3, 224, 224), dtype=np.float32))
        # im[0, 0, 0, 0] = 10
        # im[0, 0, 0, 1] = 20
        # im[0, 0, 0, 2] = 30
        # d = quantized(im)
        # pdb.set_trace()
        # test(quantized, transform, 'cpu')
        
        # torch.save({'state_dict': quantized.state_dict()}, f"./shit_q.pth.tar")
        # print(quantized.features[0].weight)
        # print(d['features.0.weight'].dtype)
        # print(d['classifier.0.weight'])
        # print(a['state_dict'].keys())
        # 
        # pdb.set_trace()
