import sys, os, pdb
from typing import Optional

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import cv2 as cv
import numpy as np


class Stone(Dataset):
    def __init__(self, num_classes: int,
            prefix: str = "../../dataset/TBM/train",
            transform: Optional[transforms.Compose] = None
    ):
        self.num_classes = num_classes
        self.prefix = prefix
        self.transform = transform
        self.imgs, self.labels = [], []
        
        for i in range(self.num_classes):
            prefix = os.path.join(self.prefix, str(i))
            tmp = os.listdir(prefix)
            self.imgs.extend(os.path.join(prefix, img) for img in tmp)
            self.labels.extend(i for _ in range(len(tmp)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int):
        image = cv.imread(self.imgs[idx], cv.IMREAD_COLOR)
        
        if self.transform:
            image = self.transform(image)
        else:  # if without ToTensor() in transforms
            mean = [0.4548, 0.4811, 0.4541]
            std =  [0.2276, 0.2212, 0.2236]
            image = image.transpose(2, 0, 1).astype(np.float32)
            image /= 255
            for i in range(3):
                image[i] -= mean[i]
                image[i] /= std[i]
            image = torch.from_numpy(image)

        label = np.array([0 for _ in range(self.num_classes)]).astype(np.float32)
        label[self.labels[idx]] = 1
        label.reshape((1, self.num_classes))
        
        # make sure the image is (N, C, H, W) format
        return (image, torch.from_numpy(label))


if __name__ == '__main__':
    mean = [0.4548, 0.4811, 0.4541]
    std =  [0.2276, 0.2212, 0.2236]
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean, std, inplace=True), 
    ])

    stone_dataset = Stone(num_classes=3, prefix="../../dataset/TBM/test", transform=None)
    stone_dataloader = DataLoader(stone_dataset, batch_size=1, shuffle=True, num_workers=1)
    
    for i, (img, label) in enumerate(stone_dataloader, 1):
        img = img.numpy()
        print(img, label)
        pdb.set_trace()
        cv.imshow("dataloader func test", img)
        cv.waitKey(500)
