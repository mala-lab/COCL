import os
import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import datetime



class ImageNet_ood(torch.utils.data.Dataset):

    def __init__(self, root, transform, txt):

        super(ImageNet_ood, self).__init__()

        self.data = []
        self.transform = transform
        with open(txt) as f:
            for item in f:
                self.data.append(os.path.join(root, item[:-1]))               

        print("ImageNet Contain {} images".format(len(self.data)))

    def __getitem__(self, index):
        
        img = self.data[index]
        with open(img, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)


        return img, 1000  # -1 is the class

    def __len__(self):
        return len(self.data)