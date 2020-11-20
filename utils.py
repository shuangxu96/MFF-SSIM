# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:21:03 2020

@author: win10
"""
import torch.utils.data as Data
from glob import glob
import os
import torchvision.transforms as transforms
from PIL import Image
import os

class Dataset(Data.Dataset):
    def __init__(self, root, transform = None):
        self.a_files = glob(os.path.join(root,'A', '*.*'))
        self.b_files = glob(os.path.join(root,'B', '*.*'))
        self.gt_files = glob(os.path.join(root,'GT', '*.*'))
        self._tensor = transforms.ToTensor()
        self.transform = transform
        
    def __len__(self):
        return len(self.a_files)
    
    def __getitem__(self, index):
        a = self._tensor(Image.open(self.a_files[index]))
        b = self._tensor(Image.open(self.b_files[index]))
        gt = self._tensor(Image.open(self.gt_files[index]))
        return a, b, gt

def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)