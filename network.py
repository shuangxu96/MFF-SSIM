# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:05:21 2020

@author: win10
"""

from blocks import ResBlock, Conv2dBlock
import torch.nn as nn
import torch
from kornia.color import RgbToGrayscale

class Net(nn.Module):
    def __init__(self, num_blocks, num_feat):
        super(Net, self).__init__()
        self.conv0 = Conv2dBlock(1, num_feat,3)
        self.main = nn.Sequential(
                *[ResBlock(num_feat,3) for i in range(num_blocks)])
        self.conv1 = Conv2dBlock(num_feat*2, 1, 3, act=None, norm=None)
        
    def forward(self, a,b):
        a = RgbToGrayscale()(a)
        b = RgbToGrayscale()(b)
        a = self.main(self.conv0(a))
        b = self.main(self.conv0(b))
        return torch.sigmoid(self.conv1(torch.cat((a,b),dim=1)))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        