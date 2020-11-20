# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 21:12:34 2019

@author: win10
"""
import torch
from MFF import MFF
from skimage.io import imread

def to_tensor(a):
    return torch.from_numpy(a).permute(2,0,1).to(torch.float32)/255.

x1 = to_tensor(imread(r'.\test_image\coffee\image1.tif'))
x2 = to_tensor(imread(r'.\test_image\coffee\image2.tif'))
x = torch.stack((x1,x2), dim=0).cuda()
model = MFF(input = x, map_mode = 'gfdf')
model.train(max_iter = 300)
model.show_curve()
model.show_image()
model.save_image('./result/fused_image_cuda.png')
