# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 23:35:54 2020

@author: win10
"""


import torch
from skimage.io import imsave
from os import makedirs
from os.path import join,exists
from MFF2 import MFF, load_single_image, load_focus_map

device = 'cuda' if torch.cuda.is_available() else 'cpu'


for i in range(1,10):
    for j in range(100):
        # read the given focus map and source images
        M = load_focus_map(r'Map_Analysis\label.png').to(device)
        A = load_single_image(r'Map_Analysis\image1.tif')
        B = load_single_image(r'Map_Analysis\image2.tif')
        
        prob=torch.rand_like(M)
        M2 = torch.abs(M-1*(prob<i/10))
        input = torch.stack((A,B), dim=0).to(device)
        
        # MFF-SSIM solver
        solver = MFF(input, map_mode=M2)
        solver.train()
        
        # Save the fusion image
        path = join(r'Map_Analysis\output\\', str(i)+'_%d.png'%(j))
        path = path.replace('\\','/')
        path0 = path.split('/')[:-1]
        if path0==[]:
            path0 = path.split('\\')[:-1]
        path0 = join(*path0)
        if exists(path0) is False:
            makedirs(path0)
        
        image = solver.fused_image.detach().squeeze(0).permute(1,2,0)
        imsave(path, image.cpu().numpy())
        imsave(join(r'Map_Analysis\output', str(i)+'_%d_Map.png'%(j)), M2.cpu().numpy())