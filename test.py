# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 14:12:12 2020

@author: win10
"""
from torchvision import transforms
from skimage.io import imread
from MFF2 import MFF
import os
from network import Net
import torch
from utils import mkdir

def to_np(x):
    x = x.squeeze()
    x = x.permute(1,2,0)
    return x.detach().cpu().numpy()

    
# Configure Net
net = Net(num_blocks=8, num_feat=128).cuda()
net.load_state_dict(torch.load('weight/block8_feat128.pth'))

# Test set and output folder
test_image1 = 'test_set/Lytro'
test_image2 = 'test_set/MFFW2'
save_path = 'save_path'
mkdir(os.path.join(save_path,'Lytro'))
mkdir(os.path.join(save_path,'MFFW'))

to_tensor = transforms.ToTensor()

# Lytro
for j in range(1,21):
    if j<=9:
        index = '0'+str(j)
    else:
        index = str(j)
    
    a = to_tensor(imread(os.path.join(test_image1,'lytro-'+index+'-A.jpg')))[None,:,:,:].cuda()
    b = to_tensor(imread(os.path.join(test_image1,'lytro-'+index+'-B.jpg')))[None,:,:,:].cuda()
    with torch.no_grad():
        net.eval()
        output = net(a,b)
    theta = output.detach().squeeze()
    theta[theta>0.5]=1
    theta[theta<=0.5]=0
    
    _input = torch.cat((a,b), 0)
    post_process = MFF(_input,map_mode=theta)
    post_process.train(a*theta+b*(1-theta))
    
    post_process.save_image(os.path.join(save_path, 'Lytro', str(j)+'_fusion.png'))
    post_process.save_map(os.path.join(save_path, 'Lytro', str(j)+'_map.png'))

# MFFW
for j in range(1,14):
    a = to_tensor(imread(os.path.join(test_image2,str(j),'image1.tif')))[None,:,:,:].cuda()
    b = to_tensor(imread(os.path.join(test_image2,str(j),'image2.tif')))[None,:,:,:].cuda()
    with torch.no_grad():
        net.eval()
        output = net(a,b)
    theta = output.detach().squeeze()
    theta[theta>0.5]=1
    theta[theta<=0.5]=0
    
    _input = torch.cat((a,b), 0)
    post_process = MFF(_input,map_mode=theta)
    post_process.train(a*theta+b*(1-theta))
    
    post_process.save_image(os.path.join(save_path, 'MFFW', str(j)+'_fusion.png'))
    post_process.save_map(os.path.join(save_path, 'MFFW', str(j)+'_map.png'))
