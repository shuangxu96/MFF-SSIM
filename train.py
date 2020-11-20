# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:11:42 2020

@author: win10
"""

import torch
import os

from network import Net
from utils import Dataset,mkdir
from torch.utils.data import DataLoader 
import datetime

batch_size = 6
lr = 1e-4
num_epoch = 50
num_blocks = 8
num_feat = 128


net = Net(num_blocks, num_feat).cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# loaders
trainset    = Dataset('dataset')
loader      = DataLoader(trainset,      batch_size=batch_size, shuffle=True) 

save_path = datetime.datetime.now().strftime("%m-%d-%H-%M")
mkdir(save_path)

best_loss = 1e+100
torch.backends.cudnn.benchmark = True
for epoch in range(num_epoch):
    ''' train '''
    for i, (A,B,GT) in enumerate(loader):
        A,B,GT = A.cuda(),B.cuda(),GT.cuda()
        temp_int = int(torch.randint(4,[1]))
        A = torch.rot90(A, temp_int, [-1,-2])
        B = torch.rot90(B, temp_int, [-1,-2])
        GT = torch.rot90(GT, temp_int, [-1,-2])
        
        #1. update
        net.train()
        net.zero_grad()
        optimizer.zero_grad()
        output = net(A,B)
        loss = torch.nn.BCELoss()(output,GT)
        loss.backward()
        optimizer.step()
        
        #2.  print
        print("[%d,%d]  Loss: %.4f" %
                (epoch+1, i+1, loss.item()))

    ''' save model ''' 
    if loss.item()<best_loss:
        best_loss = loss.item()
        torch.save(net.state_dict(), os.path.join(save_path, 'best_net.pth'))
    torch.save({'net':net.state_dict(),
                'optimizer':optimizer.state_dict(),
                'epoch':epoch},
                os.path.join(save_path, 'last_net.pth'))