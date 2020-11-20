# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:20:30 2020

@author: Shuang Xu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(act='relu', 
                   num_parameters=None, 
                   init=None):
    if act.lower() not in ['relu','prelu','leakyrelu']:
        raise Exception('Only support "relu", or "prelu". But get "%s."'%(act))
    act = act.lower()
    if act=='relu':
        return nn.ReLU(True)
    if act=='leakyrelu':
        init=0.2 if init==None else init
        return nn.LeakyReLU(init, True)
    if act=='prelu':
        num_parameters=1 if num_parameters==None else num_parameters
        init=0.25 if init==None else init
        return nn.PReLU(num_parameters, init)

def get_padder(padding_mode='reflection',
               padding=1,
               value=None):
    if padding_mode.lower() not in ['reflection','replication','zero','zeros','constant']:
        raise Exception('Only support "reflection","replication","zero" or "constant". But get "%s."'%(padding_mode))
    padding_mode = padding_mode.lower()
    if padding_mode=='reflection':
        return nn.ReflectionPad2d(padding)
    if padding_mode=='replication':
        return nn.ReplicationPad2d(padding)
    if padding_mode in ['zero','zeros']:
        return nn.ZeroPad2d(padding)
    if padding_mode in 'constant':
        value=0 if value==None else value
        return nn.ConstantPad2d(padding,value)

class Conv2d(nn.Module):
    def __init__(self, in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1,
                 dilation=1, 
                 groups=1, 
                 bias=False,
                 padding_mode='reflection',
                 padding='same',
                 value=None):
        super(Conv2d, self).__init__()
        padding = int(int(1+dilation*(kernel_size-1))//2) if padding=='same' else 0
        self.pad = get_padder(padding_mode, padding, value)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                0, dilation, groups, bias)
    def forward(self, code):
        return self.conv2d(self.pad(code))

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 act='relu', # activation function (None will disable it)
                 act_value=None, # the initialization of SST or PReLU (Ignore it if act is "relu")
                 norm=True, # batch_normalization (None will disable it)
                 padding_mode='reflection',
                 padding='same',
                 padding_value=None):
        super(Conv2dBlock, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, 
                           dilation, groups, bias, padding_mode, padding, 
                           padding_value)
        self.norm = nn.BatchNorm2d(out_channels) if norm!=None else None
        num_parameters = out_channels if act!='relu' else None
        self.activation = get_activation(act, num_parameters, act_value) if act!=None else None

    def _forward(self, data):
        code = self.conv(data)
        if self.norm!=None:
            code = self.norm(code)
        if self.activation!=None:
            code = self.activation(code)
        return code
    
    def forward(self, data):
        return self._forward(data)

class ResBlock(nn.Module):
    def __init__(self, num_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 act='relu', # activation function (None will disable it)
                 act_value=None, # the initialization of SST or PReLU (Ignore it if act is "relu")
                 norm=True, # batch_normalization (None will disable it)
                 padding_mode='reflection',
                 padding='same',
                 padding_value=None):
        super(ResBlock, self).__init__()
        num_parameters = num_channels if act!='relu' else None
        self.conv1 = Conv2d(num_channels, num_channels, kernel_size, stride, 
                           dilation, groups, bias, padding_mode, padding, 
                           padding_value)
        self.norm1 = nn.BatchNorm2d(num_channels) if norm!=None else None
        self.activation1 = get_activation(act, num_parameters, act_value) if act!=None else None
        self.conv2 = Conv2d(num_channels, num_channels, kernel_size, stride, 
                           dilation, groups, bias, padding_mode, padding, 
                           padding_value)
        self.norm2 = nn.BatchNorm2d(num_channels) if norm!=None else None
        self.activation2 = get_activation(act, num_parameters, act_value) if act!=None else None
        
    def _forward(self, data):
        code = self.conv1(data)
        if self.norm1!=None:
            code = self.norm1(code)
        if self.activation1!=None:
            code = self.activation1(code)
        
        code = self.conv2(code)
        if self.norm2!=None:
            code = self.norm2(code)
        
        code = data+code
        if self.activation2!=None:
            code = self.activation2(code)
        
        return code
    
    def forward(self, data):
        return self._forward(data)
    
