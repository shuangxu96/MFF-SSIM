# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:47:20 2019

@author: win10
"""

# Require: torch, kornia, guided_filter_pytorch
import torch
import torch.nn.functional as F
from torch import optim

from kornia import get_gaussian_kernel2d
from kornia.filters import laplacian, BoxBlur
from kornia.losses import ssim
from kornia.color import rgb_to_grayscale
from guided_filter_pytorch.guided_filter import GuidedFilter

from typing import Tuple, List
from skimage.io import imshow, imsave, imread
from skimage.morphology import remove_small_objects
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from time import time,strftime,localtime
import os
import numpy as np

def load_single_image(root):
    image = imread(root)
    if len(image.shape)==2:
        image = image[:,:,None]
    return torch.from_numpy(image).permute(2,0,1).to(torch.float32)/255.

def load_focus_map(root):
    map = imread(root)
    if len(map.shape)==3:
        map = rgb2gray(map)
    if map.max()>1.:
        return torch.from_numpy(map).to(torch.float32)/255.
    else:
        return torch.from_numpy(map).to(torch.float32)
    
def load_images(root):
    '''load all the images under ``'root'``. Return: tensor with shape
        (N,C,H,W).
    '''
    image_files = os.listdir(root)
    image = []
    for i in range(len(image_files)):
        image.append(load_single_image(os.path.join(root,image_files[i])))
    image = torch.stack(image, dim=0)
    return image

def compute_padding(kernel_size: Tuple[int, int]) -> List[int]:
    """Computes padding tuple."""
    # 4 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    assert len(kernel_size) == 2, kernel_size
    computed = [(k - 1) // 2 for k in kernel_size]
    return [computed[1], computed[1], computed[0], computed[0]]

def conv2d(input, kernel, pad_mode):
    padding_shape = compute_padding(kernel.shape[-2:])
    input_pad = F.pad(input, padding_shape, mode=pad_mode) 
    return F.conv2d(input_pad, kernel, padding=0, stride=1, groups=1)

# Class

class focus_map:
    '''
    Focus map generator
    '''
    def __init__(self, kernel_mode='avg', pad_mode='replicate'):
        self.kernel_mode = kernel_mode
        self.pad_mode = pad_mode
    
    def var_map(self, input, kernel_size):
        '''
        Input - [N,C,H,W]
        '''
        tmp_kernel = self.get_kernel(kernel_size, input.shape[1], input.device,
                                     input.dtype, self.kernel_mode)
        mu = conv2d(input, tmp_kernel, self.pad_mode)
        mu_sq = mu.pow(2)
        sigma_sq = conv2d(input.pow(2), tmp_kernel, 
                          self.pad_mode) - mu_sq  #[N,1,H,W]
        sigma_sq = sigma_sq.squeeze(1) #[N,H,W]
        return F.one_hot(torch.argmax(sigma_sq, dim=0), 
                         num_classes=input.shape[0]) #[H,W,2]
    
    def lap_map(self, input, kernel_size):
        input_lap_sq = laplacian(input, kernel_size=3).pow(2) #[N,C,H,W]
        tmp_kernel = self.get_kernel(kernel_size, input.shape[1], input.device,
                                     input.dtype, self.kernel_mode)
        lap_map = conv2d(input_lap_sq, tmp_kernel, self.pad_mode)  #[N,1,H,W]
        return F.one_hot(torch.argmax(lap_map.squeeze(1), dim=0), 
                         num_classes=input.shape[0])
    
    def gfdf_map(self, input):
        avg_kernel = BoxBlur(kernel_size=(7,7))
        input_blur = avg_kernel(input) #Eq.(1)&(2)
        RFM = (input-input_blur).abs().sum(1, keepdim=True) #Eq.(3)&(4)
        
        g = GuidedFilter(5, eps=0.3)
        AFM = g(input, RFM.expand(-1,3,-1,-1)).mean(1) #Eq.(5,6)
        IDM = torch.argmin(AFM, dim=0)

        IDM_removed = self.post_remove_small_objects(IDM) #shape:[H,W]
        input_gray = rgb_to_grayscale(input)
        IIF = input_gray[0,:,:,:]*IDM_removed + input_gray[1,:,:,:]*(1-IDM_removed) #shape:[1,H,W]
        FDM = g(IIF[None,:,:,:], IDM_removed[None,None,:,:])
        
        return torch.clamp(FDM.squeeze(0).squeeze(0), min=0, max=1)

    def get_kernel(self, kernel_size, out_channels, device, dtype, mode='box'):
        '''
        Return the kernel (used in SSIM) with shape [1, out_channels, 
        window_size, window_size]. 
        mode - 'box'(default) or 'gauss'
        '''
        if mode in ['box','avg','average']:
            k = torch.ones((kernel_size,kernel_size))
            k = k/k.sum()/out_channels
        elif mode in ['gauss','Gauss','gaussian','Gaussian']:
            k =  get_gaussian_kernel2d(
                    (kernel_size,kernel_size), (1.5,1.5))
        
        k = k.to(device).to(dtype)[None,None,:,:]
        return k.repeat(1,out_channels,1,1)
    
    def post_remove_small_objects(self, input_image, size=None):
        if size is None:
            H, W = input_image.shape
            size = 0.01*H*W
        if type(input_image) is torch.Tensor:
            ar = input_image.detach().cpu().numpy()
        print([ar.min(),ar.max()])
        tmp_image1 = remove_small_objects(ar, size)
        tmp_image2 = 1-tmp_image1
        tmp_image3 = remove_small_objects(tmp_image2, size)
        tmp_image4 = 1-tmp_image3
        tmp_image4 = tmp_image4.astype(np.float)

        if type(input_image) is torch.Tensor:
            tmp_image4 = torch.from_numpy(tmp_image4)
            tmp_image4 = tmp_image4.to(input_image.device)
        return tmp_image4

class MFF:
    r''' Multi-focus image fusion via MFF-SSIM model. 
    
    Arguments:
        input           (tensor): the source images.
        map_mode (str or tensor): the method to estimate focus map. The expected modes are:
            ``'lap'``, ``'var'`` or ``'gfdf'``. Default: ``'lap'``. The user can specify the 
            focus map if you want. When there are 2 source images, map_mode should be spcified 
            as a tensor with shape [H,W] (torch.float). When there are more than 2 source images, it should be
            specified as a tensor with shape [H,W,N] (torch.bool).
        window_size        (int): the size of window of MFF-SSIM. This argument is also used
            in the focus map estimation if you employ ``'lap'`` or ``'var'``. Default: 5e-5*H*W
    
    Shape:
        - input: :math:`(N,C,H,W)`
        
    '''
    def __init__(self, input, map_mode='lap', window_size=None, window_mode='auto', 
                    kernel_mode='box', pad_mode='replicate'):
        
        self.N,self.C,self.H,self.W = input.shape
        
        self.input = input
        self.map_mode = map_mode
        if window_size is None:
            window_size = 5e-5*self.H*self.W
        self.window_size=min(max(int(2*(window_size//2)+1),3),25)
        
        self.kernel_mode = kernel_mode
        self.pad_mode = pad_mode
        self.window_mode = window_mode
        
#        if type(map_mode) is not str and type(map_mode) is not torch.Tensor:
#            Warning('map_mode should be str or torch.Tensor, but get %s' %
#            (type(map_mode)))
        
    def train(self, ini_value=None, learning_rate=0.001, max_iter=1000, 
              opt_mode='adam'):
        '''
        Train MFF-SSIM model.
        
        Arguments: 
            ini_value (tensor): the initial fusion image with shape :math:`(1,C,H,W)`, 
                :math:`(C,H,W)` or :math:`(H,W,C)`. Default: the average image of 
                ``self.input``.
            learning_rate (float): learning rate. Default: 0.001.
            max_iter      (int): the maximum of iterations (aka, the number of epoch).
                Default: 1000
        '''
        # Process initial value
        if ini_value is None:
            fused_image = torch.mean(self.input, dim=0, keepdim=True)
        else:
            if len(ini_value.shape)==4:
                fused_image = ini_value
            elif len(ini_value.shape)==3:
                if ini_value.shape[0]==1 or ini_value.shape[0]==3:
                    fused_image = ini_value.unsqueeze(0)
                elif ini_value.shape[2]==1 or ini_value.shape[2]==3:
                    fused_image = ini_value.permute(2,0,1).unsqueeze(0)
        fused_image = fused_image.to(self.input.device).to(self.input.dtype)
        
        # Estimate focus map
        self.map = self.get_map(self.map_mode).to(self.input.device)
        
        # Create optimizer
        fused_image = fused_image.requires_grad_(True)
        if opt_mode=='adam':
            optimizer = optim.Adam([fused_image], lr=learning_rate)
        elif opt_mode=='rprop':
            optimizer = optim.Rprop([fused_image], lr=learning_rate)
        elif opt_mode=='rmsprop': # suggest
            optimizer = optim.RMSprop([fused_image], lr=learning_rate)
        elif opt_mode=='adamw':
            optimizer = optim.AdamW([fused_image], lr=learning_rate)
            
        
        # Print configuration
        if type(self.map_mode) is not str:
            map_mode = 'User Specified'
        else:
            map_mode = self.map_mode
        print(22*'='+'MFF-SSIM Fusion'+22*'=')
        if self.window_mode == 'auto':
            print('*Configuration* -> [Focus map: %s], [Window Size: %d]' 
                  % (map_mode, self.window_size )  )
        else:
            print('*Configuration* -> [Focus map: %s], [Window Size: %s]' 
                  % (map_mode, self.window_mode )  )
        print('*Configuration* -> [Iterations: %d], [Learning Rate: %s]' 
              % (max_iter, learning_rate )  )
        print('*Configuration* -> [Deivce: %s], [Dtype: %s]' 
              % (self.input.device, self.input.dtype )  )
        print('*Configuration* -> [#Images x Channel x Height x Width: %d x %d x %d x %d]'    
                % ( self.input.shape ) )

        # Update
        ssim_value=[]
        print('*Fusion Starts* -> '+strftime('%Y-%m-%d %H:%M:%S'
                                             ,localtime(time())))
        t0 = time()
        if self.input.is_cuda is False:
            print('%43s' % ('====================================='))
            print('%20s  %20s' % ('No. Iteration','MFF-SSIM Value'))
            print('%43s' % ('-------------------------------------'))
        for i in range(1,max_iter+1):
            optimizer.zero_grad()
            ssim_out = self.mff_ssim(fused_image)
            tmp_value = 1-2*ssim_out.item()
            ssim_value.append( tmp_value )
            ssim_out.backward()
            optimizer.step()
            with torch.no_grad():
                fused_image.data = torch.clamp(fused_image, min=0, max=1)
            if self.input.is_cuda is False and i%10==1 :
                print('%13d  %24.6f' % (i,tmp_value) )
        t1 = time()
        if self.input.is_cuda is False:
            print('%43s' % ('====================================='))
        print('*Fusion   Ends* -> '+strftime('%Y-%m-%d %H:%M:%S',
                                             localtime(time())))

        # Save results
        self.mff_ssim_values = ssim_value
        self.fused_image = fused_image
        self.elapsed_time = t1-t0
        print(60*'-')
        print('(a) Elapsed time %.6f seconds.'% (self.elapsed_time))
        print('(b) Initial MFF-SSIM value is %.6f' %(ssim_value[0]) )
        print('(c) Final   MFF-SSIM value is %.6f' %(ssim_value[-1]) )
        print(60*'=')
        print('Thank you for using MFF-SSIM Fusion!')
        
    def mff_ssim(self, fused_image):
        '''
        Compute MFF-SSIM value.
        
        Arguments:
            fused_image (tensor): the fusion image with shape :math:`[1,C,H,W]`
            
        Returns:
            torch.float: the MFF-SSIM value.
        '''
        ssim_map = []
        if self.window_mode == 'multiscale':
            self.window_size = [3,5,7,9,11,13,15]
            
        for i in range(self.input.shape[0]):
            if self.window_mode == 'auto':
                ssim_map.append(ssim(self.input[i,:,:,:][None,:,:,:], fused_image,
                                     self.window_size))
            elif self.window_mode == 'multiscale':
                tmp = 0
                for w in range(len(self.window_size)):
                    tmp = tmp+ssim(self.input[i,:,:,:][None,:,:,:], fused_image,
                                     self.window_size[w])
                ssim_map.append(tmp/len(self.window_size))
            
        ssim_map = torch.cat(ssim_map, dim=0) #[N,H,W]
        q = self.mapped_ssim(ssim_map, self.map)
        return q.mean()
    
    def mapped_ssim(self, a, map):
        '''
        Subfunction of ``self.mff_ssim``.
        '''
        if map.dtype is torch.long:
            map = map.to(torch.bool)
            
        a.squeeze_(1) #[N,1,H,W]
        map = map.permute(2,0,1)[:,None,:,:].repeat(1,self.C,1,1) #[N,C,H,W]
        output = torch.sum(map*a, dim=0)
        return output

    def get_map(self, map_mode):
        '''
        Estimate the focus map.  [H,W,N]
        '''
        if type(map_mode) is str:
            map_generator=focus_map(kernel_mode=self.kernel_mode, 
                                    pad_mode=self.pad_mode)
            if map_mode in ['var', 'variance']:
                return map_generator.var_map(self.input, self.window_size)
            elif map_mode in ['lap', 'laplacian', 'LAP']:
                return map_generator.lap_map(self.input, self.window_size)
            elif map_mode in ['gfdf', 'GFDF']:
                map = map_generator.gfdf_map(self.input)
                return torch.stack((map,1-map),dim=-1)
            
        else:
            if type(map_mode) is not torch.Tensor:
                map_mode = torch.from_numpy(map_mode)
            if map_mode.dtype is torch.long:
                if len(map_mode.shape)==2: #[H,W]
                    map_mode = map_mode.to(torch.long)
                    return F.one_hot(map_mode, num_classes=self.N)
                elif len(map_mode.shape)==3: #[H,W,N]
                    return map_mode.to(torch.long)
            else:
                if len(map_mode.shape)==2:
                    map_mode = torch.stack((map_mode,1-map_mode),dim=-1)
                return map_mode.to(torch.float32)

    
    # utils
    
    def show_image(self):
        '''
        Display the fused image.
        '''
        image = self.fused_image.detach().squeeze(0).permute(1,2,0)
        imshow(image.cpu().numpy())
    
    def save_image(self, root):
        '''
        Save the fused image to ``'root'``. ``'root'`` should be like ``'xxx/xxx/xx.png'``
        '''
        root0 = root.split('/')[:-1]
        if root0==[]:
            root0 = root.split('\\')[:-1]
        root0 = os.path.join(*root0)
        if os.path.exists(root0) is False:
            os.makedirs(root0)
        
        image = self.fused_image.detach().squeeze(0).permute(1,2,0)
        imsave(root, image.cpu().numpy())
    
    def save_map(self, root):
        '''
        Save the focus map to ``'root'``. ``'root'`` should be like ``'xxx/xxx/xx.png'``
        '''
        root0 = root.split('/')[:-1]
        if root0==[]:
            root0 = root.split('\\')[:-1]
        root0 = os.path.join(*root0)
        if os.path.exists(root0) is False:
            os.makedirs(root0)
            
        if len(self.map.shape)==2:
            imsave(root, self.map.cpu().numpy())
            
        if len(self.map.shape)==3:
            if self.map.shape[-1]>=3:
                tmp = root.split('/')[-1].split('.')
                file_name0 = tmp[0]
                file_name1 = tmp[1]
                for i in range(self.map.shape[-1]):
                    file_name = file_name0+str(i+1)+'.'+file_name1
                    save_root = os.path.join(root0, file_name)
                    tmp_map   = self.map[:,:,i].to(torch.float).cpu().numpy()
                    imsave(save_root, tmp_map)
            else:
                imsave(root, self.map[:,:,0].to(torch.float).cpu().numpy())
                
    def show_curve(self, root=None):
        '''
        Display the MFF-SSIM value versus iterations.
        '''
        plt.plot(self.mff_ssim_values)









        