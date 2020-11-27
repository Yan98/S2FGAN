#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is the concrete implementation of S2FGAN.
This module structure is following:
    make_kernel is used to intialise the kernel for blurring image
    Blur, a layer used to apply blur kerbel to input
    PixelNorm, a layer used to apply pixel normalization
    EqualConv1d, convolution 1d with equalized learning trick
    EqualConv2d, convolution 2d with equalized learning trick
    Equallinear, linear layerwith equalized learning trick
    MLP, a multi layer perceptro consists of Equalinear and leakly relu
    AMT, attribute mapping networks.
    ModulatedConv2d, the modulated convolution 2d.
    Encoder, the encoder of S2FGAN.
    UpBlock, the upblock for the decoder of S2FGAN.
    Im2Col, the function used to unfold feature maps into chunks. e.g  unfold 8x8 features maps into 4 2x2 chunks. 
    Discriminatorblock, the discriminator block for the discriminator of S2FGAN.
    Discriminator, the discrimantor of S2FGAN.
    VGGPL, the perceptual loss based on VGG16.
"""

import torch
import math
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
        
#create blur kernel
def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k

#Blur the image   
class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, x):
        x = upfirdn2d(x, self.kernel, pad=self.pad)
        return x
    
#Pixel Normalization         
class PixelNorm(nn.Module):
    def __init__(self,esp = 1e-8):
        super().__init__()
        self.esp = esp

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) +  self.esp)      

#Equlized convlution 1d
class EqualConv1d(nn.Module):
    def __init__(self,  in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        """
        Return, None
        Parameters
        ----------
        in_channels, int, the channels of input
        out_channels, int, the channles expanded by the convolution
        kernel_size, int, the size of kernel needed.
        stride: int, controls the cross correlation during convolution
        padding: int, the number of gride used to pad input.
        bias: bool, controls adding of learnable biase
        Returns
        -------
        None
        """     
        #intialze weight 
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size))
        #calculate the scales for weight
        self.scale = 2  / math.sqrt(in_channel *  out_channel * kernel_size)
        
        self.stride = stride
        self.padding = padding
        
        #create bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None
            
    def forward(self,x):
        """
        Return, the convolutioned x.
        Parameters
        ----------
        x: pytorch tensor, used for the input of convolution
        Returns
        -------
        the convolutioned x
        """  
        x = F.conv1d(x, self.weight * self.scale,bias=self.bias, stride=self.stride, padding=self.padding)
        return x

#Equlized convlution 2d
class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        """
        Return, None
        Parameters
        ----------
        in_channels, int, the channels of input
        out_channels, int, the channles expanded by the convolution
        kernel_size, int, the size of kernel needed.
        stride: int, controls the cross correlation during convolution
        padding: int, the number of gride used to pad input.
        bias: bool, controls adding of learnable biase
        Returns
        -------
        None
        """  
        #intialze weight
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        #calculate the scales for weight
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding
        
        #create bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, x):
        """
        Return, the convolutioned x.
        Parameters
        ----------
        x: pytorch tensor, used for the input of convolution
        Returns
        -------
        the convolutioned x
        """          
        x = F.conv2d(x, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
        return x

#Equlized Linear 
class EqualLinear(nn.Module):
    def __init__( self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation= False):
        super().__init__()
        """
        Return, None
        Parameters
        ----------
        in_dim, int, number of features for input
        out_dim, int, number of features for output
        bias: bool, controls adding of learnable biase
        lr_mul: int, the scales of biase
        activation: bool, controls the use of leakly relu.
        Returns
        -------
        None
        """   
        #intialze weight
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        
        #stores if use activation
        self.activation = activation
        
        #calculate the scales for weight
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        #create bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
            
    def forward(self, x):
        """
        Return, the transformed x.
        Parameters
        ----------
        x: pytorch tensor, used for the input of linear.
        Returns
        -------
        the transformed x.
        """           
        
        if self.activation:
            #apply activation after transformation
            x = F.linear(x, self.weight * self.scale)
            x = fused_leaky_relu(x, None if self.bias is None else self.bias * self.lr_mul)
        else:
            #use linear only
            x = F.linear(x, self.weight * self.scale, bias= (None if self.bias is None else self.bias * self.lr_mul))
        return x
    
#Multi layer perceptron   
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dim=256, n_layers=3, lr_mul = 1):
        """
        Return, None
        Parameters
        ----------
        in_dim, int, number of features for input
        out_dim, int, number of features for output
        dim: int, number of features for middle layers
        n_layers: int, number of linear layer.
        lr_mul: int, the scales of biase
        Returns
        -------
        None
        """          
        super().__init__()
        layers = [EqualLinear(in_dim, dim,lr_mul = lr_mul,activation= True)]
        for _ in range(n_layers - 2):
            layers += [EqualLinear(dim, dim,lr_mul = lr_mul,activation= True)]
        layers += [EqualLinear(dim, out_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Return, the transformed x.
        Parameters
        ----------
        x: pytorch tensor, used for the input of linear.
        Returns
        -------
        the transformed x.
        """  
        return self.model(x.view(x.size(0), -1))
    
    
#Attribute Mapping Networks
class AMN(nn.Module):
    def __init__(self, in_channel):
        """
        Return, None
        Parameters
        ----------
        in_channels, int, number of channels for input
        Returns
        -------
        None
        """    
        super().__init__()
        self.w = EqualConv1d(in_channel, 64, 3,padding = 1, bias=False)
        self.h = EqualConv1d(in_channel, 64, 3,padding = 1, bias=False)
        self.n = EqualConv1d(64, 1, 3, padding = 1, bias=False)
        
    def forward(self,x):
        """
        Return, the transformed x.
        Parameters
        ----------
        x: pytorch tensor.
        Returns
        -------
        the transformed x.
        """  
        #apply convolution and calcualte cross simiarily
        f = self.w(x)
        f = f / torch.norm(f,p=2,dim = 1,keepdim= True)
        
        return (self.n(f.bmm(f.permute(0,2,1)).bmm(self.h(x)))).sum(dim = 1)  
    
#Modulated Convolution   
class ModulatedConv2d(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,style_dim,demodulate=True,upsample=False, blur_kernel=[1, 3, 3, 1], eps = 1e-8):
        """
        Return, None
        Parameters
        ----------
        in_channels, int, the channels of input
        out_channels, int, the channles expanded by the convolution
        kernel_size, int, the size of kernel needed.
        style_dim, int, dimensionality of attribute latent space.
        demodulate, int, decide applying demodulation
        upsample, bool, decide if upsample the input
        blur_kernel, [int], the kernel used to blur input.
        eps, int, used for numerical stablity
        Returns
        -------
        None
        """          
        super().__init__()
        self.eps = eps
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        
        #calculate padding for upsample, and intialize blur layer for upsample and blur.
        if upsample:
            p = (len(blur_kernel) - 2) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + 2 - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=2)
        #intialise the scale for convolution weights, and calculate padding for convolution
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.padding = kernel_size // 2
        #intialise convoltion weight
        self.weight     = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        #map the attribute editing latent space to scale convolution weight
        self.modulation = EqualLinear(style_dim,in_channel, bias_init=1)
        self.demodulate = demodulate


    def forward(self, x, style):
        """
        Return, the transformed x.
        Parameters
        ----------
        x: pytorch tensor. for appearance latent space.
        style: pytorch tensor. for attribute editing latent space.
        Returns
        -------
        the transformed x.
        """ 
        batch, in_channel, height, width = x.shape
        
        #scales weight of convoltion
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style
        #apply demodulation
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        #reshape weight
        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)

        if self.upsample:
            #apply upsample, blur and conv tranpose 2d.
            x = x.view(1, batch * in_channel, height, width)
            weight = weight.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size)
            
            x = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = x.shape
            x = x.view(batch, self.out_channel, height, width)
            x = self.blur(x)

        else:
            # apply convolution 2d. 
            x   = x.view(1, batch * in_channel, height, width)
            x = F.conv2d(x, weight, padding=self.padding, groups=batch)
            _, _, height, width = x.shape
            x = x.view(batch, self.out_channel, height, width)

        return x


            
class Encoder(nn.Module):
    def __init__(self, in_channels=1, dim=64, n_downsample = 5,  max_dim = 512):
        super().__init__()
        """
        Return, None
        Parameters
        ----------
        in_channels, int, the channels of input
        dim, int, the output channels for vision block.
        n_downsamples, int, the times of downsampling  for appearance latent space.
        max_dim: int, the maximum number of channels for convolution output.
        Returns
        -------
        None
        """     
        #vision block
        layers = [
            nn.Sequential(
            EqualConv2d(in_channels, dim, 7,padding = 3),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2,inplace=True),
            )
        ]

        # down block
        dim_cur  = dim
        dim_next = dim * 2
        for _ in range(n_downsample):
            layers += [
                nn.Sequential(
                    EqualConv2d(dim_cur, dim_next, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(dim_next),
                    FusedLeakyReLU(dim_next),
                    )
            ]
            dim_cur  = dim_next
            dim_next = min(max_dim,dim_next * 2) 
        dim = min(max_dim,dim_cur)
        self.model  = nn.ModuleList(layers)
        
        #Appearance feature transform
        toLatent = [nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(), PixelNorm(), MLP(dim_cur,dim,dim,4, lr_mul = 0.01)]
        self.toLatent  = nn.Sequential(*toLatent)

    def forward(self, x):
        
        """
        Return, the transformed x.
        Parameters
        ----------
        x: pytorch tensor. the input for generator of S2FGAN.
        Returns
        -------
        todecode, pytorch tensor, the latent code of appearance latent space, used to create unit-skip connections
        x, pytorch tensor, the latent code of attribute editing space.
        """    
        todecode = []
        for index,m in enumerate(self.model):
            x = m(x)  
            if index >= 3:
                todecode += [x]
        x = self.toLatent(x)  
        return  todecode,x
#upBlock      
class UpBlock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size = 3,style_dim = 512,upsample=True,blur_kernel=[1, 3, 3, 1], demodulate=True):
        super().__init__()
        """
        Return, None
        Parameters
        ----------
        in_channels, int, the channels of input
        out_channels, int, the channles expanded by the convolution
        kernel_size, int, the size of kernel needed.
        style_dim, int, dimensionality of attribute latent space.
        upsample, bool, decide if upsample the input
        blur_kernel, [int], the kernel used to blur input.
        demoulated, bool, decide applying demodulation
        Returns
        -------
        None
        """  
        self.conv  =  ModulatedConv2d(in_channel,out_channel,kernel_size,style_dim,upsample=upsample,blur_kernel=blur_kernel,demodulate=demodulate)
        self.activate   = FusedLeakyReLU(out_channel)
    def forward(self, x, style):
        """
        Return, the transformed x.
        Parameters
        ----------
        x: pytorch tensor. latent code of appearance latent space.
        style: pytorch tensor, latent code of attribute editing latent space.
        Returns
        -------
        x, pytorch tensor, the transformed x.
        """        
        x  = self.conv(x,style)
        x  = self.activate(x)
        return x
        
       
class Decoder(nn.Module):
    def __init__(self, out_channels=3, dim=64, c_dim = 12, step = 1):
        super().__init__()
        """
        Return, None
        Parameters
        ----------
        out_channels, int, the channels of output.
        c_dim, int, the number of attributes
        max_dim: int, the maximum number of channels for convolution input..
        Returns
        -------
        None
        """           
        #semantic for attributes
        self.directions = nn.Parameter(torch.zeros(1, c_dim, 512))
        
        self.map        = AMN(c_dim + 1)
        
        self.step = step

        self.shortcut1 = nn.ModuleList([
            UpBlock(512,512),                    #4-8
            UpBlock(512,512,upsample = False)
            ])
        
        self.shortcut2 = nn.ModuleList([
            UpBlock(1024,512),                    #8-16
            UpBlock(512,512,upsample = False)
            ])
        
        self.res = nn.ModuleList([
            UpBlock(1024,512),                   #16-32
            UpBlock(512,512,upsample = False),
            UpBlock(512,256),                    #32-64
            UpBlock(256,256,upsample = False),
            UpBlock(256,128),                    #64-128 
            UpBlock(128,128,upsample = False),
            UpBlock(128,64),                     #128-256
            UpBlock(64,64,  upsample = False),
            UpBlock(64,32),                      #256-512
            UpBlock(32,32,  upsample = False)            
            ])
        
        # Output layer
        self.rgb = nn.ModuleList([
            EqualConv2d(128, out_channels, 7, padding = 3), #for 128 x 128 resolution
            EqualConv2d(64, out_channels, 7, padding = 3),  #for 256 x 256 resolution
            EqualConv2d(32, out_channels, 7, padding = 3),  #for 512 x 512 resolution 
            ])
    def forward(self, content_code,latent, c):
        
        """
        Return, the transformed content code.
        Parameters
        ----------
        content_code, pytorch tensor. latent code of appearance latent space.
        style, pytorch tensor, latent code of attribute editing latent space.
        c, pytorch tensor, the attributes shifting vector.
        Returns
        -------
        img, pytorch tensor, the transformed content code.
        """  
        #get semantic emebedding
        d      = self.directions.repeat(c.size(0),1,1)
        latent = self.map(torch.cat((latent.view(-1,1,512),d * c.view(-1,c.size(1),1)),1)) 
        
        #skip connections
        x = self.shortcut1[0](content_code[2],latent)
        x = self.shortcut1[1](x,latent)
        
        x = torch.cat((x,content_code[1]),1)
        x = self.shortcut2[0](x,latent)
        x = self.shortcut2[1](x,latent)
        
        x = torch.cat((x,content_code[0]),1)
        
        if self.step == 1:
            res = self.res[:-4]
        elif self.step == 2:
            res = self.res[:-2]
        elif self.step == 3:
            res = self.res
        for b in res:
            x = b(x,latent)
        img = self.rgb[self.step - 1](x)
        return img

#discriminator block, consists of equalized conv2d and learly relu layer.                 
class DiscriminatorBlock(nn.Module):
    def __init__(self,in_filters, out_filters):
        super().__init__()
        self.model = nn.Sequential(
            EqualConv2d(in_filters, out_filters, 4, stride=2, padding=1), 
            FusedLeakyReLU(out_filters)
            )
    def forward(self,x):
        return self.model(x)
 
#used to unfold the feature maps into chunks. e.g  unfold 8x8 features maps into 4 2x2 chunks.    
def im2col(x, kernel_size, stride):
    return Im2Col.apply(x, kernel_size, stride) 
def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = im2col(x, block_size, block_size)
    return unfolded_x.view(n * block_size ** 2, c, h // block_size, w // block_size) 
class Im2Col(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride):
        ctx.shape, ctx.kernel_size, ctx.stride = (x.shape[2:], kernel_size, stride)
        return F.unfold(x, kernel_size=kernel_size, stride=stride)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.enable_grad():
            shape, ks, stride = ctx.shape, ctx.kernel_size, ctx.stride
            return (
                F.fold(grad_output, shape, kernel_size=ks, stride=stride),
                None,
                None,
            )
        
        
class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), c_dim=12, n_strided = 5, step = 1):
        super().__init__()
        """
        Return, None
        Parameters
        ----------
        img_shape, (int,int,int), the shape of input
        c_dim, int, the number of attributes
        downsamples, int, number of downsamples before last layers.
        Returns
        -------
        None
        """  
        channels, img_size, _ = img_shape
        
       	layers = [DiscriminatorBlock(channels, 64)]
        curr_dim = 64
        self.step = step
        for _ in range(n_strided - 1):
            layers += [DiscriminatorBlock(min(curr_dim,1024), min(curr_dim * 2,1024))]
            curr_dim *=2
        
       	curr_dim = min(curr_dim,1024)
        self.model = nn.ModuleList(layers)

        # Output 1: PatchGAN
        self.out1 = nn.Sequential(
            nn.Flatten(),
            EqualLinear(curr_dim * 4 ** 2,1024,activation= True),
            EqualLinear(1024, 1,bias = False)
            )
        # Output 2: Class prediction
        self.out2 = nn.Sequential(
            nn.Flatten(),
            EqualLinear(curr_dim * 4 ** 2,1024,activation= True),
            EqualLinear(1024, c_dim,bias = False)
            )

    def forward(self, x):
        for m in self.model:
            x = m(x)
        if self.step == 1:
            out_adv = self.out1(x)
            out_cls = self.out2(x)
        elif self.step == 2:
            out_adv = self.out1(space_to_depth(x,2))
            out_cls = self.out2(F.adaptive_avg_pool2d(x, (4,4)))
        elif self.step == 3:
            out_adv = self.out1(space_to_depth(x,4))
            out_cls = self.out2(F.adaptive_avg_pool2d(x, (4,4)))
            
        return out_adv.view(out_cls.size(0), -1), out_cls.view(out_cls.size(0), -1)  

#the implementation of vgg perceptual loss
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        model = torchvision.models.vgg16(pretrained=True)
        blocks.append(model.features[:4].eval())
        blocks.append(model.features[4:9].eval())
        blocks.append(model.features[9:16].eval())
        blocks.append(model.features[16:23].eval())
        blocks = nn.ModuleList(blocks)
        for b in blocks:
            for l in b:
                l.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.weights = [1.0/8, 1.0/4, 1.0/2, 1.0] 
        
    def forward(self, input, target):
        loss = 0.0
        x = input
        y = target
        for i,block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y) * self.weights[i]
        return loss 
