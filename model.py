"""
This module is the concrete implementation of S2FGAN.
This module structure is following:
    make_kernel is used to intialise the kernel for blurring image
    Blur, a layer used to apply blur kerbel to input
    PixelNorm, a layer used to apply pixel normalization
    EqualConv1d, convolution 1d with equalized learning trick
    EqualConv2d, convolution 2d with equalized learning trick
    Equallinear, linear layerwith equalized learning trick
    Embedding, attribute mapping networks.
    Encoder, the encoder of S2FGAN.
    StyledConv, the upblock for the decoder of S2FGAN.
    Discriminator, the discrimantor of S2FGAN.
    VGGPerceptualLoss, the perceptual loss based on VGG19.
"""

import math
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.nn.init import normal_
from torch import autograd, optim
from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix


#Pixel Normalization 
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

#create blur kernel
def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

    
#Blur Layer
class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out
    
class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out

#Equlized convlution 2d
class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
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
       
        
        super().__init__()

        #intialize weight
        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        
        #calculate sacles for weight
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        #create bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        """
        Return, the convolutioned x.
        Parameters
        ----------
        x: pytorch tensor, used for the input of convolution
        Returns
        -------
        the convolutioned x
        """          
       
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
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
        
        super().__init__()

        #initialize weight
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        #create bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        #store activation function
        self.activation = activation

        #calculate sacles for weight 
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
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
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1]
    ):
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
        downsample, bool, decide if downsample the input
        blur_kernel, [int], the kernel used to blur input.
        Returns
        -------
        None
        """          
        
        
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def forward(self, input, style):
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
        
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out

#trainable input layer for decoder
class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
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
        
        super().__init__()

        self.conv1 = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=True,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.activate1 = FusedLeakyReLU(out_channel)
        
        self.conv2 = ModulatedConv2d(
            out_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.activate2 = FusedLeakyReLU(out_channel)

    def forward(self, input, style):
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
        out = self.conv1(input, style)
        out = self.activate1(out)
        out = self.conv2(out,style)
        out = self.activate2(out)
        return out


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
        
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size))
        self.scale = 2  / math.sqrt(in_channel *  out_channel * kernel_size)
        
        self.stride = stride
        self.padding = padding
        
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
    
class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out

#Block for Attribute Mapping Network              
class Modify(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.model  = nn.Sequential(
            EqualConv1d(in_channel, 64, 3,padding = 1, bias=False),
            nn.LeakyReLU(0.2, inplace = True),  
            EqualConv1d(64, 64, 3,padding = 1, bias=False),
            nn.LeakyReLU(0.2, inplace = True), 
            )
        
       	self.w = EqualConv1d(64, 64, 3,padding = 1, bias=False)
        self.h = EqualConv1d(64, 64, 3,padding = 1, bias=False)
        self.n = EqualConv1d(64, 64, 3, padding = 1, bias=False)
        
       	self.skip = EqualConv1d(in_channel, 64, 1, bias=False)
        
    def forward(self,input):
        x = self.model(input)
        f = self.w(x)
        f = f / (torch.norm(f,p=2,dim = 1,keepdim= True) + 1e-8)
        x = self.n(f.bmm(f.permute(0,2,1)).bmm(self.h(x)))
        return x + self.skip(input)

#Attribute Mapping Network    
class Embeding(nn.Module):
    def __init__(self, c_dim):
        super().__init__()
        self.directions = nn.Parameter(torch.zeros(1, c_dim, 512))
        self.b1         = Modify(c_dim + 1)
        self.b2         = Modify(64)
        self.b3         = Modify(64)
        self.b4         = Modify(64)
        self.b5         = EqualConv1d(64, 1, 1, bias=False)
        
    def forward(self,x,a, reg = False):
        d = self.directions.repeat(a.size(0),1,1)
        is_reconstruct = ((a.sum(1, keepdim = True) != 0.0).float()).view(a.size(0),1,1)
        d = torch.cat((d * a.view(-1,a.size(1),1),x.view(x.size(0),1,512) * is_reconstruct),1)
        d = self.b1(d)
        d = self.b2(d)
        d = self.b3(d)
        d = self.b4(d)
        d = self.b5(d).view(-1,512)
        if reg:
            return d
        else:
            return x + d
        
#encoder
class Encoder(nn.Module):
    def __init__(self, in_channels=1, dim=64, n_downsample = 5,  max_dim = 512, noise = False):
        super().__init__()

        pool_size = {
            32  : 4,
            64  : 3,
            128 : 2,
            256 : 2,
            512 : 1,
            }        

        self.vision = ConvLayer(in_channels,dim,1)
        
        conv_layers   = []
        linear_layers = []
        # Downsampling
        dim_cur  = dim
        dim_next = dim * 2
        for _ in range(n_downsample):
            conv_layers   += [
                nn.Sequential(
                    ResBlock(dim_cur,dim_next),
                    ResBlock(dim_next,dim_next,downsample= False)
                    )
            ]
            
            linear_layers += [nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size[dim_next]),
                nn.Flatten(),
                EqualLinear(dim_next * pool_size[dim_next] ** 2, 512, lr_mul = 0.01, activation="fused_lrelu"),
                *[EqualLinear(512, 512, lr_mul = 0.01, activation="fused_lrelu") for _  in range(3)]
                )
                ]
            
            dim_cur  = dim_next
            dim_next = min(max_dim,dim_next * 2) 
        
        self.model  = nn.ModuleList(conv_layers)
        self.linear = nn.ModuleList(linear_layers)
        self.norm   = PixelNorm()
        extra_dimension = 100 if noise else 0
        self.final  = nn.Sequential(
            EqualLinear(512 + extra_dimension, 512, lr_mul = 0.01, activation="fused_lrelu"),
            *[EqualLinear(512, 512, lr_mul = 0.01, activation="fused_lrelu") for _  in range(4)]
            )

    def forward(self, x, noise = None):
        
        x     = self.vision(x)
        style = 0
        
        for index in range(len(self.model)):
            x      = self.model[index](x)
            style += self.linear[index](x)
        style = style / (index + 1)
        style = self.norm(style)
        if noise != None:
            noise = self.norm(noise)
            style = torch.cat((style,noise),1)
        style = self.final(style)
        return style


#decoder
class Generator(nn.Module):
    def __init__(
        self,
        c_dim,
        style_dim = 512,
        n_mlp = 8,
        channel_multiplier= 1,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = ModulatedConv2d(
                        512,
                        512,
                        3,
                        style_dim,
                        upsample= False,
                        blur_kernel=blur_kernel,
                        demodulate=True,
        )
        self.activate1 = FusedLeakyReLU(512)
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.convs = nn.ModuleList([
            StyledConv(512,512,3,style_dim,blur_kernel),                                             #4   -  8
            StyledConv(512,512,3,style_dim,blur_kernel),                                             #8   -  16
            StyledConv(512,512,3,style_dim,blur_kernel),                                             #16  -  32
            StyledConv(512,256 * channel_multiplier,3,style_dim,blur_kernel),                        #32  -  64
            StyledConv(256 * channel_multiplier, 128 * channel_multiplier,3,style_dim,blur_kernel),  #64  -  128
            StyledConv(128 * channel_multiplier, 64 * channel_multiplier,3,style_dim,blur_kernel),   #128 -  256
            ])

        self.to_rgbs = nn.ModuleList([
            ToRGB(512, style_dim),                       #8
            ToRGB(512, style_dim),                       #16
            ToRGB(512, style_dim),                       #32
            ToRGB(256 * channel_multiplier, style_dim),  #64
            ToRGB(128 * channel_multiplier, style_dim),  #128
            ToRGB(64 * channel_multiplier, style_dim),   #256
            ])
        
    def forward(self,style):
        x    = self.input(style)
        x    = self.conv1(x,style)
        x    = self.activate1(x)
        skip = self.to_rgb1(x,style)
            
        for index in range(len(self.convs)):
            x    = self.convs[index](x,style)
            skip = self.to_rgbs[index](x,style,skip)
        return skip
        

#convolution layer with dowmsample and activation function
class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)

#residual block
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], downsample = True):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out
    
#domain discriminator     
class GradReverse(Function):
	@staticmethod
	def forward(ctx, x, beta = 1.0):
		ctx.beta = beta
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		grad_input = grad_output.neg() * ctx.beta
		return grad_input, None
    
class Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        normal_(self.weight, 0, 0.001)
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(0))
        self.scale = (1 / math.sqrt(in_dim))

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale, bias=self.bias)
        return out
    
class Domain_Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.feature = Linear(512, 512)
    self.relu    = nn.ReLU(inplace = True)
    self.fc      = Linear(512, 1)

  def forward(self,x):
    x = GradReverse.apply(x)
    x = self.feature(x)
    x = self.relu(x)
    x = self.fc(x)
    return x

class Classifier(nn.Module):
    def __init__(self,c_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(512, c_dim))
        self.c_dim = c_dim
        nn.init.xavier_uniform_(self.W.data, gain=1)

    
    def forward(self,x, ortho = False):
        self.W_norm = self.W / self.W.norm(dim=0)
        if not ortho:
            return torch.matmul(x,self.W_norm)
        else:
            return torch.matmul(x,self.W_norm), nn.L1Loss()(self.W_norm.transpose(1,0).matmul(self.W_norm), torch.diag(torch.ones(self.c_dim,device = x.device)))
        
    def edit(self, x, a):
        self.W_norm = self.W / self.W.norm(dim=0)
        d = self.W_norm.view(1,512,-1)
        a = a.view(a.size(0),1,-1)
        return x + (d * a).sum(-1)
        

#model discriminator 
class Discriminator(nn.Module):
    def __init__(self, in_channels, c_dim, model_type, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.convs = nn.Sequential(
            ConvLayer(in_channels, 64 * channel_multiplier, 1),               #256
            ResBlock(64 * channel_multiplier,  128 * channel_multiplier),     #256 - 128
            ResBlock(128 * channel_multiplier, 256 * channel_multiplier),     #128 - 64
            ResBlock(256 * channel_multiplier, 512),                          #64  - 32
            ResBlock(512, 512),                                               #32  - 16
            ResBlock(512, 512),                                               #16  - 8
            ResBlock(512, 512)                                                #8   - 4
            )

        self.final_linear = nn.Sequential(
            EqualLinear(512 * 4 * 4, 512, activation="fused_lrelu"),
            EqualLinear(512, 1),
        )
        
        if model_type == 1:
            self.W = nn.Sequential(
                EqualLinear(512 * 4 * 4, 512, activation="fused_lrelu"),
                EqualLinear(512, c_dim),
            )

        self.model_type = model_type        

    def forward(self, input):
        out = self.convs(input)
        batch, channel, height, width = out.shape
        out = out.view(batch, -1)
        if self.model_type == 0:
            return self.final_linear(out), (out * 0).detach()
        else:
            return self.final_linear(out), self.W(out)
        
    
def requires_grad(model, flag=True):
    """
    Return None
    Parameters
    ----------
    model : pytorch model
    flag  : bool, default true
        
    Returns
    -------
    None
    
    set requires_grad flag for model
    
    """     
    
    for p in model.parameters():
        p.requires_grad = flag
        

#calculate generator loss
def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

#VGG Perceptual loss    
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        model = torchvision.models.vgg19(pretrained=True)
        blocks.append(model.features[:2].eval())
        blocks.append(model.features[2:7].eval())
        blocks.append(model.features[7:12].eval())
        blocks.append(model.features[12:21].eval())
        blocks.append(model.features[21:30].eval())
        blocks = nn.ModuleList(blocks)
        self.blocks = torch.nn.ModuleList(blocks)
        self.weights = [1/32.0,1.0/16, 1.0/8, 1.0/4, 1.0] 
        
        for p in self.parameters():
            p.requires_grad = False
        
    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        loss = 0.0
        x = input
        y = target
        for i,block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y) * self.weights[i]
        return loss  

#The function is used downsample and binarize the input  
def downsample(masks):
    masks = F.interpolate(masks,scale_factor= 1/2, mode="bilinear",align_corners=True,recompute_scale_factor=True)   
    m = masks >= 0 #.5
    masks[m]  = 1
    masks[~m] = 0
    return masks   

#calculte r1 loss
def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty
    
class Model(nn.Module):
    def __init__(self, args,c_dim, augment):  
        super().__init__()
        self.args = args
        self.encoder_sketch       = Encoder(1,128, 5)
        self.encoder_img          = Encoder(3,64, 6)
        self.generator            = Generator(c_dim)
        self.classifier           = Classifier(c_dim)
        if args.model_type == 1:
            self.edit             = Embeding(c_dim)  
        self.img_discriminator    = Discriminator(3,c_dim,args.model_type)
        self.domain_discriminator = Domain_Discriminator()
        self.vgg                  = VGGPerceptualLoss()
        self.augment              = augment 
        
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
        
        
        if args.model_type == 0:
        
            self.g_optim = optim.Adam(
                [{'params' : list(self.encoder_sketch.parameters()) + list(self.encoder_img.parameters()) + list(self.generator.parameters())},
                 {'params' : self.classifier.parameters(),"betas": (0.9,0.999), "weight_decay": 0.0005},
                 {'params' : list(self.domain_discriminator.parameters()),"betas": (0.9,0.999), "weight_decay": 0.0005}
                 ],
                lr= args.lr,
                betas=(0, 0.99)
            )
            
            self.d_optim = optim.Adam(
                self.img_discriminator.parameters(),
                lr=args.lr * d_reg_ratio,
                betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
            )
            
        else:
            self.g_optim = optim.Adam(
                [{'params' : list(self.encoder_sketch.parameters()) + list(self.encoder_img.parameters()) + list(self.edit.parameters()) + list(self.generator.parameters())},
                 {'params' : list(self.domain_discriminator.parameters()),"betas": (0.9,0.999), "weight_decay": 0.0005}
                 ],
                lr= args.lr,
                betas=(0, 0.99),
            )
            
            self.d_optim = optim.Adam(
                [{'params' : self.img_discriminator.parameters()},
                 {'params' : self.classifier.parameters(),"betas": (0.9,0.999), "weight_decay": 0.0005}],
                lr=args.lr * d_reg_ratio,
                betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
                )
            
    def forward(self, img = None,sketch = None,sampled_ratio = None, label = None, target_mask = None, domain_img = None, domain_sketch = None, ada_aug_p = None, noise = None,train_discriminator = False, d_regularize = False, train_generator = False, generate = False):
            augment = self.augment    
            if train_discriminator or d_regularize:
                requires_grad(self.encoder_sketch, False)
                requires_grad(self.encoder_img, False)
                requires_grad(self.generator, False)
                requires_grad(self.domain_discriminator, False)
                requires_grad(self.img_discriminator, True)
                if self.args.model_type == 1:
                    requires_grad(self.edit, False)
                    requires_grad(self.classifier, False)
                else:
                    requires_grad(self.classifier, True)
                
            
            else:
                requires_grad(self.encoder_sketch, True)
                requires_grad(self.encoder_img, True)
                requires_grad(self.generator, True)
                requires_grad(self.domain_discriminator, True)
                requires_grad(self.img_discriminator, False)
                if self.args.model_type == 1:
                    requires_grad(self.edit, True)
                    requires_grad(self.classifier, True)
                else:
                    requires_grad(self.classifier, False)
                
            if train_discriminator:
                
                if self.args.model_type == 0:
                    img_latent    = self.encoder_img(img)
                    fake_img      = self.generator(img_latent)
                    
                    if self.args.augment:
                        real_img_aug, _  = augment(img, ada_aug_p)
                        fake_img, _      = augment(fake_img, ada_aug_p)
                    else:
                        real_img_aug = img
                                   
                    fake_img_pred, _   = self.img_discriminator(fake_img)
                    real_img_pred, bce = self.img_discriminator(real_img_aug)
                    
                    return fake_img_pred, real_img_pred, bce
                else:
                    img_latent    = self.encoder_img(img)
                    img_latent_1  = self.edit(img_latent, sampled_ratio)
                    fake_img      = self.generator(img_latent_1)
                                   
                    bce = nn.MSELoss()(self.classifier(img_latent),  label * 2 - 1) 
                    
                    if self.args.augment:
                        real_img_aug, _  = augment(img, ada_aug_p)
                        fake_img, _      = augment(fake_img, ada_aug_p)
                    else:
                        real_img_aug = img
                    
                    fake_img_pred, _ = self.img_discriminator(fake_img)
                    real_img_pred, real_class = self.img_discriminator(real_img_aug)
                    
                    outer_bce = nn.BCEWithLogitsLoss()(real_class,  label)
                    
                    return  fake_img_pred, real_img_pred, bce + outer_bce * 0.0
            
            if d_regularize:
                real_pred_img, _  = self.img_discriminator(img)
                r1_loss           = d_r1_loss(real_pred_img,img)
                return r1_loss 
            
            if train_generator:
                img_latent       = self.encoder_img(img)
                sketch_latent    = self.encoder_sketch(downsample(sketch))
                sketch_loss      = nn.L1Loss()(sketch_latent, img_latent.detach())
                reconstruct_img  = self.generator(img_latent)
                vgg_loss         = self.vgg(reconstruct_img,img)
                reconstruct_loss = nn.L1Loss()(reconstruct_img,img)
                domain_loss      = nn.BCEWithLogitsLoss()(self.domain_discriminator(img_latent.detach()), domain_img) + \
                                   nn.BCEWithLogitsLoss()(self.domain_discriminator(sketch_latent),  domain_sketch)
                    
                if self.args.model_type == 0:
                    
                    bce,orthologoy      = self.classifier(img_latent, True)
                    bce                 = nn.MSELoss()(bce,  label * 2 - 1) 
                
                    if self.args.augment:
                        reconstruct_img, GC  = augment(reconstruct_img, ada_aug_p)
                    
                    fake_pred_img, _    = self.img_discriminator(reconstruct_img)
                    g_loss_img          = g_nonsaturating_loss(fake_pred_img)
                    
                    g_total  = sketch_loss * 2.5 +\
                               domain_loss * 0.1 +\
                               vgg_loss * 2.5 +\
                               reconstruct_loss * 2.5 +\
                               g_loss_img +\
                               bce * 0.5 +\
                               orthologoy
                    return g_total
                
                else:
                    img_latent_1    = self.edit(img_latent,  sampled_ratio)
                    sketch_latent_1 = self.edit(sketch_latent,    sampled_ratio)
                    
                    fake_img        = self.generator(img_latent_1)
                    reg             = self.edit(img_latent, sampled_ratio * 0.0, reg = True).abs().mean()
                    
                    latent_reconstruct  = (self.edit(self.edit(img_latent.detach(),sampled_ratio), -sampled_ratio) - img_latent.detach()).abs().mean()
                    base_score          = self.classifier(img_latent).detach() + sampled_ratio
                    edit_loss           = nn.MSELoss()(self.classifier(img_latent_1), base_score)
                    domain_loss         = domain_loss + \
                                          nn.BCEWithLogitsLoss()(self.domain_discriminator(img_latent_1.detach()),   domain_img) + \
                                          nn.BCEWithLogitsLoss()(self.domain_discriminator(sketch_latent_1),domain_sketch)
                               
                    if self.args.augment:
                        fake_img, _  = augment(fake_img, ada_aug_p)
            
                    fake_pred_img, fake_class = self.img_discriminator(fake_img)
                    g_loss_img                = g_nonsaturating_loss(fake_pred_img)
                    
                    outer_edit = nn.BCEWithLogitsLoss()(fake_class,target_mask * 1.0)
                    
                    g_total = vgg_loss * 2.5 +\
                              reg * 0.1 +\
                              (edit_loss + outer_edit * 0.0) * 1.0 +\
                              reconstruct_loss * 1.0  +\
                              latent_reconstruct * 1.0 +\
                              sketch_loss * 2.5 +\
                              domain_loss  * 0.05 +\
                              g_loss_img   
                              
                    return g_total
            
            if generate:
                img    = self.encoder_img(img)
                sketch = self.encoder_sketch(downsample(sketch))
                if self.args.model_type == 0:
                    sketch = self.classifier.edit(sketch,sampled_ratio)
                else:
                    sketch = self.edit(sketch,sampled_ratio)
                img    = self.generator(img)
                sketch = self.generator(sketch)
                return img,sketch
                
        
        
        
        

