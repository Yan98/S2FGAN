#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is used to train the S2F GAN.
The module structure is the following:
    - A parser used to read the parameters from users.
    - Check if cua installed
    - Create folders "images" and "saved_models" to store the samples and checkpoints during training.
    - Set torch home that is used by pytorch to store the pretrained models 
    - Initialize S2F GAN and optmizers
    - A data_prefetecher is used to load the inputs to cuda during training.
    - A process function used to scales inputs
    - A sample_images function used to sample images during training.
    - A R1_penalty used to calculate the gradient penalty of predicting true samples as fake samples
    - A train loop used to call and excute the script.

The training logs will be stored in log.txt
"""

import argparse
import datetime
import numpy as np
from models import Encoder,VGGPerceptualLoss,Decoder,Discriminator
from apex import amp
from dataset import CeleDataset
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import os
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0015, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--lambda_vgg", type=int, default=3, help="weight of vgg perceptual loss")
parser.add_argument("--lambda_reconst", type=int, default=3, help="weight of image reconstruction loss")
parser.add_argument("--lambda_ratio", type=int, default=3, help="weight of attributes reconstruction loss")
parser.add_argument("--channels", type=int, default=1, help="number of image channels, sketch:1, mask:18")
parser.add_argument("--sample_interval", type=int, default= 1, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")
parser.add_argument("--TORCH_HOME", type=str, default="None", help="where to load/save pytorch pretrained models")
parser.add_argument("--NumOfImage", type=int, default= 10, help = "number of images in the zip")
parser.add_argument("--imageZip", type=str, default= "data/CelebAMask-HQ-Sample.zip", help = "input image zip")
parser.add_argument("--imagePath",type=str, default= "CelebAMask-HQ-Sample/CelebA-HQ-img", help = "path of images in the zip")
parser.add_argument("--hedEdgeZip", type=str, default= "data/hed_edge_256-Sample.zip", help = "hed sketch zip")
parser.add_argument("--hedEdgePath", type=str, default= "hed_edge_256-Sample", help = "path of sketch in the zip")
parser.add_argument("--maskZip",  type=str, default= "data/CelebAMaskHQ-mask_256-sample.zip", help = "mask zip")
parser.add_argument("--label_path", type = str, default = "data/CelebAMask-HQ-attribute-anno.txt", help = "attributes annotation text file of CelebAMask-HQ")
parser.add_argument("--task_type", type = int, default = 0, help = "0- edge to image, 1-mask to image")
parser.add_argument(
    "--selected_attrs",
    type = list,
    nargs="+",
    help="selected attributes for the CelebAMask-HQ dataset",
    default=["Smiling", "Male","No_Beard", "Eyeglasses","Young", "Bangs", "Narrow_Eyes", "Pale_Skin", "Big_Lips","Big_Nose","Mustache","Chubby"],
)
parser.add_argument(
    "--ATMDTT",
    type = list,
    nargs="+",
    help="Attributes to manipulate during testing time",
    default=    
    [[1,0,0,0,0,0,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0,0,0,0,0]
     ]
)

opt = parser.parse_args()

#write the paramters to train S2FGAN in log.txt
with open("log.txt","a") as f:
    f.write(str(opt) + "\n")

#create folders to store samples and checkpoints
os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
#Set TORCH_HOME to system enviroment.
if opt.TORCH_HOME != "None":
    os.environ['TORCH_HOME'] = opt.TORCH_HOME

if opt.img_height == 128:
    STEP = 1
elif opt.img_height == 256:
    STEP = 2
elif opt.img_height == 512:
    STEP = 3
else: 
    raise SystemExit("Unrecognized Image Resolution") 

#Sanity check of GPU installation
if not torch.cuda.is_available():
    raise SystemExit("GPU Required")

#Speed up training
torch.backends.cudnn.benchmark=True

#Initialze loss functions
criterion_reconst = torch.nn.L1Loss()
criterion_GAN     = torch.nn.MSELoss()
criterion_ratio   = torch.nn.MSELoss()

#Initialize loss weight
lambda_vgg     = opt.lambda_vgg
lambda_reconst = opt.lambda_reconst
lambda_ratio   = opt.lambda_ratio
 
#Initialize S2FGAN and perceptual loss
E1         = Encoder(opt.channels)
vgg        = VGGPerceptualLoss()
D1         = Decoder(c_dim = len(opt.selected_attrs),step = STEP)
des1       = Discriminator((3, opt.img_height, opt.img_width), c_dim = len(opt.selected_attrs),step = STEP)

#move S2FGAN and perceptual loss to cuda.
E1 = E1.cuda()
vgg= vgg.cuda()
D1 = D1.cuda()
    
des1  = des1.cuda()
criterion_reconst.cuda()
criterion_GAN.cuda()
criterion_ratio.cuda()
    
#Initialize model optimizers
optimizer_G = torch.optim.Adam(
    [{'params': E1.parameters()},
     {'params': D1.parameters()}
     ],
    lr=opt.lr, 
    betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(
    des1.parameters(), 
    lr=opt.lr, 
    betas=(opt.b1, opt.b2))

#using AMP libray to place part of model paramters as float16
[E1,D1,des1,vgg],[optimizer_D,optimizer_G] = amp.initialize([E1,D1,des1,vgg],[optimizer_D,optimizer_G],opt_level = "O1", num_losses = 2) 

#If there are multi-GPUS, the data parallel will be used.
if torch.cuda.device_count() > 1:
    E1     = nn.DataParallel(E1)
    vgg    = nn.DataParallel(vgg)
    D1     = nn.DataParallel(D1)
    des1   = nn.DataParallel(des1)
    
#if the starting epoch is not 0, then load the checkpoints.    
if opt.epoch != 0:
    checkpoints = torch.load("saved_models/checkpoits_%d.pt" % opt.epoch)
    E1.load_state_dict(checkpoints["E1"])
    D1.load_state_dict(checkpoints["D1"])
    des1.load_state_dict(checkpoints["des1"])
    optimizer_G.load_state_dict(checkpoints["optimizer_G"])
    optimizer_D.load_state_dict(checkpoints["optimizer_D"])
    amp.load_state_dict(checkpoints["amp"])

#The mean and standard deviation for normalizing S2FGAN inputs.
mean = torch.tensor([0.5 * 255]).cuda().view(1,1,1,1)
std = torch.tensor([0.5 * 255]).cuda().view(1,1,1,1)

def process(x,img,labels):
    """
    Return normalized x and images
    Parameters
    ----------
    X : torch.cuda Tensor of shape (batch_size,channels,height,width)
    img : torch.cuda Tensor of shape (batch_size,channels,height,width)
    labels : torch.cuda Tensor of shape (batch_size, number of attributes)
    
    Returns
    -------
    X : noramlised X.
    img : normalised img.
    labels: same
    """
    #sanity check to, transfer dtype of x and img to fload32.
    x        = x.float()
    img      = img.float()
    
    if opt.task_type != 1:
        x        =  (x.sub_(mean).div_(std))
    img      =  (img.sub_(mean).div_(std))
    return x,img,labels

class data_prefetcher():
    '''
    A wrapper of dataloader, to load the data to cuda and process it during S2F training.
    '''
    def __init__(self, data):
        """
        Return None
        Parameters
        ----------
        data : pytorch data loader.
        
        Returns
        -------
        None
        
        Initialize cuda stream and preload the data when intialize the classes
        """
        self.data = iter(data)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        """
        Return None
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        load the data to cuda and process data using process function. Here is concurrent happens.
        """        
        try:
            self.next_input  = next(self.data)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = process(*[i.cuda(non_blocking=True) for i in self.next_input])
            
    def next(self):
        """
        Return None
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        Synchronise the stream, return preloaded data, and load data for next batch. 
        """            
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        self.preload()
        return input



# Initialize pytorch dataloaders
dataloader = DataLoader(
    CeleDataset(opt,True, STEP),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=1
)

#calculate the batch size based on ATMDTT for the validation data loader
val_batch_size = len(opt.ATMDTT) + 2

val_dataloader = DataLoader(
    CeleDataset(opt,False, STEP),
    batch_size=val_batch_size,
    shuffle=True,
    num_workers=1
)

#Rename the cuda tensor for using purpose
Tensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
BoolTensor = torch.cuda.BoolTensor


#Intialise the intensity control parameters
LABELS = Tensor(opt.ATMDTT)
scale =  Tensor([-3.0,-2.5, -2.0,-1.5, -1.0, -0.5,0,0.5, 1.0,1.5,2.0,2.5,3.0]).view(13,1)

#The rgb color used to visualise the mask
color_list = [[204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

#Transfer the binary mask to rgb color for visualization purpose
def toColor(mask):
    mask = mask.view(opt.channels,128,128)
    img  = torch.zeros((3,128,128),device = mask.device)
    for idx,color in enumerate(color_list):
        img[0,mask[idx] == 1] = color[0]
        img[1,mask[idx] == 1] = color[1]
        img[2,mask[idx] == 1] = color[2]
    return (img.view(1,3,128,128) - mean)/std

#used to sample images during training
def sample_images(batches_done):
    """
    Return None
    Parameters
    ----------
    batches_done:int, how many batches the model has been trained. It is used for naming samples.
    
    Returns
    -------
    None
    Sample images according to the ATMDTT and intensity controls.
    Save the samples to images folder.
    """  
    x, img, label = process(*[i.cuda(non_blocking=True) for i in next(iter(val_dataloader))])
    current_label = torch.cat((LABELS,label[-2:]))
    img_samples = None
    
    for e,i,l,a in zip(x,img,current_label,label):
        
        if STEP == 3:
            scale_factor = 4
        elif STEP == 2:
            scale_factor = 2
        else:
            scale_factor = 1
        if opt.task_type == 0:
            d  =  e.view(-1,opt.channels,128,128).repeat(1,3,1,1)
        else:
            d  = toColor(e)
        d =  F.interpolate(d,scale_factor = scale_factor, mode = 'bilinear')[0]
        e  =  e.view(1,opt.channels,128,128).repeat(13,1,1,1) 
        l  =  l.view(1,len(opt.selected_attrs)).repeat(13,1) * scale
        decode,latent = E1(e)
        im = D1(decode,latent,l)
        im = torch.cat([x for x in im],-1) 
        img_sample = torch.cat((d,i,im),-1).unsqueeze(0)
        img_samples = img_sample if img_samples is None else torch.cat((img_samples,img_sample),-2)
    save_image(img_samples, "images/%d.png" %  batches_done, nrow=16, normalize=True, range=(-1,1))
              
def R1_penalty(real_pred,real_img):
    """
    Return None
    Parameters
    ----------
    real_pred: (batch_size, 1) The prediction scores of real images 
    real_img:  (batch_size, channels, height, weight) The batch of images corresponding to real_pred
    Returns
    -------
    grad_penalty: The gradient penalty of predicting real images as fake images.
    Note the sign of real_pred not been changed.
    """  
    grad_real = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )[0]
    grad_penalty = (
        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
        ).mean()
    grad_penalty = 10 / 2 * grad_penalty
    return grad_penalty

# ----------
#  Training
# ----------
#Record starting time, to estimate the time need for training.
start_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    pref = data_prefetcher(dataloader)
    data = pref.next()
    #record how many batch has been processed in this epoch
    i = 0
    while data is not None:
    
        x,img,label = data
        
        #samples attribute shiting vector
        sampled_ratio = -Variable(Tensor(np.random.uniform(-1,3, (x.size(0), len(opt.selected_attrs))))) * label
        target_ratio  = label + sampled_ratio
        img.requires_grad = True
        
        # -----------------
        #  Train Generator
        # -----------------
                    
        optimizer_G.zero_grad()
        decode,latent = E1(x)
        
        # sample image
        sample_img  = D1(decode,latent, sampled_ratio)
        #reconstruct image
        reconst_img   = D1(decode,latent,torch.zeros((x.size(0),len(opt.selected_attrs)),device = x.device))
        
        #adv loss
        sample_valid, predict_ratio = des1(sample_img)           
        loss_GAN    = torch.mean(F.softplus(-sample_valid))
        #pixel loss    
        loss_pixel  = criterion_reconst(reconst_img, img).mean()
        #vgg loss
        loss_vgg    = vgg(reconst_img, img).mean()
        #attribute reconstruction loss
        loss_ratio  =  criterion_ratio(predict_ratio, target_ratio).mean()
        #total loss    
        loss_G = loss_GAN + loss_pixel * lambda_reconst + loss_ratio * lambda_ratio + loss_vgg * lambda_vgg 
        with amp.scale_loss(loss_G, optimizer_G,loss_id = 0) as scaled_loss:
            scaled_loss.backward()   
            
        optimizer_G.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()  
        valid, clf = des1(img)
        fake, _    = des1(sample_img.detach()) 
        loss_cls   = criterion_ratio(clf, label)
        #calculate accuracy 
        acc_D  = ((clf > 0) == (label > 0)).sum().type(Tensor) / label.numel()
        # Gradient penalty
        gradient_penalty = R1_penalty(valid,img)
        loss_D   = torch.mean(F.softplus(-valid)) + torch.mean(F.softplus(fake)) + loss_cls + gradient_penalty
        with amp.scale_loss(loss_D,optimizer_D,loss_id = 1) as scaled_loss:
            scaled_loss.backward()
        optimizer_D.step()
        # Estimate the time left to train S2FGAN.
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - start_time) / (batches_done + 1))
        
        # Sample Image and Print log
        if batches_done % opt.sample_interval == 0:
            # Print log
            with open("log.txt","a") as f:
                f.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D adv: %f, gp: %f, cls: %f, acc: %f] [G loss: %f, pixel: %f, vgg: %f, ratio: %f, adv: %f] ETA: %s"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        loss_D.item() - loss_cls.item() - gradient_penalty.item(),
                        gradient_penalty.item(),
                        loss_cls.item(),
                        acc_D.item(),
                        loss_G.item(),
                        loss_pixel.item(),
                        loss_vgg.item(),
                        loss_ratio.item(),
                        loss_GAN.item(),
                        time_left,
                    ) +"\n"
                )
            #sample images according to sample interval
            with torch.no_grad():
                E1.eval()
                D1.eval()
                sample_images(batches_done)    
                E1.train()
                D1.train()
        #load data for next batch.
        data = pref.next()
        #record how many batch has been processed in this epoch
        i += 1
        
    # Save model checkpoints
    if opt.checkpoint_interval != -1 and (epoch + 1) % opt.checkpoint_interval == 0:
        checkpoits = {
            "E1":E1.state_dict(),
            "D1":D1.state_dict(),
            "des1":des1.state_dict(),
            "optimizer_G":optimizer_G.state_dict(),
            "optimizer_D":optimizer_D.state_dict(),
            "amp": amp.state_dict()
            }
        torch.save(checkpoits, "saved_models/checkpoits_%d.pt" %  epoch)

            
            
            
