"""
This module is used to train the S2F GAN.
The module structure is the following:
    - A print function used to record training log 
    - A parser used to read the parameters from users.
    - Set torch home that is used by pytorch to store the pretrained models 
    - Initialize S2F GAN and optmizers
    - A data_prefetecher is used to load the inputs to cuda during training.
    - A train function used to call and excute the script.
The training logs will be stored in log.txt
"""

import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import utils
from model import Model as S2FGAN
from dataset import CeleDataset
from non_leaking import augment
import time
import datetime
import torch.backends.cudnn as cudnn

#Speed up training
cudnn.benchmark = True

#write the paramters to train S2FGAN in log.txt
def print(x):
    with open("log.txt","a") as f:
        f.write(str(x) + "\n")
        
def accumulate(model1, model2, decay=0.999):
    """
    Return None
    Parameters
    ----------
    model1 : pytorch model
    model2 : pytorch model
    decay  : int, default 0.999, the speed of updating model1 parameter
        
    Returns
    -------
    None
    Update model1 paramter by model2 paramter.
    """  
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
        
class data_prefetcher():
    '''
    A wrapper of dataloader, to load the data to cuda and process it during S2F training.
    '''
    def __init__(self, loader):
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
        self.loader = iter(loader)
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
            self.next_input  = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = [i.cuda(non_blocking=True) for i in self.next_input]
            
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

def sample_data(loader,device):
    """
    Return normalized sketch, normalized images and label
    Parameters
    ----------
    loader : pytorch loader
    device : cuda device name
    
    Returns
    -------
    sketch : noramlised X.
    img    : normalised img.
    labels : same
    """
    while True:
        pref = data_prefetcher(loader)
        data = pref.next()
        while data is not None:
            [sketch,img,label] = data
            sketch = (sketch - 255 * 0.5) / (255 * 0.5)
            img    = (img - 255 * 0.5) / (255 * 0.5)
            label  = label
            data = pref.next()
            yield [sketch,img,label]


def train(args, dataloader_train,dataloader_val, models, g_optim, d_optim, device):
    
    """
    Return normalized sketch, normalized images and label
    Parameters
    ----------
    args             : args for S2FGAN
    dataloader_train : dataloader for training
    dataloader_val   : dataloader for evaluation
    models           : S2FGAN models
    g_optim          : generator optimizer 
    d_optim          : discriminator optimizer
    device           : cuda device
    
    Returns
    -------
    None
    A trained S2FGAN.
    """
    
    [model,model_ema] = models
    
    #speed data loading and process data
    loader     = sample_data(dataloader_train, device)
    loader_val = sample_data(dataloader_val,device)
    
    print("Trianing start")

    loss_dict = {}

    model_module = model.module

    #intialize paramters for adaptive discriminator agumentation.
    accum = 0.5 ** (32 / (10 * 1000))
    ada_augment = torch.tensor([0.0, 0.0], device=device)
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    ada_aug_step = args.ada_target / args.ada_length
    r_t_stat = 0

    #Record starting time, to estimate the time need for training.    
    start_time = time.time()

    for idx in range(args.iter):
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        sketch,img,label = next(loader)
        
        #samples attribute shiting vector
        sampled_ratio = torch.FloatTensor(np.random.uniform(-4,4, (sketch.size(0), c_dim))).to(device)
        sampled_mask  = torch.FloatTensor(np.random.randint(0,2, (sketch.size(0), 1)) * 1.0).to(device)
        sampled_ratio = sampled_ratio * sampled_mask
        target_ratio  = (label * 2 - 1)  + sampled_ratio
        target_mask   = target_ratio >= 0
        
        #create domain label for sketch and img
        domain_sketch = torch.zeros((sketch.size(0),1)).type(torch.FloatTensor).to(device)
        domain_img    = torch.ones((img.size(0),1)).type(torch.FloatTensor).to(device)
                    
        
        fake_img_pred, real_img_pred, bce = model(img,sketch,sampled_ratio,label,target_mask, ada_aug_p = ada_aug_p, train_discriminator = True)
        
        d_loss    = F.softplus(-real_img_pred).mean() + F.softplus(fake_img_pred).mean() + bce.mean()
        
        loss_dict["d_loss"] = d_loss
        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()

        for real_pred in [real_img_pred]:
            if args.augment and args.augment_p == 0:
                ada_augment_data = torch.tensor(
                    (torch.sign(real_pred).sum().item(), real_pred.shape[0]), device=device
                )
                ada_augment += ada_augment_data
    
                if ada_augment[1] > 255:
                    pred_signs, n_pred = ada_augment.tolist()
                    r_t_stat = pred_signs / n_pred
                    
                    if r_t_stat > args.ada_target:
                        sign = 1
    
                    else:
                        sign = -1
                    ada_aug_p += sign * ada_aug_step * n_pred
                    ada_aug_p = min(1, max(0, ada_aug_p))
                    ada_augment.mul_(0)
                
        d_regularize = i % args.d_reg_every == 0
        if d_regularize:
            img.requires_grad = True
            
            r1_loss = model(img, d_regularize = True)
            r1_loss = r1_loss.mean()
        
            d_optim.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every).backward()
            d_optim.step()

        loss_dict["r1"] = r1_loss
        
        img.requires_grad = False
        
        #samples attribute shiting vector
        sampled_ratio = torch.FloatTensor(np.random.uniform(-4,4, (sketch.size(0), c_dim))).to(device)
        sampled_mask  = torch.FloatTensor(np.random.randint(0,2, (sketch.size(0), 1)) * 1.0).to(device)
        sampled_ratio = sampled_ratio * sampled_mask
        target_ratio  = (label * 2 - 1)  + sampled_ratio
        target_mask   = target_ratio >= 0

        
        g_loss = model(img,sketch,sampled_ratio,label,target_mask, domain_img,domain_sketch, ada_aug_p = ada_aug_p,train_generator = True)
        g_loss = g_loss.mean() 
        
        loss_dict["g_loss"]     = g_loss
        
        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()

        accumulate(model_ema, model_module, accum)
        
        loss_reduced = loss_dict

        d_loss    = loss_reduced["d_loss"].item()
        g_loss    = loss_reduced["g_loss"].item()
        r1        = loss_reduced["r1"].item()
        
        #Print log
        if i % 10 == 0:
            # Determine approximate time left
            batches_done = idx
            batches_left = args.iter - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - start_time) / (batches_done + 1))
                
            print(
                (
                    f"Epoch[{idx}/{args.iter}]; augment: {ada_aug_p:.4f}; "
                    f"d_loss: {d_loss:.4f}; g_loss: {g_loss:.4f}; r1: {r1:.4f}; ETA: {time_left}"
                )
            )
            
        #sample images
        if i % 400 == 0:
            sketch,img,label = next(loader_val)
            with torch.no_grad():
                samples = None
                for e, j,l in zip(sketch,img,torch.cat((LABELS,label[-2:]))):
                    d =   e.view(1,args.img_height,args.img_width).repeat(3,1,1)
                    e  =  e.view(1,1,256,256).repeat(13,1,1,1) 
                    l  =  l.view(1,12).repeat(13,1) * SCALE
                    k,im = model_ema(j.view(1,3,256,256),sketch = e,sampled_ratio = l,generate = True) 
                    im = torch.cat([x for x in im],-1) 
                    sample = torch.cat((d,k.view(3,256,256),j,im),-1).unsqueeze(0)
                    samples = sample if samples is None else torch.cat((samples,sample),-2)
                    
                utils.save_image(
                    samples,
                    f"sample/{str(i).zfill(6)}.png",
                    nrow= 16,
                    normalize=True,
                    range=(-1, 1),
                    )
                 
        # Save model checkpoints
        if i % 10000 == 0:
            torch.save(
                {
                    "model"    :model_module.state_dict(),
                    "model_ema":model_ema.state_dict()
                },
                f"checkpoint/{str(i).zfill(6)}.pt",
            )


if __name__ == "__main__":
    device = "cuda"
    
    parser = argparse.ArgumentParser(description="S2FGAN trainer")

    parser.add_argument(
                        "--iter", 
                        type=int, 
                        default=100, 
                        help="total training iterations"
                        )
    parser.add_argument(
                        "--batch",
                        type=int, 
                        default = 4, 
                        help="batch sizes"
                        )
    
    parser.add_argument(
                        "--r1",
                        type=float, 
                        default=1, 
                        help="weight of the r1 regularization"
                        )

    parser.add_argument(
                        "--d_reg_every",
                        type=int,
                        default=16,
                        help="interval of the applying r1 regularization",
                        )

    parser.add_argument(
                        "--lr", 
                        type=float, 
                        default=0.002, 
                        help="learning rate"
                        )

    parser.add_argument(
                        "--augment", 
                        type=bool, 
                        default=True, 
                        help="apply discriminator augmentation"
                        )
                    
    parser.add_argument(
                        "--augment_p",
                        type=float,
                        default=0,
                        help="probability of applying augmentation. 0 = use adaptive augmentation",
                        )
    parser.add_argument(
                        "--ada_target",
                        type=float,
                        default=0.6,
                        help="target augmentation probability for adaptive augmentation",
                        )
    
    parser.add_argument(
                        "--ada_length",
                        type=int,
                        default=500 * 1000,
                        help="target duraing to reach augmentation probability for adaptive augmentation",
                        )
    parser.add_argument(
                        "--ada_every",
                        type=int,
                        default=256,
                        help="probability update interval of the adaptive augmentation",
                        )
    
    parser.add_argument(
                        "--img_height", 
                        type=int, 
                        default=256, 
                        help="size of image height"
                        )
    
    parser.add_argument(
                        "--img_width", 
                        type=int, 
                        default=256, 
                        help="size of image width"
                        )
    parser.add_argument(
                        "--NumOfImage", 
                        type=int, 
                        default= 10, 
                        help = "number of images in the zip"
                        )
    
    parser.add_argument(
                        "--imageZip", 
                        type=str, 
                        default= "data/CelebAMask-HQ-Sample.zip"
                        )
    
    parser.add_argument(
                        "--hedEdgeZip", 
                        type=str, 
                        default= "data/hed_edge_256-Sample.zip"
                        )
    
    parser.add_argument(
                        "--hedEdgePath", 
                        type=str, 
                        default= "hed_edge_256-Sample"
                        )
    
    parser.add_argument(
                        "--imagePath",
                        type=str, 
                        default= "CelebAMask-HQ-Sample/CelebA-HQ-img"
                        )
    
    parser.add_argument(
                        "--TORCH_HOME", 
                        type=str, 
                        default="None", 
                        help="where to load/save pytorch pretrained models"
                        )
    
    parser.add_argument(
                        "--selected_attrs",
                        type = list,
                        nargs="+",
                        help="selected attributes for the CelebAMask-HQ dataset",
                        default=["Smiling", "Male","No_Beard", "Eyeglasses","Young", "Bangs", "Narrow_Eyes", "Pale_Skin", "Big_Lips","Big_Nose","Mustache","Chubby"],
                        )
    
    parser.add_argument(
                        "--label_path", 
                        type = str, 
                        default = "data/CelebAMask-HQ-attribute-anno.txt", 
                        help = "attributes annotation text file of CelebAMask-HQ"
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
    
    parser.add_argument(
                        "--model_type", 
                        type = int, 
                        default = 0, 
                        help = "0- S2F-DIS, 1- S2F-DEC"
                        )

    args = parser.parse_args()
    
    args.start_iter = 0
    c_dim = len(args.selected_attrs)
    
    #create folders to store samples and checkpoints
    os.makedirs("sample", exist_ok=True)
    os.makedirs("checkpoint", exist_ok=True)
    
    #Set TORCH_HOME to system enviroment.
    if args.TORCH_HOME != "None":
        os.environ['TORCH_HOME'] = args.TORCH_HOME


    #Sanity check of GPU installation
    if not torch.cuda.is_available():
        raise SystemExit("GPU Required")

    #initialization
    model     = S2FGAN(args,c_dim,augment).to(device) 
    
    model_ema = S2FGAN(args,c_dim,augment).to(device)
    
    accumulate(model_ema, model, 0)
    model_ema.eval()
    
    #get model optimizer
    
    g_optim = model.g_optim
    d_optim = model.d_optim
            
    model = nn.DataParallel(model)
    #initialize dataloader
    
    dataset = CeleDataset(args, True)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        num_workers = 4,
        drop_last = True
    )

    dataset_val = CeleDataset(args, False)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=len(args.ATMDTT) + 2,
        num_workers=4
    )
    
    
    #Intialise the intensity control parameters for demonstration
    LABELS = torch.FloatTensor(args.ATMDTT).to(device)
    SCALE =  torch.FloatTensor([-4.0,-3.0, -2.0,-1.5, -1.0, -0.5,0,0.5, 1.0,1.5,2.0,3.0,4.0]).to(device).view(13,1)
    
    #start training
    train(args,loader,dataloader_val,[model,model_ema], g_optim, d_optim, device)
