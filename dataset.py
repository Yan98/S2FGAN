#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is the concrete implementation of pytorch dataset.
The module structure is the following:
    - extract_zip is used to read the image from the zip into memory
    - read_image parse the file to PIL image
    - mask_standarise is used to change the rgb mask to binary mask
    - CelebDataset is the wrapper of CelebAMASK-HQ dataset.
"""

import torch.utils.data as data
from PIL import Image
from io import BytesIO
import  albumentations as A
import numpy as np
import torch
import zipfile



def extract_zip(input_zip):
    '''
    Parameters
    ----------
    input_zip : zipfile, the zipfile need to be read in memory
    Returns
    -------
    dict:  a dictionary maps the path to ".jpg" or ".png" image in the zipfile
    '''
    input_zip=zipfile.ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist() if name.endswith(".jpg") or name.endswith(".png")}

def read_image_from_zip(file,path,height = None,width = None):
    """
    Parameters
    ----------
    file: zipfile, the zipfile need to be read
    path: str, the path to read in the file.
    height: int, the height of the image desired
    width: int, the width of the image desired.
    
    Returns
    -------
    img: a PIL image with desired height and width
    """ 
    
    img = Image.open(BytesIO(file[path]))
    
    if height != None and width != None:
        img = img.resize((height,width))
    return img

#mask_standarise is used to change the rgb mask to binary mask
def mask_standardlize(mask,depth = 18):
    """
    Return, A Binary mask 
    Parameters
    ----------
    mask: (height,width), a numpy array that stores the mask
    depth: int, the number of components in the mask
    
    Returns
    -------
    target: a binary mask
    """     

    h,w  = mask.shape
    target = np.zeros((h,w,depth))
    
    for k in range(depth):
        mm = (mask == k+1)
        target[mm,k] = 1
    return target


class CeleDataset(data.Dataset):
    '''
    The pytorch dataset wrapper for the CelebAMASK-HQ dataset.
    
    '''
    def __init__(self,params,train = True,STEP = 1):
        """
        Return, None 
        Parameters
        ----------
        params: A parser file which contains the parameters for the class.
        train: boolean, decide if the class is used for trainning.
        
        Returns
        -------
        None
        """  
        #record paramters, intialize zipfile reading
        #generate a dict that maps int to files
        #read the attributes specificed by the users
        #create data augmentation class
        
        self.params      = params
        self.image_zip   = extract_zip(params.imageZip)
        if params.task_type == 0:
            self.hedZip      = extract_zip(params.hedEdgeZip)
        else:
            self.mask_zip    = extract_zip(params.maskZip)
        self.indexToPath = self.generate_path(train)
        self.att         = self.get_annotations()
        self.train       = train
        self.step        = STEP
        self.aug = A.Compose({
            A.RandomSizedCrop(min_max_height = (int(self.params.img_height * 0.8),self.params.img_height),height = self.params.img_height,width = self.params.img_width, p = 0.5),
            A.HorizontalFlip(p=0.5)
        })
        
        self.rs  = A.Resize(128,128)
        
    def get_annotations(self):
        """
        Return, A dict contains the attributes of interest. 
        Parameters
        ----------
        None
        
        Returns
        -------
        annotations, dict, read the selected attributes, and store it in the annoations.
        """ 
        annotations = {}
        lines = [line.rstrip() for line in open(self.params.label_path, "r")]
        self.label_names = lines[1].split()
        for _, line in enumerate(lines[2:]):
            filename, *values = line.split()
            labels = []
            for attr in self.params.selected_attrs:
                idx = self.label_names.index(attr)
                labels.append((1 if (values[idx] == "1") else -1))
            annotations[filename.replace(".jpg",".png")] = labels
        return annotations

        
    def generate_path(self,train):
        """
        Return, A dict that mapps integers to files.
        Parameters
        ----------
        train, bool, decide which files to read. Training and testing will lead reading diffirent files
        
        Returns
        -------
        selected_index_ToPath, dict, the dictionary contains the mapping of integer and files
        """ 
        #read all files intially
        indexToPath = dict()
        index = 0
        for file in range(self.params.NumOfImage):
            file = str(file)
            file += ".png"
            indexToPath[index] = [
                self.params.imagePath + "/" +  file.replace(".png",".jpg"),
                file
                ]
            index += 1
        #select the files according to the paramters, and sort index.    
        selected_indexToPath = dict()
        new_index = 0
        for k, value in indexToPath.items():
            
            if not train:
                if k % 20 == 0:
                    selected_indexToPath[new_index] = value
                    new_index+=1  
            else:
                if k % 20 != 0:
                    selected_indexToPath[new_index] = value
                    new_index+=1
                
        return selected_indexToPath
    

    def  __getitem__(self, index):
        
        """
        Return, x,img,label
        Parameters
        ----------
        index: int, the index of the file need to be read
        
        Returns
        -------
        x: pytorch float tensor, depends on the task type, it could be sketch, low resolution image, mask.
        img: pytorch float tensor, the ground truth image corresponds to x.
        labels: pytorch float tensor, the attributes of the img
        """ 
        #get path for x and image
        image_path, segt_path = self.indexToPath[index]
        
        #read image
        img = read_image_from_zip(self.image_zip,image_path,self.params.img_height,self.params.img_width)
        img  = np.array(img)

        #read x depends on task type
        if self.params.task_type == 0:
            x    = read_image_from_zip(self.hedZip,self.params.hedEdgePath + "/" + segt_path.replace(".png",".jpg"),self.params.img_height,self.params.img_width)
        
        elif self.params.task_type == 1:
            x    = read_image_from_zip(self.mask_zip,segt_path,self.params.img_height,self.params.img_width)
        else:
            raise SystemExit("Unrecognized task type")

        if self.params.task_type == 0:
            x    = np.array(x)
        elif self.params.task_type == 1:
            x    = mask_standardlize(np.array(x)[...,0],self.params.channels)
  
        if self.train:
            augmented = self.aug(image = img,mask = x)
            img = augmented['image']
            x = augmented['mask']
        
        if self.step != 1:
            x = self.rs(image = x)['image']
            
        img     = torch.FloatTensor(img).permute(2,0,1)
        
        if self.params.task_type ==  0: 
            x  = torch.FloatTensor(x).unsqueeze(2)
        x  = torch.FloatTensor(x).permute(2,0,1)

        #read labels into pytorch float tensor. 
        label = self.att[segt_path]
        label = torch.FloatTensor(np.array(label))
        
        return x,img,label
    

    def __len__(self):
        """
        Return,  the number of ground truth images in the files
        Parameters
        ----------
        None
        
        Returns
        -------
        The number of ground truth images in the files
        """         
        
        return len(self.indexToPath)





