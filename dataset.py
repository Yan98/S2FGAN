#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is the concrete implementation of pytorch dataset.
The module structure is the following:
    - extract_zip is used to read the image from the zip into memory
    - read_image_from_zip parse the file to PIL image
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

class CeleDataset(data.Dataset):
    
    '''
    The pytorch dataset wrapper for the CelebAMASK-HQ dataset.
    
    '''
    
    def __init__(self,params,train = True):
        
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
        
        global selected_attrs,label_path
        
        selected_attrs  = params.selected_attrs
        label_path      = params.label_path
        
        self.params      = params
        self.image_zip   = extract_zip(params.imageZip)
        self.hedZip      = extract_zip(params.hedEdgeZip)
        self.indexToPath = self.generate_path(train)
        self.att         = self.get_annotations()
        self.train       = train
        self.aug = A.Compose({
        A.RandomSizedCrop(min_max_height = (int(self.params.img_height * 0.8),self.params.img_height),height = self.params.img_height,width = self.params.img_width, p = 0.5),
        A.HorizontalFlip(p=0.5)
        })
    
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
        lines = [line.rstrip() for line in open(label_path, "r")]
        self.label_names = lines[1].split()
        for _, line in enumerate(lines[2:]):
            filename, *values = line.split()
            labels = []
            for attr in selected_attrs:
                idx = self.label_names.index(attr)
                labels.append((1 if (values[idx] == "1") else 0))
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
        Return, sketch,img,label
        Parameters
        ----------
        index: int, the index of the file need to be read
        
        Returns
        -------
        sketch : pytorch float tensor, input sketch
        img    : pytorch float tensor, the ground truth image corresponds to sketch.
        labels : pytorch float tensor, the attributes of the img
        """ 
        
        #get path for image and sketch
        
        image_path, sketch_path = self.indexToPath[index]
        
        #read image  into numpy array
        img = read_image_from_zip(self.image_zip,image_path,self.params.img_height,self.params.img_width)
        img  = np.array(img)
        
        #read sketch into numpy array
        hed_edge    = read_image_from_zip(self.hedZip,self.params.hedEdgePath + "/" + sketch_path.replace(".png",".jpg"),self.params.img_height,self.params.img_width)
        sketch      = np.array(hed_edge)
        
        #augment sketch and image if in the training mode
        if self.train:
            augmented = self.aug(image = img,mask = sketch)
            img = augmented['image']
            sketch = augmented['mask']

        img     = torch.FloatTensor(img).permute(2,0,1)
        sketch  = torch.FloatTensor(sketch).unsqueeze(2).permute(2,0,1)
        
        #read labels into pytorch float tensor.
        
        label = self.att[sketch_path]
        label = torch.FloatTensor(np.array(label))
        
        return sketch,img,label
        
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