#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is used to process the data of S2FGAN.
This module structure is following,
    - A forloop to combine mask for diffirent components of a image.
    - A forloop to extract the hed edge of images.
"""

import os
import cv2
from io import BytesIO
from PIL import Image
import numpy as np
import zipfile
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--size",     type=int, default=128,   help= "The size of images to save")
parser.add_argument("--img_num",  type=int, default=10, help= "Number of images in total")
parser.add_argument("--inputZip", type=str, default= "../data/CelebAMask-HQ-Sample.zip", help="The input zip")
parser.add_argument("--imagePath",type=str, default= "CelebAMask-HQ-Sample/CelebA-HQ-img", help="path of image for input zip")
parser.add_argument("--maskPath",type=str, default= "CelebAMask-HQ-Sample/CelebAMask-HQ-mask-anno", help="path of mask for input zip")
parser.add_argument("--edge_detector",type=str, default= "edge_detector", help="Path of pretrained HED edge detector")
opt = parser.parse_args()

print(opt)

label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

size = opt.size
img_num = opt.img_num
folder_base = opt.inputZip

#The folder to save the combined foor loop.
folder_save = f'CelebAMaskHQ-mask_{size}.zip'

try:
    os.remove(f"{folder_save}.zip", )
except:
    pass

#open input zip, and intialize outputzip.
output_zip = zipfile.ZipFile(folder_save,mode = "w")
input_zip  = zipfile.ZipFile(folder_base)

for k in range(img_num):
    folder_num = k // 2000
    im_base = np.zeros((size, size))
    for idx, label in enumerate(label_list):
        filename = os.path.join(opt.maskPath, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
        try:
            im = np.array(Image.open(BytesIO(input_zip.read(filename))).resize((size,size)))
            im = im[:, :, 0]
            im_base[im != 0] = (idx + 1)
        except:
            pass
        
    filename_save = str(k)+".png"
    img_file = BytesIO()
    Image.fromarray(im_base).convert('RGB').save(img_file,"PNG")
    output_zip.writestr(filename_save,img_file.getvalue())
    

output_zip.close()
input_zip.close()

#The final layer used for HED edge detector.
class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]
    
#read the caffe model of HED edge detor.     
class HED:
    def __init__(self,path,height,width):
        
        protoPath = os.path.sep.join([path,"deploy.prototxt"])
        modelPath = os.path.sep.join([path,"hed_pretrained_bsds.caffemodel"])
        net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        cv2.dnn_registerLayer("Crop", CropLayer)
        
        self.net = net
        self.h   = height
        self.w   = width
        
    def render(self,img):
        
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(self.w,self.h),
                                     mean=(104.00698793, 116.66876762, 122.67891434),
                                     swapRB=False, crop=False)
        self.net.setInput(blob)
        hed = self.net.forward()
        hed = cv2.resize(hed[0, 0], (self.w, self.h)) 
        hed = 255- (hed*255).astype("uint8")

        return cv2.cvtColor(hed, cv2.COLOR_GRAY2RGB)

#open input zip, and intialise output zip for edge images.
input_zip  = zipfile.ZipFile(folder_base)
output_zip = zipfile.ZipFile(f"hed_edge_{size}.zip",mode = "w")

#intialize hed edge detector.
hed = HED(opt.edge_detector,size,size)

for k in range(img_num):
    
    path = os.path.join(opt.imagePath,f"{k}.jpg")
    img  = np.array(Image.open(BytesIO(input_zip.read(path))).resize((size,size)))[...,::-1]
    img  = hed.render(img)
    filename_save = str(k)+".png"
    img_file = BytesIO()
    Image.fromarray(img).convert('L').save(img_file,"PNG")
    output_zip.writestr(filename_save,img_file.getvalue())

input_zip.close()
output_zip.close()






    
    
    
    
    