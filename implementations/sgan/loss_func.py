import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import cv2
from torchvision import transforms

import torchvision.models as models
import PIL.Image as pilimg
import numpy as np

def feature_extract(img_p):
    feature_map=[]
    # Read image
    img = pilimg.open(img_p)

    # Display image
    img.show()

    # Fetch image pixel data to numpy array
    im = np.array(img)
    im = Variable(torch.from_numpy(im).unsqueeze(0)).float()

    #vgg16 = models.vgg16()
    vgg16 = models.vgg16(pretrained=True)
    vgg16.classifier=vgg16.classifier[:-1]
    feature_map.append(vgg16.classifier(im).data.data)
    vgg16.classifier=vgg16.classifier[:-1]
    feature_map.append(vgg16.classifier(im).data.data)
    vgg16.classifier=vgg16.classifier[:-1]
    feature_map.append(vgg16.classifier(im).data.data)

    print(feature_map[0].shape)
    print(feature_map[1].shape)
    print(feature_map[2].shape)
    return feature_map

img1_vgg = feature_extract('./eximg1.jpeg')
img2_vgg = feature_extract('./eximg2.png')
img3_vgg = feature_extract('./')

def perceptual_loss(x_synth_vgg,x_gt_vgg,x_final_vgg):
    loss=0
    for n in len(x_synth_vgg): #3 layer
        for i in len(x_synth_vgg[n]):
            loss += abs(x_synth_vgg[n][i]-x_gt_vgg[n][i])
            loss += abs(x_final_vgg[n][i]-x_gt_vgg[n][i])
    return loss

perceptualLoss = perceptual_loss(img1_vgg,img2_vgg)

def style_loss(x_synth_vgg,x_gt_vgg,x_final_vgg):
    loss=0
    Kn1=1/(3*1*4096)
    Kn2=1/(3*1*1000)
    for n in len(x_synth_vgg):
        print("n이 4096이면 kn1, 1000이면 kn2 n:",len(x_synth_vgg[n]))
        if len(x_synth_vgg[n])==1000:
            for i in len(x_synth_vgg[n]):
                loss += abs(Kn2*(x_synth_vgg[n].T*x_synth_vgg[n] - x_gt_vgg[n].T*x_gt_vgg[n]))
                loss += abs(Kn2*(x_final_vgg[n].T*x_final_vgg[n] - x_final_vgg[n].T*x_final_vgg[n]))
        elif len(x_synth_vgg[n]==4096):
            for i in len(x_synth_vgg[n]):
                loss += abs(Kn1*(x_synth_vgg[n].T*x_synth_vgg[n] - x_gt_vgg[n].T*x_gt_vgg[n]))
                loss += abs(Kn1*(x_final_vgg[n].T*x_final_vgg[n] - x_final_vgg[n].T*x_final_vgg[n]))

