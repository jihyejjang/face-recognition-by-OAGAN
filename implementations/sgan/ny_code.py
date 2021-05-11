import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.datasets import MNIST
from torchsummary import summary

cuda = True if torch.cuda.is_available() else False
print("cuda: ", cuda)



class FaceCompletion(nn.Module):
    def __init__(self):
        super(FaceCompletion, self).__init__()

        # 1) [conv + IN + ReLU] * 3
        # (64, 128, 128) -> (512, 16, 16)
        # instance norm에 1d를 써야하나, 2d를 써야하나? -> 1d로 하면 에러남
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU()
        )

        # 2) [deconv + IN + ReLU] * 3
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )

        # 3) conv + tanh
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 3, 7, stride=1, padding=3),
            nn.Tanh()
        )


    def forward(self, x):
        out=self.block1(x)
        print("1st feature map:", out.shape)

        out=self.block2(out)
        print("2nd feature map:", out.shape)

        out=self.block3(out)
        print("3rd feature map:", out.shape)
        # out=out.view(out.size(0), -1)
        # out=self.fc_layer(out)
        # print("fc layer shape:", out.shape)
        return out

# main()
model = FaceCompletion()

# mask : predicted occlusion mask, (1, 128, 128)
# oa_feature : OA 모듈의 output, (64, 128, 128)

mask = torch.rand(1, 128, 128)
oa_feature = torch.rand(64, 128, 128)
input_img = torch.matmul(mask, oa_feature)

print("mask shape: ", mask.size())
print("oa_feature shape: ", oa_feature.size())
print("input_img shape: ", input_img.size())  # (64, 128, 128)
print("=================================================================")

summary(model, (64, 128, 128))

