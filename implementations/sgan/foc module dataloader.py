#!/usr/bin/env python
# coding: utf-8

# ## pytorch custom dataset loader

# In[ ]:


#get_ipython().run_line_magic('matplotlib', 'inline')
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

class OAGandataset():
    def __init__(self):
        self.img_size=128
        self.dir="./data/real_data(resized)"
        folders = os.listdir(self.dir)  #폴더명 오름차순으로 정렬
        folders = [f for f in folders if not f.startswith('.')] #windows는 주석처리하고 쓰세용~
        folders = sorted(folders)
        file_names = [os.listdir(os.path.join(self.dir , f)) for f in folders]
        self.file_name = sum(file_names,[])#flatten
        self.label=[] #레이블 생성
        for f in range(len(file_names)):
            for i in range(len(file_names[f])):
                self.label.append(folders[f])

    def __len__(self): # folder 갯수 = 사람 수
        return len(self.file_name)

    def __getitem__(self,index):
        trans = transforms.Compose([transforms.Resize((self.img_size,self.img_size)),
                                    transforms.ToTensor()])
        #print (os.getcwd())

        person_dir = os.path.join(self.dir,self.label[index],self.file_name[index])
        print ("디렉토리",person_dir)
        image = Image.open(person_dir)
        image = trans(image)
        label = self.label[index]

        return image, label

OAGan_dataset = OAGandataset()

print ("train : 총 {} 장의 image".format(OAGan_dataset.__len__()))

def show(img,y,color=False): #미리보기
    npimg=img.numpy()
    npimg_tr=np.transpose(npimg,(1,2,0))
    plt.imshow(npimg_tr)

'''
train_dataloader = DataLoader(OAGan_dataset,
                        shuffle=True,
                        num_workers=0,
                        batch_size=3) #batch size 수정

dataiter = iter(train_dataloader)

example_batch = next(dataiter)

#concatenated = torch.cat(example_batch[0],0)

#imshow(torch.cat((example_batch[0],example_batch[1]),0))

show(example_batch[0][0],example_batch[1][0])


net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
counter = []
loss_history = []
iteration_number= 0
'''


# ### dataloader로 불러온 batch sample 보기

# In[ ]:


train_dataloader = DataLoader(OAGan_dataset,
                        shuffle=True,
                        num_workers=0,
                        batch_size=3) #batch size 수정

dataiter = iter(train_dataloader)

example_batch = next(dataiter)

#concatenated = torch.cat(example_batch[0],0)

#imshow(torch.cat((example_batch[0],example_batch[1]),0))

show(example_batch[0][0],example_batch[1][0])

