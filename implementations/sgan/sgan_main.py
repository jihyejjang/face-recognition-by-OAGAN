# TODO: ---우선적으로해야할것---dataloader,dataset선별

import argparse
import os
import numpy as np
from dataloader import *
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--num_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
# print(opt)

cuda = True if torch.cuda.is_available() else False



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class IdentityPadding(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(IdentityPadding, self).__init__()

        self.pooling = nn.MaxPool2d(1, stride=stride)
        self.add_channels = out_channels - in_channels

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
        out = self.pooling(out)
        return out

# TODO: 아직 3* 반영안됨, channel 다시 해줘야됨 -> 완료
# 코드 출처 : https://dnddnjs.github.io/cifar10/2018/10/09/resnet/
# https://github.com/eriklindernoren/PyTorch-GAN/blob/a163b82beff3d01688d8315a3fd39080400e7c01/implementations/srgan/models.py#L18
# 여길보니 residual block 할때 in, out channel이 동일함.
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sample=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        if down_sample:
            self.down_sample = IdentityPadding(in_channels, out_channels, stride)
        else:
            self.down_sample = None

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)

        out += shortcut
        out = self.relu(out)
        return out

# TODO: generator module 두개 한 class로 합치는 작업 -> 완료
# 참고링크: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cogan/cogan.py
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # TODO : 밑에 3줄이 의미하는 것 찾아 수정 or 삭제하기
        # self.label_emb = nn.Embedding(opt.num_classes, opt.latent_dim)

        # self.init_size = opt.img_size // 4  # Initial size before upsampling
        # self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.FaceOcclusion_1=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # -----
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            # -----
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            # -----
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU()
            # -----
        )
        self.FaceOcclusion_2=nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        )

        self.FaceCompletion=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            # -----
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # -----
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        # occlusion aware module
        out_predicted=self.FaceOcclusion_1(x)
        out_InvertedM=torch.ones(1, 1, 128, 128) - x
        out_predictedM=self.FaceOcclusion_2(out_predicted)
        out_oa=torch.matmul(out_predicted, out_predictedM)

        # face completion module
        out_synth=self.FaceCompletion(out_oa)
        out_fc=torch.matmul(out_InvertedM, out_synth)
        out_filter=torch.matmul(x, out_predictedM)
        out_final=out_filter + out_fc

        return out_final



# class FaceOcclusion(nn.Module):
#     def __init__(self):
#         super(FaceOcclusion, self).__init__()
#         # TODO : 밑에 3줄이 의미하는 것 찾아 수정 or 삭제하기
#         # self.label_emb = nn.Embedding(opt.num_classes, opt.latent_dim)
#         #
#         # self.init_size = opt.img_size // 4  # Initial size before upsampling
#         # self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))
#
#         self.block1 = nn.Sequential(
#             nn.Conv2d(3,64,kernel_size=7, stride=1, padding=3), # 왜 3인가?
#             nn.InstanceNorm2d(64),
#             nn.ReLU()
#         )
#         self.block2=nn.Sequential(
#             nn.Conv2d(64,128,kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(256),
#             nn.ReLU()
#         )
#         self.block3=nn.Sequential(
#             ResidualBlock(256, 256),
#             ResidualBlock(256, 256),
#             ResidualBlock(256, 256)
#         )
#         self.block4=nn.Sequential(
#             nn.ConvTranspose2d(256,128, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(128),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(64),
#             nn.ReLU()
#         )
#         self.block5=nn.Sequential(
#             nn.Conv2d(64,1, kernel_size=7, stride=1, padding=3),
#             nn.Sigmoid()
#         )
#
# class FaceCompletion(nn.Module):
#     def __init__(self):
#         super(FaceCompletion, self).__init__()
#         self.block1 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(512),
#             nn.ReLU()
#         )
#
#         self.block2 = nn.Sequential(
#             nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(256),
#             nn.ReLU(),
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(128),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(64),
#             nn.ReLU()
#         )
#
#         # 3) conv + tanh
#         self.block3 = nn.Sequential(
#             nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
#             nn.Tanh()
#         )
#
#
#     def forward(self, x):
#         out=self.block1(x)
#         # print("1st feature map:", out.shape)
#
#         out=self.block2(out)
#         # print("2nd feature map:", out.shape)
#
#         out=self.block3(out)
#         # print("3rd feature map:", out.shape)
#         # out=out.view(out.size(0), -1)
#         # out=self.fc_layer(out)
#         # print("fc layer shape:", out.shape)
#         return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        # Output layers
        # TODO: sgan 그대로 써도될지. 논문에는 conv(adv), conv(attr)임
        # https://github.com/znxlwm/pytorch-pix2pix/blob/3059f2af53324e77089bbcfc31279f01a38c40b8/network.py#L104- patch gan discriminator code
        # 소현 원본 코드
        # self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1),nn.Sigmoid())
        # self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.num_classes + 1), nn.Softmax()) # 우리는 이게 attribute가 아니라 face인거지
        # 논문에 나와있는 discriminator architecture 참고해 수정함
        self.adv_layer = nn.Sequential(nn.Conv2d(128 * ds_size ** 2, 1, kernel_size=3, stride=1, padding=1),
                                       nn.Sigmoid()
        )
        self.attr_layer = nn.Sequential(nn.Conv2d(128 * ds_size ** 2, opt.num_classes, kernel_size=2, stride=1, padding=0),
                                        nn.Softmax())  # attribute classification대신 얼굴 인식 수행

    def forward(self, x):
        out = self.discriminator_block(x)
        #out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.attr_layer(out)

        return validity, label

# Loss functions - TODO: 지혜,승건 수정 부분
# loss 합치는거 그냥 sum of scala vector*loss 로 하면될듯?
# 참고링크: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cogan/cogan.py 210줄
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

### discriminator에 사용되는 attribute loss !!!미완성!!!
# attribute loss는 정답 이미지~복원한 이미지 각각 attribute의 mse로, paired image에만 사용됨
# 복원된 이미지는 Generator의 return인 out_final에 해당함 (아래 코드의 generator에 해당하는듯?)
# 정답 이미지는 dataloader에서 불러와야 할듯?
# attribute_loss = nn.MSELoss(input, target)  # 순서대로 정답 이미지, 복원한 이미지


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()
    # attribute_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# data loader
# TODO: 지혜 dataloader.py 여기에 써주세요~
# TODO: py파일 불러오기로쓸거면 return 만들어주세요
# 밑에 오류는 언니가 dataloader만들면 없어질거임.

paired_dataset = OAGandataset(paired=True, folder_numbering=False)
#unpaired_dataset = OAGandataset(unpaired=True, folder_numbering=False)

train_dataloader_p = DataLoader(paired_dataset,
                                shuffle=True,
                                num_workers=0,
                                batch_size= opt.batch_size)
# #train_dataloader_up = DataLoader(unpaired_dataset,
#                             shuffle=True,
#                             num_workers=0,
#                             batch_size=30)

# Optimizers
# TODO: 10-4가 0.0001맞나? -> 1E-4는 1*10^(-4)와 동일함
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
#  TODO: alternating training 보고 디자인하기
# ----------

#나도 TODO넣고싶은데 어케함???????
#paired image training (unpaired도 따로 만들고, loss도 상황에 따라 적용)
for epoch in range(opt.n_epochs):
    for i, (imgs, imgs_gt, labels) in enumerate(train_dataloader_p):
        #print(imgs.shape)
        #print(imgs_gt.shape)
        #print(labels.shape)
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        fake_aux_gt = Variable(LongTensor(batch_size).fill_(opt.num_classes), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        #z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        #print(z.shape)

        # Generate a batch of images
        gen_imgs = generator(real_imgs)
        print(gen_imgs.shape)

        # Loss measures generator's ability to fool the discriminator
        validity, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, fake_aux_gt)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), fake_aux_gt.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
