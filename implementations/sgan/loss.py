import numpy as np
# from sgan_main import *
import torch
# from gan_model import *
import torch.nn as nn
import torchvision.models as models



class sganloss():

    def __init__(self):
        self.img = torch.randn([1, 3, 128, 128])
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16_features = [self.vgg16.features[:1],
                            self.vgg16.features[:3],
                            self.vgg16.features[:6],
                            self.vgg16.features[:8],
                            self.vgg16.features[:11],
                            self.vgg16.features[:13],
                            self.vgg16.features[:15],
                            self.vgg16.features[:18],
                            self.vgg16.features[:20],
                            self.vgg16.features[:22],
                            self.vgg16.features[:25],
                            self.vgg16.features[:27],
                            self.vgg16.features[:29]]

    def pixel_loss(self, final, gt, inverse_M, M, alpa, beta):
        one = torch.matmul(inverse_M, (final - gt))
        one = torch.norm(one, 1)
        one = np.dot(alpa, one.detach())

        two = torch.matmul(M, (final - gt))
        two = torch.norm(two, 1)
        two = np.dot(beta, two.detach())

        return one + two

#TODO: smooth loss 수정해야됨 (지혜)
    def smooth_loss(self, img, final, M):
        # x_final
        width = img.size(2)
        height = img.size(3)
        
        #f_wid_1_c1b1 = np.array(final)[0][0][:][1:]
        #f_wid_0_c1b1 = np.array(final)[0][0][:][:-1]
        #a1 = nn.L1Loss(f_wid_1_c1b1-f_wid_9_c1b1,
        pixel=[] #비교 대상 픽셀
        near=[]  #비교 대상의 인접픽셀
        for batch in range(len(final)):
            for channel in range(len(final[0])):
                near.append(final[batch][channel][:][1:])
                pixel.append(final[batch][channel][:][:-1]) #맞나?

        
#         first_np_final = np.array(final)[0][0][:][1:]  # 128,128
#         second_np_final = np.array(final)[0][1]
#         third_np_final = np.array(final)[0][2]
        
#         np_M = np.array(M.detach())
        
#         fin1 = 0
#         fin2 = 0
#         fin3 = 0
#         finM = 0
        
        
        
#         for i in range(0, width):
#             for j in range(0, height):
#                 a1 = first_np_final[:][:][i][j + 1] - first_np_final[:][:][i][j]
#                 a1_ = nn.L1Loss(a1, reduction='sum')
#                 a1__ = torch.norm(a1,1)
#                 print("a1_",a1_)
#                 print("a1__",a1__)
#                 b1 = first_np_final[i + 1][j] - first_np_final[i][j]
#                 b1 = nn.L1Loss(b1, reduction='sum')
#                 fin1 = fin1 + a1 + b1

#                 a2 = second_np_final[i][j + 1] - second_np_final[i][j]
#                 a2 = nn.L1Loss(a2, reduction='sum')
#                 b2 = second_np_final[i + 1][j] - second_np_final[i][j]
#                 b2 = nn.L1Loss(b2, reduction='sum')
#                 fin2 = fin2 + a2 + b2

#                 a3 = third_np_final[i][j + 1] - third_np_final[i][j]
#                 a3 = nn.L1Loss(a3, reduction='sum')
#                 b3 = third_np_final[i + 1][j] - third_np_final[i][j]
#                 b3 = nn.L1Loss(b3, redcution='sum')
#                 fin3 = fin3 + a3 + b3

#                 c = np_M[i][j + 1] - np_M[i][j]
#                 c = nn.L1Loss(c, reduction='sum')
#                 d = np_M[i + 1][j] - np_M[i][j]
#                 d = nn.L1Loss(d, reduction='sum')
#                 finM = finM + c + d

#         return fin1 + fin2 + fin3 + finM

    # print(len(vgg16_features)) # 13

    def perceptual_loss(self, synth, final, gt):
        first = 0
        second = 0
        for i in range(len(self.vgg16_features)):
            a = torch.norm(self.vgg16_features[i](synth) - self.vgg16_features[i](gt), 1)
            first = first + a

        for j in range(len(self.vgg16_features)):
            b = torch.norm(self.vgg16_features[j](final) - self.vgg16_features[j](gt), 1)
            second = second + b

        return first + second

    # print(vgg16_features[0](img).size(2))
    def style_loss(self,synth, final, gt):
        first = 0
        second = 0
        
        for i in range(len(self.vgg16_features)):
            kn = 1 / (self.vgg16_features[i](synth).size(1) * self.vgg16_features[i](synth).size(2) * self.vgg16_features[i](synth).size(3))
            
            s = self.vgg16_features[i](synth)
            sT = np.transpose(s.detach(),(0,1,3,2))
#             print("s",s.shape)
#             print("s",np.transpose(s.detach(),(0,1,3,2)).shape)
            
            
            g = self.vgg16_features[i](gt)
            gT = np.transpose(g.detach(),(0,1,3,2))
#             print("g",g.shape)
#             print("g",np.transpose(g.detach(),(0,1,3,2)).shape)
            
            a = torch.matmul(sT, s) - torch.matmul(gT, g)
            
            
            a = a * kn
            # a = nn.L1Loss(a)
            a = torch.norm(a, 1)
            first = first + a

        for j in range(len(self.vgg16_features)):
            kn = 1 / (self.vgg16_features[j](synth).size(1) * self.vgg16_features[j](synth).size(2) * self.vgg16_features[j](synth).size(3))
            
            f=self.vgg16_features[j](final)
            #print("f",f.shape)
            fT=np.transpose(f.detach(),(0,1,3,2))
            #print("fT",fT.shape)
            g = self.vgg16_features[j](gt)
            gT = np.transpose(g.detach(),(0,1,3,2))
            #print("g",g.shape)
            #print("gT",gT.shape)
            
            b = torch.matmul(fT,f) - torch.matmul(gT,g)
            
            b = b * kn
            # b = nn.L1Loss(b)
            b = torch.norm(b, 1)
            second = second + b

        return first + second

    def l2_norm(self,x):
        norm = x.norm(p=2, dim=1, keepdim=True)
        x_normalized = x.div(norm.expand_as(x))
        return x_normalized



# # # vgg16 = vgg16(img).detach()
# #
# # # vgg16.features_1=vgg16.features[:1]
# # # vgg16.features_2=vgg16.features[:3]
# # # vgg16.features_3=vgg16.features[:6]
# # # vgg16.features_4=vgg16.features[:8]
# # # vgg16.features_5=vgg16.features[:11]
# # # vgg16.features_6=vgg16.features[:13]
# # # vgg16.features_7=vgg16.features[:15]
# # # vgg16.features_8=vgg16.features[:18]
# # # vgg16.features_9=vgg16.features[:20]
# # # vgg16.features_10=vgg16.features[:22]
# # # vgg16.features_11=vgg16.features[:25]
# # # vgg16.features_12=vgg16.features[:27]
# # # vgg16.features_13=vgg16.features[:29] # [1,512,8,8]
# # #
#
# #
# # for i in range(len(vgg16_features)):
# # print(vgg16_features[3](img).shape)
# # aa = torch.randn([1,3,128,128])
# # print('v', vgg16_features[3](aa).shape)
#
# def pixel_loss(final, gt, inverse_M, M, alpa, beta):
#     one = torch.matmul(inverse_M, (final-gt))
#     one = nn.L1Loss(one, reduction = 'sum')
#     one = np.dot(alpa, one)
#
#     two = torch.matmul(M, (final-gt))
#     two = nn.L1Loss(two, reduction = 'sum')
#     two = np.dot(beta, two)
#
#     return one + two
#
# def smooth_loss(img, final, M):
#     # x_final
#     width = img.size(2)
#     height = img.size(3)
#     first_np_final = np.array(final)[0][0] # 128,128
#     second_np_final = np.array(final)[0][1]
#     third_np_final = np.array(final)[0][2]
#
#     np_M = np.array(M)
#     fin1, fin2, fin3, finM = 0
#     for i in range(0, width):
#         for j in range(0, height):
#             a1 = first_np_final[i][j+1] - first_np_final[i][j]
#             a1 = nn.L1Loss(a1, reduction = sum)
#             b1 = first_np_final[i+1][j] - first_np_final[i][j]
#             b1= nn.L1Loss(b1, reduction = sum)
#             fin1 = fin1 + a1 + b1
#
#             a2 = second_np_final[i][j + 1] - second_np_final[i][j]
#             a2 = nn.L1Loss(a2, reduction = 'sum')
#             b2 = second_np_final[i + 1][j] - second_np_final[i][j]
#             b2 = nn.L1Loss(b2, reduction = 'sum')
#             fin2 = fin2 + a2 + b2
#
#             a3 = third_np_final[i][j + 1] - third_np_final[i][j]
#             a3 = nn.L1Loss(a3, reduction= 'sum')
#             b3 = third_np_final[i + 1][j] - third_np_final[i][j]
#             b3 = nn.L1Loss(b3, redcution = 'sum')
#             fin3 = fin3 + a3 + b3
#
#             c = np_M[i][j + 1] - np_M[i][j]
#             c = nn.L1Loss(c, reduction= 'sum')
#             d = np_M[i + 1][j] - np_M[i][j]
#             d = nn.L1Loss(d, reduction ='sum')
#             finM = finM + c + d
#
#     return fin1 + fin2 + fin3 + finM
# # print(len(vgg16_features)) # 13
#
# def perceptual_loss(synth, final, vgg16_features, gt):
#     first = 0
#     second = 0
#     for i in range(len(vgg16_features)):
#         a = torch.norm(vgg16_features[i](synth) - vgg16_features[i](gt), 1)
#         first = first + a
#
#     for j in range(len(vgg16_features)):
#         b = torch.norm(vgg16_features[j](final) - vgg16_features[j](gt), 1)
#         second = second + b
#
#     return first + second
#
# # print(vgg16_features[0](img).size(2))
# def style_loss(synth, final, vgg16_features, gt):
#     first = 0
#     second = 0
#     kn = 1 / (vgg16_features.size(1) * vgg16_features.size(2) * vgg16_features.size(3))
#     for i in range(len(vgg16_features)):
#         a = torch.matmul(vgg16_features[i](synth.transpose(0,1)), vgg16_features[i](synth) - vgg16_features[i](gt.transpose(0,1)), vgg16_features[i](gt))
#         a = a * kn
#         # a = nn.L1Loss(a)
#         a = torch.norm(a, 1)
#         first = first + a
#
#     for j in range(len(vgg16_features)):
#         b = torch.matmul(vgg16_features[j](final.transpose(0,1)), vgg16_features[j](final) - vgg16_features[j](gt.transpose(0,1)), vgg16_features[j](gt))
#         b = b * kn
#         # b = nn.L1Loss(b)
#         b = torch.norm(b, 1)
#         second = second + b
#
#     return first + second
#
# def l2_norm(x):
#     norm = x.norm(p=2, dim=1, keepdim = True)
#     x_normalized = x.div(norm.expand_as(x))
#     return x_normalized
