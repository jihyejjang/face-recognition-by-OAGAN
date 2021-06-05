# import numpy as np
# from gan_model import *
#
# img = torch.randn([1,3,128,128])
# vgg16 = models.vgg16(pretrained=True)
# # vgg16 = vgg16(img).detach()
#
# vgg16.features_1=vgg16.features[:1]
# vgg16.features_2=vgg16.features[:3]
# vgg16.features_3=vgg16.features[:6]
# vgg16.features_4=vgg16.features[:8]
# vgg16.features_5=vgg16.features[:11]
# vgg16.features_6=vgg16.features[:13]
# vgg16.features_7=vgg16.features[:15]
# vgg16.features_8=vgg16.features[:18]
# vgg16.features_9=vgg16.features[:20]
# vgg16.features_10=vgg16.features[:22]
# vgg16.features_11=vgg16.features[:25]
# vgg16.features_12=vgg16.features[:27]
# vgg16.features_13=vgg16.features[:29] # [1,512,8,8]
#
# vgg16_features = [vgg16.features[:1],
# vgg16.features[:3],
# vgg16.features[:6],
# vgg16.features[:8],
# vgg16.features[:11],
# vgg16.features[:13],
# vgg16.features[:15],
# vgg16.features[:18],
# vgg16.features[:20],
# vgg16.features[:22],
# vgg16.features[:25],
# vgg16.features[:27],
# vgg16.features[:29]]
# def pixel_loss(final, gt, alpa, beta):
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
# def smooth_loss(img, final):
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
#     first, second = 0
#     for i in range(len(vgg16_features)):
#         a = vgg16_features[i](synth) - vgg16_features[i](gt)
#         a = nn.L1Loss(first)
#         first = first + a
#
#     for j in range(len(vgg16_features)):
#         b = vgg16_features[j](final) - vgg16_features[j](gt)
#         b = nn.L1Loss(second)
#         second = second + b
#
#     return first + second
#
# # print(vgg16_features[0](img).size(2))
# def style_loss(synth, final, vgg16_features, gt):
#     first, second = 0
#     kn = 1 / (vgg16_features.size(1) * vgg16_features.size(2) * vgg16_features.size(3))
#     for i in range(len(vgg16_features)):
#         a = torch.matmul(vgg16_features[i](synth.transpose(0,1)), vgg16_features[i](synth) - vgg16_features[i](gt.transpose(0,1)), vgg16_features[i](gt))
#         a = a * kn
#         a = nn.L1Loss(a)
#         first = first + a
#
#     for j in range(len(vgg16_features)):
#         b = torch.matmul(vgg16_features[j](final.transpose(0,1)), vgg16_features[j](final) - vgg16_features[j](gt.transpose(0,1)), vgg16_features[j](gt))
#         b = b * kn
#         b = nn.L1Loss(b)
#         second = second + b
#
#     return first + second
import numpy as np
# from sgan_main import *
import torch
# from gan_model import *
# import torch.nn as nn
import torchvision.models as models
img = torch.randn([1,3,128,128])
vgg16 = models.vgg16(pretrained=True)
# # vgg16 = vgg16(img).detach()
#
# # vgg16.features_1=vgg16.features[:1]
# # vgg16.features_2=vgg16.features[:3]
# # vgg16.features_3=vgg16.features[:6]
# # vgg16.features_4=vgg16.features[:8]
# # vgg16.features_5=vgg16.features[:11]
# # vgg16.features_6=vgg16.features[:13]
# # vgg16.features_7=vgg16.features[:15]
# # vgg16.features_8=vgg16.features[:18]
# # vgg16.features_9=vgg16.features[:20]
# # vgg16.features_10=vgg16.features[:22]
# # vgg16.features_11=vgg16.features[:25]
# # vgg16.features_12=vgg16.features[:27]
# # vgg16.features_13=vgg16.features[:29] # [1,512,8,8]
#
vgg16_features = [vgg16.features[:1],
vgg16.features[:3],
vgg16.features[:6],
vgg16.features[:8],
vgg16.features[:11],
vgg16.features[:13],
vgg16.features[:15],
vgg16.features[:18],
vgg16.features[:20],
vgg16.features[:22],
vgg16.features[:25],
vgg16.features[:27],
vgg16.features[:29]]

# for i in range(len(vgg16_features)):
print(vgg16_features[3](img).shape)
# aa = torch.randn([1,3,128,128])
# print('v', vgg16_features[3](aa).shape)
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
