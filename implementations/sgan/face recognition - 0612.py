#!/usr/bin/env python
# coding: utf-8

# # 0. Show Image

# In[9]:


import cv2
from skimage.io import imread
# file_dir = '../face-alignment-master/test/assets/aflw-test.jpg'
file_dir = 'PRLAB/16_나혜/VID_20181012_144116.mp4_93536778.png'

input =io.imread(file_dir)
preds = fa.get_landmarks(input)

img = cv2.imread(file_dir)

for pos in preds[0]:
    x = pos[0]
    y = pos[1]    
    
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # 1. Landmark

# In[6]:


pip install -r requirements.txt


# In[5]:


import face_alignment
from skimage import io

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)


# In[2]:


input = io.imread('../face-alignment-master/test/assets/aflw-test.jpg')
preds = fa.get_landmarks(input)


# In[32]:


input.shape


# In[3]:


print(type(preds[0][0]))


# In[4]:


preds[0].shape


# In[5]:


preds[0]


# In[6]:


preds[0][67] # preds[image index][landmark index][0: x axis, 1: y axis]


# In[11]:


preds[0][67][1] # y_pos


# In[3]:


import numpy as np
import cv2

img = cv2.imread('../face-alignment-master/test/assets/aflw-test.jpg')

for pos in preds[0]:
    x = pos[0]
    y = pos[1]
    img = cv2.line(img, (x, y), (x, y), (0, 0, 255), 3)

print(img.shape)
print(img)
    
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[31]:


img[2].shape


# In[20]:


len(img[0][0])


# # 2-1. Bounding Box - Full

# In[35]:


cv2.imshow('image', input)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[57]:


input[449][449][0] # input[Height][Width][channel]


# In[66]:


print(len(input))


# In[65]:


print(len(input[0]))


# In[64]:


print(len(input[0][0]))


# In[8]:


x_max = 0
x_min = len(input)
y_max = 0
y_min = len(input[0])

print(x_min, y_min)
print(x_max, y_max)


# In[72]:


len(preds[0]) # point수 68개


# In[74]:


len(preds[0][0]) # x, y 좌표 → 2개


# In[77]:


preds[0][0][0] # 0번 index 이미지에서 0번째 랜드마크의 x좌표


# In[5]:


# preds[image_number][landmark_number][0: x axis, 1: y axis]

n = 0 # image_number → 여러장에 박스칠때를 생각해서 만든 변수

x_max = 0
x_min = len(input)
y_max = 0
y_min = len(input[0])

for i in range(0, len(preds[0])):
    if (x_max < preds[n][i][0]):
        x_max = preds[n][i][0]
    if (x_min > preds[n][i][0]):
        x_min = preds[n][i][0]
    if (y_max < preds[n][i][1]):
        y_max = preds[n][i][1]
    if (y_min > preds[n][i][1]):
        y_min = preds[n][i][1]
        
# print(x_min, y_min)
# print(x_max, y_max)


# In[61]:


import numpy as np
import cv2

img = cv2.imread('../face-alignment-master/test/assets/aflw-test.jpg')


for pos in preds[0]:
    x = pos[0]
    y = pos[1]
    img = cv2.line(img, (x, y), (x, y), (0, 255, 0), 3)
    img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)

print(x_min, y_min)
print(x_max, y_max)
    
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # 2-2. Bounding Box - Region1

# In[26]:


import face_alignment
from skimage import io

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)


# In[53]:


file_dir = '../face-alignment-master/test/assets/aflw-test.jpg'
# file_dir = '../face-alignment-master/test/assets/alfw-test_ex (7).jpg'

input = io.imread(file_dir)
preds = fa.get_landmarks(input)


# In[100]:


print(preds[0][36]) # point1 - right point of right eye
print(preds[0][39]) # point2 - left point of right eye
print(preds[0][42]) # point3 - right point of left eye
print(preds[0][45]) # point4 - left point of left eye
print(preds[0][30]) # point5 - point of nose
print(y_max) # point7


# In[58]:


# preds[image_number][landmark_number][0: x axis, 1: y axis]

n = 0 # image_number → 여러장에 박스칠때를 생각해서 만든 변수

x_max = 0
x_min = len(input)
y_max = 0
y_min = len(input[0])

for i in range(0, len(preds[0])):
    if (x_max < preds[n][i][0]):
        x_max = preds[n][i][0]
    if (x_min > preds[n][i][0]):
        x_min = preds[n][i][0]
    if (y_max < preds[n][i][1]):
        y_max = preds[n][i][1]
    if (y_min > preds[n][i][1]):
        y_min = preds[n][i][1]
        
# print(x_min, y_min)
# print(x_max, y_max)


# In[65]:


import numpy as np
import cv2

img = cv2.imread(file_dir)

for pos in preds[0]:
    x = pos[0]
    y = pos[1]
    img = cv2.line(img, (x, y), (x, y), (0, 255, 0), 3) # B, G, R    
    img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
    img = cv2.rectangle(img, (x_min, y_min), (x_max, face_point5[1]), (0, 255, 255), 3)
    img = cv2.line(img, (face_point5[0], face_point5[1]), (face_point5[0], face_point5[1]), (0, 255, 0), 3)

print(x_min, y_min)
print(x_max, y_max)
    
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[64]:


print(len(preds[0]))
print(preds[0])


# # 2-3. Bounding Box - Region2

# In[1]:


import face_alignment
from skimage import io

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)


# In[2]:


# file_dir = '../face-alignment-master/test/assets/aflw-test.jpg'
file_dir = '../face-alignment-master/test/assets/alfw-test_ex (8).jpg'

input = io.imread(file_dir)
preds = fa.get_landmarks(input)


# In[3]:


# preds[image_number][landmark_number][0: x axis, 1: y axis]

n = 0 # image_number → 여러장에 박스칠때를 생각해서 만든 변수

x_max = 0
x_min = len(input)
y_max = 0
y_min = len(input[0])

for i in range(0, len(preds[0])):
    if (x_max < preds[n][i][0]):
        x_max = preds[n][i][0]
    if (x_min > preds[n][i][0]):
        x_min = preds[n][i][0]
    if (y_max < preds[n][i][1]):
        y_max = preds[n][i][1]
    if (y_min > preds[n][i][1]):
        y_min = preds[n][i][1]
        
# print(x_min, y_min)
# print(x_max, y_max)


# In[4]:


print(preds[0][36]) # point1 - right point of right eye
print(preds[0][39]) # point2 - left point of right eye
print(preds[0][42]) # point3 - right point of left eye
print(preds[0][45]) # point4 - left point of left eye
print(preds[0][30]) # point5 - point of nose
print(y_max) # point7


# In[5]:


import numpy as np

#face_point[point_idx][x, y]
face_point= np.zeros((5,2))
print(face_point.shape)

face_point[0][0] = preds[0][36][0] # point1 - right point of right eye
face_point[0][1] = preds[0][36][1]
face_point[1][0] = preds[0][39][0] # point2 - left point of right eye
face_point[1][1] = preds[0][39][1] 
face_point[2][0] = preds[0][42][0] # point3 - right point of left eye
face_point[2][1] = preds[0][42][1]
face_point[3][0] = preds[0][45][0] # point4 - left point of left eye
face_point[3][1] = preds[0][45][1]
face_point[4][0] = preds[0][30][0] # point5 - point of nose
face_point[4][1] = preds[0][30][1]

print(face_point)


# In[6]:


eye_y = int(0) # point 1, 2, 3, 4의 y좌표 평균(eye line)
for i in range(0, 4):
    eye_y = eye_y + face_point[i][1]
eye_y = eye_y/4.0
print(eye_y)


# In[7]:


face_point[4][1]


# In[8]:


dist = eye_y - abs(face_point[4][1] - eye_y) # vertical eye-to-nose distance

print(y_min, dist)
if(y_min > dist): # 부호를 반대로하면 눈썹 부분이 사라질 수 있음
    y_min = dist
print(y_min)


# In[9]:


print(type(x_min))
print(type(y_min))
print(type(x_max))
print(type(y_max))

print(type(face_point[4][1]))


# In[51]:


import cv2

img = cv2.imread(file_dir)

for pos in preds[0]:
    x = pos[0]
    y = pos[1]
    img = cv2.line(img, (x, y), (x, y), (0, 255, 0), 3) # (B, G, R)
    
    img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(face_point[4][1])), (25, 175, 0), 3)
    
    img = cv2.line(img, (int(face_point[0][0]), int(face_point[0][1])), (int(face_point[0][0]), int(face_point[0][1])), (0, 0, 255), 3)
    img = cv2.line(img, (int(face_point[1][0]), int(face_point[1][1])), (int(face_point[1][0]), int(face_point[1][1])), (0, 0, 255), 3)
    img = cv2.line(img, (int(face_point[2][0]), int(face_point[2][1])), (int(face_point[2][0]), int(face_point[2][1])), (0, 0, 255), 3)
    img = cv2.line(img, (int(face_point[3][0]), int(face_point[3][1])), (int(face_point[3][0]), int(face_point[3][1])), (0, 0, 255), 3)
    img = cv2.line(img, (int(face_point[4][0]), int(face_point[4][1])), (int(face_point[4][0]), int(face_point[4][1])), (0, 0, 255), 3)

print(x_min, y_min)
print(x_max, y_max)
    
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # 2-3. Crop

# In[119]:


import cv2

img = cv2.imread(file_dir)

roi_image = img.copy() 
roi_image = img[int(y_min):int(face_point[4][1]+1), int(x_min):int(x_max+1)]

cv2.imshow("src", img)
cv2.imshow("dst", roi_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # 2-4. Resize

# In[121]:


import cv2

resized = cv2.resize(roi_image, dsize=(91, 64), interpolation=cv2.INTER_AREA) # x, y

print(img.shape)
print(roi_image.shape)
print(resized.shape)

cv2.imshow("src", roi_image)
cv2.imshow("dst", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()


# # 2-5. Save(numpy array → jpg)

# In[125]:


print(file_dir)
print(file_dir[:-4])

save_dir = file_dir[:-4] + '_resized.jpg'
print(save_dir)


# In[128]:


cv2.imwrite(save_dir, resized)


# In[ ]:




