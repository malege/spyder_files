# -*- coding: utf-8 -*-

# =============================================================================
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# img=cv2.imread('D:/pictures/bm.jpg')
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# plt.subplot(1,2,1)
# plt.imshow(img)
# plt.title('src')
# # cv2.imshow('bm',img)
# # cv2.waitKey(0)
# =============================================================================


# =============================================================================
# import numpy as np
# a= [float(a) for a in np.random.randint(0,30,30)]
# a =np.array(a)
# a= a.reshape(5,6)
# print(a,'\n')
# maximums,minimums,avgs = a.max(axis=0),a.min(axis=0),a.sum(axis=0)/a.shape[0]
# print(maximums,minimums,avgs,'\n')
# for i in range(6):
#     a[:,i]= (a[:,i]-avgs[i])/(maximums[i]-minimums[i])
# print(a)
# =============================================================================


# =============================================================================
# import numpy as np
# 
# class NetWork():
#     def __init__(self,num):
#         np.random.seed(1)
#         self.w = np.random.randn(num,1)
#         self.b = 0.2
#     def forward(self,x):
#         z = np.dot(x,self.w) + self.b
#         return z
# 
# net = NetWork(5)
# x = [1,2,3,4,5]
# y = net.forward(x)
# print(y)
# =============================================================================


# =============================================================================
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# ########     四个不同的滤波器    #########
# img = cv2.imread('d:/pictures/out.png')
# 
# # 均值滤波
# img_mean = cv2.blur(img, (5,5))
# 
# # 高斯滤波
# img_Guassian = cv2.GaussianBlur(img,(5,5),0)
# 
# # 中值滤波
# img_median = cv2.medianBlur(img, 7)
# 
# # 双边滤波
# img_bilater = cv2.bilateralFilter(img,17,100,75)
# 
# # 展示不同的图片
# titles = ['srcImg','mean', 'Gaussian', 'median', 'bilateral']
# imgs = [img, img_mean, img_Guassian, img_median, img_bilater]
# 
# for i in range(5):
#     cv2.imshow('im%d'% i,imgs[i])
# cv2.waitKey(0)
# #     plt.figure(figsize = (15,15))
# #     plt.subplot(5,1,i+1)
# #     imgs[i] = cv2.cvtColor(imgs[i],cv2.COLOR_BGR2RGB)
# #     plt.imshow(imgs[i])
# #     plt.title(titles[i])
# # plt.show()
# =============================================================================


# import colorsys
# import random
# from PIL import Image
# import matplotlib.pyplot as plt
# # 输入文件
# filename = 'D:/pictures/bpData/data0/000182.jpg'
# # 目标色值
# target_hue = 300

# # 读入图片，转化为 RGB 色值5
# image = Image.open(filename).convert('RGBA')
# w, h = image.size
# image.thumbnail((w//5, h//5))
# # 将 RGB 色值分离
# # image.load()
# r, g, b, a = image.split()
# result_r, result_g, result_b, result_a = [], [], [], []
# # 依次对每个像素点进行处理
# for pixel_r, pixel_g, pixel_b, pixel_a in zip(r.getdata(), g.getdata(), b.getdata(), a.getdata()):
#     # 转为 HSV 色值
#     h, s, v = colorsys.rgb_to_hsv(pixel_r / 255., pixel_b / 255., pixel_g / 255.)
#     # 转回 RGB 色系
#     if s < 0.3:
#         # s=random.uniform(0.8,1)
#         s = s + 0.5
#     if v < 0.32:
#         v = v + 0.5
#     rgb = colorsys.hsv_to_rgb(target_hue/360, s, v)
#     pixel_r, pixel_g, pixel_b = [int(x * 255.) for x in rgb]
#     # 每个像素点结果保存
#     result_r.append(pixel_r)
#     result_g.append(pixel_g)
#     result_b.append(pixel_b)
#     result_a.append(pixel_a)

# r.putdata(result_r)
# g.putdata(result_g)
# b.putdata(result_b)
# a.putdata(result_a)

# # 合并图片
# image = Image.merge('RGBA', (r, g, b, a))
# plt.imshow(image)
# # 输出图片
# image.save('D:/pictures/xsz1.png')





# =============================================================================
# import os
# import numpy as np
# import cv2
# # import matplotlib.pyplot as plt
# 
# def colorChange(image,hue):
#     image = cv2.resize(image,None,fx=0.4,fy=0.4,interpolation=cv2.INTER_CUBIC)
#     hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
#     h,s,v = cv2.split(hsv)
#     # hh = np.mean(h)
#     ss = np.mean(s)
#     vv = np.mean(v)
#     if ss > 255*0.3 and vv > 255*0.3:
#         hsv[:,:,0] = hue
#         img =  cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
#         return img
#     elif ss < 255*0.5 and vv > 255*0.4:
#         hsv[:,:,0] = hue
#         hsv[:,:,1] = s + 0.7*255
#         img =  cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
#         return img
#     elif vv < 255*0.4:
#         hsv[:,:,0] = hue
#         hsv[:,:,1] = 0.6*255 + s
#         hsv[:,:,2] = 0.3*255 + v
#         img =  cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
#         return img
#     else:
#         img =  cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
#         return img
# 
# imgDir = 'D:/pictures/bpData/data0'
# imgList = os.listdir(imgDir)
# 
# # image = cv2.imread('D:/pictures/bpData/data0/46-1.jpg')
# # img = colorChange(image, 0)
# # cv2.imshow('hh',img)
# # cv2.waitKey(0)
# 
# for i in range(10):
#     
#     print('process color %d...' % i)
#     for item in imgList: 
#         imgPath = os.path.join(os.path.abspath(imgDir), item)
#         # print(imgPath)
#         # 读入图片
#         image = cv2.imread(imgPath)
#         r = np.random.randint(0, 10)
#         hue = r + i*18
#         img = colorChange(image, hue)
#         dirPath = 'D:/pictures/bpData/color' + str(i)
#         if not os.path.exists(dirPath):
#             os.makedirs(dirPath) 
#         cv2.imwrite(os.path.join(os.path.abspath(dirPath), item), img)
# =============================================================================


# import os
# import numpy as np
# import colorsys
# from PIL import Image
# import matplotlib.pyplot as plt

# imgDir = 'D:/pictures/bpData/data'
# imgList = os.listdir(imgDir)

# for i in range(10):
#     for item in imgList:
        
#         imgPath = os.path.join(os.path.abspath(imgDir), item)
#         # 读入图片，转化为 RGB 色值
#         image = Image.open(imgPath).convert('RGB')
#         hue = np.random.randint(i*36, i*36 + 36)
#         # 将 RGB 色值分离
#         image.load()
#         r, g, b = image.split()
#         result_r, result_g, result_b = [], [], []
#         # 依次对每个像素点进行处理
#         for pixel_r, pixel_g, pixel_b in zip(r.getdata(), g.getdata(), b.getdata()):
#             # 转为 HSV 色值
#             h, s, v = colorsys.rgb_to_hsv(pixel_r / 255., pixel_b / 255., pixel_g / 255.)
#             # print(h)
#             # 转回 RGB 色系
#             rgb = colorsys.hsv_to_rgb(hue/360, s, v) #+ h
#             # print(rgb)
#             pixel_r, pixel_g, pixel_b = [int(x * 255.) for x in rgb]
#             # 每个像素点结果保存
#             result_r.append(pixel_r)
#             result_g.append(pixel_g)
#             result_b.append(pixel_b)
        
#         r.putdata(result_r)
#         g.putdata(result_g)
#         b.putdata(result_b)
        
#         # 合并图片
#         img = Image.merge('RGB', (r, g, b))
#         # plt.imshow(img)
        
#         # 输出图片
#         dirPath = 'D:/pictures/bpData/color' + str(i)5
#         if not os.path.exists(dirPath):
#             os.makedirs(dirPath) 
#         img.save(os.path.join(os.path.abspath(dirPath), item))





# import cv2
# im1 = cv2.imread('d:/pictures/lena.jpg')
# im2 = cv2.imread('d:/pictures/bm.jpg')
# im2 = cv2.resize(im2,(im1.shape[1],im1.shape[0]),0)
# # im = 0.5 * im1 #+ 0.5 * im2
# img = im1[100:390,115:421,:]
# im = cv2.addWeighted(im1, 0.5, im2, 0.5, 0.0)
# # im = im/255
# print(im)
# cv2.imshow('mix',img)
# cv2.waitKey(0)


# from PIL import Image
# import matplotlib.pyplot as plt
# im = Image.open('d:/pictures/lena.jpg')

# plt.imshow(im)
# plt.title('lena')

# import os, shutil
# def file_name(file_dir):
#     # for root, dirs, files in os.walk(file_dir):
#     for x in os.walk(file_dir):
#         # print("root", root)      # 当前目录路径
#         # print("dirs", dirs)      # 当前路径下所有子文件夹
#         # print("files", files)    # 当前路径下所有非子文件夹的文件
#         print(x[0])
       
# # file_name('d:/pictures/data/val')
        
# data_dir = 'd:/pictures/data/val'
# features_dir = 'd:/pictures/features'
# shutil.copytree(data_dir, os.path.join(features_dir, data_dir[3:]))


# coding=utf-8
import os, random, shutil
 
def moveFile(fileDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    picknumber = int(filenumber * ratio)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    for name in sample:
        shutil.move(os.path.join(fileDir, name), os.path.join(tarDir, name))
    return
 
 
if __name__ == '__main__':
    ori_path = 'd:/pictures/data/train'  # 最开始train的文件夹路径
    split_Dir = 'd:/pictures/data/val'   # 移动到新的文件夹路径
    ratio=0.2  # 抽取比例
    for firstPath in os.listdir(ori_path):
        fileDir = os.path.join(ori_path, firstPath)  # 原图片文件夹路径
        tarDir = os.path.join(split_Dir, firstPath)  # val下子文件夹名字
        if not os.path.exists(tarDir): #如果val下没有子文件夹，就创建
            os.makedirs(tarDir)
        moveFile(fileDir)  # 从每个子类别开始逐个划分