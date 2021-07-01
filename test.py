# -*- coding: utf-8 -*-
# import csv
# csv_file = csv.reader(open('d:/pictures/color.csv')) 
# #print(csv_file)
# out = open('d:/pictures/test.txt','a',newline='')
# csv_write = csv.writer(out,dialect = 'excel')
# data = []
# for item in csv_file:
#  	#print(item)
#     csv_write.writerow(item)
#     data.append(item)
# #print(data)
# out.close()
# print("write over")

# with  open('d:/pictures/test.csv','w',newline='') as csvFile:
#     #csvFile = open('d:/pictures/test.csv','w',newline='')
#     writer = csv.writer(csvFile)
#     #先写columns_name
#     #writer.writerow(["index","a_name","b_name"])
#     #写入多行用writerows
#     #writer.writerows([[1,2,3],[0,1,2],[4,5,6]])
#     writer.writerows(data)
#     #csvFile.close()

# import pandas as pd
# #任意的多组列表
# a = [1,2,3]
# b = [4,5,6]
# #字典中的key值即为csv中的列名
# dataFrame = pd.DataFrame({'a_name':a,'b_name':b})
# #将DataFrame存储为csv,index表示是否显示行名,default=True
# dataFrame.to_csv('d:/pictures/0001.csv',index=False,sep='')


# import csv
# with open('d:/pictures/test.csv','r') as csvFile:
#     reader = csv.DictReader(csvFile)
#     column = [(row['ColorCode'],row['ColorName'], row['RGB']) for row in reader]
#     # reader = csv.reader(csvFile)
#     # column = [(row[1],row[2], row[3]) for row in reader]
#     #print(type(column[100][2]))
#     #print(column)
#     for i in range(len(column)):
#         RGB = column[i][2]
#         RGB = RGB.split(':')
#         #print(type(RGB))
#         #print(RGB)
#         RGB = [int(x) for x in RGB]
#         print(RGB)

# # -*- coding: utf-8 -*-
# def gramma(x):
#     if x > 0.04045:
#         return (( x + 0.055 ) / 1.055 ) ** 2.4
#     else:   
#         return x / 12.92

# def f(t):
#     if t > 0.008856: 
#         return t ** ( 1/3 )
#     else: 
#         return ( 7.787 * t ) + ( 4 / 29 )    
 
# def rgb2xyz(sR, sG , sB):
#     # sR, sG and sB (Standard RGB) input range = 0 ÷ 255
#     # X, Y and Z output refer to a D65/2° standard illuminant.
    
#     var_R = gramma( sR / 255 ) * 100
#     var_G = gramma( sG / 255 ) * 100 
#     var_B = gramma( sB / 255 ) * 100
    
#     X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
#     Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
#     Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505
#     return X, Y, Z

# def xyz2lab(X,Y,Z):
#     #Reference-X, Y and Z refer to specific illuminants and observers.
#     # d65 2°
#     Reference_X = 95.047		
#     Reference_Y = 100.000
#     Reference_Z = 108.883
    
#     var_X = f(X / Reference_X)
#     var_Y = f(Y / Reference_Y)
#     var_Z = f(Z / Reference_Z)
    
#     CIE_L = ( 116 * var_Y ) - 16
#     CIE_a = 500 * ( var_X - var_Y )
#     CIE_b = 200 * ( var_Y - var_Z )
#     return CIE_L, CIE_a, CIE_b

# def rgb2lab(sR, sG , sB):
#     X, Y, Z = rgb2xyz(sR, sG , sB)
#     L, a, b = xyz2lab(X,Y,Z)
#     print(L, a, b)

# rgb2lab(255,111,97)

# from collections import defaultdict
# import cv2
# import numpy as np
# img = cv2.imread('d:/pictures/xxx.jpg')
# #print(img.shape)
# bgr = defaultdict(int)
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         key = (img[i][j][0], img[i][j][1], img[i][j][2])
#         bgr[key] += 1
# bgr = bgr.items()
# #print(type(bgr))
# k = 50
# dominant_colors = sorted(bgr, key=lambda x: -x[1])[:k]
# dominant_colors = [x[0] for x in dominant_colors]
# print('dominant_colors:',dominant_colors)

# dominantColors =[]
# for x in range(len(dominant_colors)): 
#     img1 = np.zeros((50,80,3),dtype=np.uint8)
#     img1[:,:,0] = dominant_colors[x][0]
#     img1[:,:,1] = dominant_colors[x][1]
#     img1[:,:,2] = dominant_colors[x][2]
#     dominantColors.append(img1)
# #显示提取的测试图片的主要颜色
# for i in range(len(dominantColors)-1):
#     dominantColors[0] = np.hstack((dominantColors[0],dominantColors[i+1]))
# cv2.imshow('dominant Colors', dominantColors[0])
# cv2.waitKey(0)


# coding: utf-8
from collections import defaultdict
import cv2
import os,sys,csv
import numpy as np
import matplotlib.pyplot as plt
def hsv2rgb(H,S,V):
    # H, S and V input range = 0 ÷ 1.0
    # R, G and B output range = 0 ÷ 255
    H = H / 180
    S = S / 255
    V = V / 255
    if S == 0:
        R = V * 255
        G = V * 255
        B = V * 255
    else:
        var_h = H * 6
        if var_h == 6: 
            var_h = 0            
        var_i = int( var_h )   
        var_1 = V * ( 1 - S )
        var_2 = V * ( 1 - S * ( var_h - var_i ) )
        var_3 = V * ( 1 - S * ( 1 - ( var_h - var_i ) ) )
        
        if var_i == 0:
            var_r = V
            var_g = var_3 
            var_b = var_1 
        elif var_i == 1:
            var_r = var_2
            var_g = V
            var_b = var_1
        elif var_i == 2:
            var_r = var_1
            var_g = V
            var_b = var_3 
        elif var_i == 3:
            var_r = var_1
            var_g = var_2
            var_b = V     
        elif var_i == 4:
            var_r = var_3
            var_g = var_1
            var_b = V    
        else:  
            var_r = V
            var_g = var_1
            var_b = var_2  
        R = var_r * 255
        G = var_g * 255
        B = var_b * 255
    return R,G,B

def _f(t):
    if (t ** 3) > 0.008856: 
        return t ** 3
    else: 
        return ( t - 16 / 116 ) / 7.787 
    
def _gramma(x):
    if x > 0.0031308:
        return 1.055 * ( x ** ( 1 / 2.4 ) ) - 0.055
    else:   
        return 12.92 * x
    
def lab2xyz(CIE_L,CIE_a,CIE_b):
    # Reference-X, Y and Z refer to specific illuminants and observers.
    # d65 2°   95.047	100.000	108.883
    
    CIE_L = CIE_L / 2.55
    CIE_a = CIE_a - 128
    CIE_b = CIE_b - 128
    # print('l,a,b',CIE_L,CIE_a,CIE_b)
    
    Reference_X = 95.047		
    Reference_Y = 100.000
    Reference_Z = 108.883
    
    # var_Y = _f(( CIE_L + 16 ) / 116)
    # var_X = _f(CIE_a / 500 + var_Y)
    # var_Z = _f(var_Y - CIE_b / 200)
    
    var_Y = ( CIE_L + 16 ) / 116
    var_X = CIE_a / 500 + var_Y
    var_Z = var_Y - CIE_b / 200
    
    var_Y = _f(var_Y)
    var_X = _f(var_X)
    var_Z = _f(var_Z)
        
    X = var_X * Reference_X
    Y = var_Y * Reference_Y
    Z = var_Z * Reference_Z
    return X,Y,Z

def xyz2rgb(X,Y,Z):
    # X, Y and Z input refer to a D65/2° standard illuminant.
    # sR, sG and sB (standard RGB) output range = 0 ÷ 255
    
    var_X = X / 100
    var_Y = Y / 100
    var_Z = Z / 100
    
    var_R = var_X *  3.2406 + var_Y * -1.5372 + var_Z * -0.4986
    var_G = var_X * -0.9689 + var_Y *  1.8758 + var_Z *  0.0415
    var_B = var_X *  0.0557 + var_Y * -0.2040 + var_Z *  1.0570
    
    var_R = _gramma(var_R)
    var_G = _gramma(var_G)
    var_B = _gramma(var_B)
    
    sR = var_R * 255
    sG = var_G * 255
    sB = var_B * 255
    return sR,sG,sB
def ff(x):
    if x > 255:
       return 255
    elif x < 0:
       return 0
    else:
        return x
def lab2rgb(CIE_L,CIE_a,CIE_b):
    X,Y,Z = lab2xyz(CIE_L,CIE_a,CIE_b)
    sR,sG,sB = xyz2rgb(X,Y,Z)
    
    sR = round(sR)
    sG = round(sG)
    sB = round(sB)
    
    sR = ff(sR)
    sG = ff(sG)
    sB = ff(sB)
    return sR,sG,sB

def find_histogram(labels):
    """
    create a histogram with k clusters
    :param: labels
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(labels)) + 1)
    (hist, _) = np.histogram(labels, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    # hist = sorted(hist,reverse = True)
    hist = sorted(hist,key = lambda x:-x)
    # print(hist)
    return hist

def get_multi_dominant_colors(image): 
    "获取的主颜色顺序 b,g,r"
    image = cv2.resize(image,None,fx=0.2,fy=0.2,interpolation=cv2.INTER_CUBIC)
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    # cv2.imshow('testImage',image)
    data = image.reshape((-1,3))
    data = np.float32(data)
    
    # 定义中心 (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,\
                 300, 1e-4)
        
    # flags = cv2.KMEANS_RANDOM_CENTERS 
    flags = cv2.KMEANS_PP_CENTERS
    
    k = 10
    compactness, labels, centers = cv2.kmeans(data, k, None,\
                                              criteria, 10, flags)
    for i in range(k):
        centers[i] = [round(x) for x in centers[i]]
    centers = np.uint8(centers)
    # print('centers：\n', centers)
    res = centers[labels.flatten()]
    dst = res.reshape((-1,3))
    # 原图量化显示
    # dst = dst.reshape(image.shape)
    # cv2.imshow('dst',dst)
    # cv2.waitKey(0)
    
    bgr = defaultdict(int)
    for i in range(dst.shape[0]):
        key = (dst[i][0],dst[i][1],dst[i][2])
        bgr[key] += 1
    bgr = bgr.items()
    dominant_colors = sorted(bgr,key=lambda x: -x[1])[:k]
    dominant_colors = [x[0] for x in dominant_colors]
    
    # dominant_colors = np.array(dominant_colors).flatten()
    
    # 取k个聚类中心的均值作为色卡的'rgb'值
    # dominant_colors = np.array(dominant_colors)
    # dominant_colors = np.mean(dominant_colors, axis = 0)
    # dominant_colors = [int(x) for x in dominant_colors]
    
    # 每个聚类中心乘以该聚类中心所属簇中像素点个数占总像素点个数的比例
    # weight = find_histogram(labels)
    # dominant_colors = np.dot(weight,dominant_colors)
    # dominant_colors = np.uint8(dominant_colors)
    
    # for i in range(k):
    #     h,s,v = dominant_colors[i][0],dominant_colors[i][1],dominant_colors[i][2]
    #     dominant_colors[i] = hsv2rgb(h,s,v)
    #     dominant_colors[i] = [int(t) for t in dominant_colors[i]] 
    
    # for i in range(k):
    #     l,a,b = dominant_colors[i][0],dominant_colors[i][1],dominant_colors[i][2]
    #     dominant_colors[i] = lab2rgb(l,a,b)
    #     dominant_colors[i] = [int(t) for t in dominant_colors[i]]
    
    print('dominant_colors:',dominant_colors) 
# =============================================================================
#     # 将量化后的某主颜色显示在原图位置
#     im = np.zeros(image.shape,dtype=np.uint8)
#     im = im.reshape(-1,3)
#     for x in range(dst.shape[0]):
#         temp = (dst[x][0],dst[x][1],dst[x][2])
#         if dominant_colors[0] == temp:
#             for j in range(3):
#                 im[x][j] = dst[x][j]
#     im = im.reshape(image.shape)
#     cv2.imshow('im',im)
#     # cv2.imwrite('d:/pictures/tes.jpg',im)
# =============================================================================
    return dominant_colors

# # 读取图片目录，并将识别出来的主色写入csv文件
# imgDir = 'd:/pictures/sk'
# imgList = os.listdir(imgDir)
# print(imgList)
# with open('d:/pictures/sk.csv','w',newline='') as csvFile:
#     writer = csv.writer(csvFile)
#     # 先写columns_name
#     writer.writerow(["rgb","ColorCode"])
#     for item in imgList:
#         imgPath = os.path.join(os.path.abspath(imgDir),item)#.replace('\\','/')
#         # print(imgPath)
#         image = cv2.imread(imgPath)
#         if image is None:
#             print("Error loading Image...")
#             sys.exit()
#         colors = get_multi_dominant_colors(image)
#         r,g,b = colors[2],colors[1],colors[0]
#         rgb = str(str(r)+':'+str(g)+':'+str(b))
#         print('rgb:',rgb)
#         imageName = item.split('.')[0]
#         # print(imageName)
#         # item = (rgb,imageName)
#         writer.writerow((rgb,imageName))
 
       
image = cv2.imread('D:/pictures/nm.jpg')
print(image.shape)
colors = get_multi_dominant_colors(image)
im =[]
for x in range(len(colors)):
    img = np.zeros((40,80,3),dtype=np.uint8)
    img[:,:,0] = colors[x][2]
    img[:,:,1] = colors[x][1]
    img[:,:,2] = colors[x][0]
    im.append(img)
#显示提取的测试图片的k种主要颜色
for i in range(len(im)-1):
    im[0] = np.hstack((im[0],im[i+1]))
# im[0] = cv2.cvtColor(im[0],cv2.COLOR_RGB2BGR)  
# cv2.imshow('dominant Colors', im[0])
# cv2.waitKey(0)
    
plt.figure(figsize = (9,9))
plt.title('dominant colors',fontsize=20)
# im[0] = cv2.cvtColor(im[0],cv2.COLOR_BGR2RGB)
plt.imshow(im[0])  

# =============================================================================
# # 用plt显示图像
# titles = [u'color1', u'color2', u'color3', u'color4',  u'color5']    
# plt.subplot(1,6,1)
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# plt.imshow(image) 
# plt.title('src')
# img = np.zeros((500,500,3),dtype = np.uint8)
# for i in range(len(colors)):
#     img[:,:,0] = colors[i][0]
#     img[:,:,1] = colors[i][1]
#     img[:,:,2] = colors[i][2]
#     #cv2.imshow('color%d' % i, img) 
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     plt.subplot(1,6,i+2)
#     plt.imshow(img) 
#     plt.title(titles[i])  
#     #plt.xticks([]),plt.yticks([])  
# plt.show()
# cv2.waitKey(0)
# =============================================================================

# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# x= np.random.randint(0,255,(25,5))
# y= np.random.randint(0,255,(35,5))
# z = np.vstack((x,y))
# #z= np.arange(30).reshape(5,6)
# #z = z.reshape((-1,1))
# z = z.flatten()
# #z= np.squeeze(z)
# print(z)
# z = np.float32(z)
# plt.hist(z,30,[0,30],color = 'g')
# plt.show()

# #图像直方图
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# img = cv2.imread("d:/pictures/bm.jpg")

# # 
# # plt.hist(img.ravel(), 256, [0, 256],color='r')
# # plt.show()

# bins = np.arange(256).reshape(256,1)
# color = [(255,0,0),(0,255,0),(0,0,255)]
# hight = 300
# histImage = np.zeros((hight,256,3),dtype = np.uint8)
# for ch,col in enumerate(color):
    
#     hist = cv2.calcHist([img],[ch],None,[256],[0,256])
#     cv2.normalize(hist,hist,0,256,cv2.NORM_MINMAX)
#     #hist = np.int32(np.around(hist))
#     pts = np.int32(np.column_stack((bins,hist)))
#     cv2.polylines(histImage,[pts],False,col)   
# histImage = np.flipud(histImage)    
# cv2.imshow('histImage',histImage)
# cv2.waitKey(0)


# hist = cv2.calcHist([img],[0],None,[256],[0,256])
# #minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist) 
# cv2.normalize(hist,hist,0,256,cv2.NORM_MINMAX)
# hist = np.int32(np.around(hist))
# for x,y in enumerate(hist):
#     cv2.line(histImage,(x,0),(x,y),(255,0,0))
# histImage = np.flipud(histImage) #沿水平轴翻转矩阵
# cv2.imshow('histImage',histImage)
# cv2.waitKey(0)

# for x in range(256): 
#     #h = int(hight*0.9*hist[x] /maxVal)
#     #cv2.line(histImage,(x,hight),(x,hight-h),color=[0, 0, 255]) 
#     cv2.line(histImage,(x,hight),(x,hight-hist[x]),color=[0, 0, 255])
# cv2.imshow('histImage',histImage)
# cv2.waitKey(0)


# l,a,b = rgb2lab(4, 87, 86)
# color0 = LabColor(lab_l=l, lab_a=a, lab_b=b)
# print(l,a,b)

# l1,a1,b1 = rgb2lab(23, 96, 84)
# color1 = LabColor(lab_l=l1, lab_a=a1, lab_b=b1)
# print(l1,a1,b1)

# d_e_cie94 = round(delta_e_cie1994(color0, color1), 2)
# print(d_e_cie94)