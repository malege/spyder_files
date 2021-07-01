# -*- coding: utf-8 -*-
"提取图片主要颜色与色卡库计算色差"
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1994 #, delta_e_cmc
from collections import defaultdict
from sklearn.cluster import KMeans
import numpy as np
import cv2,csv,sys

def gramma(x):
    if x > 0.04045:
        return (( x + 0.055 ) / 1.055 ) ** 2.4
    else:   
        return x / 12.92

def f(t):
    if t > 0.008856: 
        return t ** ( 1/3 )
    else: 
        return ( 7.787 * t ) + ( 4 / 29 )    
 
def rgb2xyz(sR, sG , sB):
    # sR, sG and sB (Standard RGB) input range = 0 ÷ 255
    # X, Y and Z output refer to a D65/2° standard illuminant.
    
    var_R = gramma( sR / 255 ) * 100
    var_G = gramma( sG / 255 ) * 100 
    var_B = gramma( sB / 255 ) * 100
    
    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505
    return X, Y, Z

def xyz2lab(X,Y,Z):
    # Reference-X, Y and Z refer to specific illuminants and observers.
    # d65 2°   95.047	100.000	108.883
    # d65 10°  94.811	100.000	107.304
    
    Reference_X = 95.047		
    Reference_Y = 100.000
    Reference_Z = 108.883
    
    var_X = f(X / Reference_X)
    var_Y = f(Y / Reference_Y)
    var_Z = f(Z / Reference_Z)
    
    CIE_L = ( 116 * var_Y ) - 16
    CIE_a = 500 * ( var_X - var_Y )
    CIE_b = 200 * ( var_Y - var_Z )
    return CIE_L, CIE_a, CIE_b

def rgb2lab(sR, sG , sB):
    X, Y, Z = rgb2xyz(sR, sG , sB)
    L, a, b = xyz2lab(X,Y,Z)
    return L, a, b

def get_multi_dominant_colors(image, k): 
    "获取的多种主颜色，顺序 b,g,r， 返回一个元素为(b, g, r)元组的列表"
    image = cv2.resize(image,None,fx=0.2,fy=0.2,\
                       interpolation=cv2.INTER_CUBIC)
    cv2.imshow('testImage',image)
    data = image.reshape((-1,3))
    data = np.float32(data)
    
    clt = KMeans(n_clusters= k, random_state=0).fit(data)
    centers = np.uint8(clt.cluster_centers_)
    res = centers[clt.labels_.flatten()]
    dst = res.reshape(-1,3)
    
    bgr = defaultdict(int)
    for i in range(dst.shape[0]):
        key = (dst[i][0],dst[i][1],dst[i][2])
        bgr[key] += 1 
    bgr = bgr.items()
    dominant_colors = sorted(bgr,key=lambda x: -x[1])[:k]
    dominant_colors = [x[0] for x in dominant_colors]
    
    # print('dominant_colors:\n%s' % dominant_colors,'\n')
    imgs =[]
    for x in range(len(dominant_colors)):
        img1 = np.zeros((40,80,3),dtype=np.uint8)
        img1[:,:,0] = dominant_colors[x][0]
        img1[:,:,1] = dominant_colors[x][1]
        img1[:,:,2] = dominant_colors[x][2]
        imgs.append(img1)
        
    # 显示提取的输入测试图片的k种主要颜色，并将它们按列拼接在一起显示
    for i in range(len(imgs)-1):
        imgs[0] = np.hstack((imgs[0],imgs[i+1]))
    cv2.imshow('dominant Colors', imgs[0])
    
    return dominant_colors  # 返回一个列表,列表元素为(b, g, r)元组

def colorCompare(image):
    
    # print('Input dominant color number k(eg: 1-20) you want to obtain:')
    # k = int(input().strip())
    
    colors = get_multi_dominant_colors(image, k=10)
    with open('d:/pictures/bpRecognition/tpgsk.csv','r') as csvFile:
        reader = csv.DictReader(csvFile)
        column = [(row['ColorCode'], row['rgb']) for row in reader]    
             
    # 将获取的k种主色分别与色卡库比较并寻找色差最小的m种色号
    im = []  
    similarColors =[]
    for j in range(len(colors)):
    # for j in range(1):
        rgb = (colors[j][2],colors[j][1],colors[j][0])
        l,a,b = rgb2lab(rgb[0], rgb[1] , rgb[2])
        color0 = LabColor(lab_l=l, lab_a=a, lab_b=b)
        delta_e_cie2000s = []
        # delta_e_cmcs = []
        for i in range(len(column)):
            RGB = column[i][1]
            RGB = RGB.split(':')
            RGB = [int(x) for x in RGB]
            
            l1,a1,b1 = rgb2lab(RGB[0], RGB[1] , RGB[2])
            color1 = LabColor(lab_l=l1, lab_a=a1, lab_b=b1)
            d_e_cie2000 = delta_e_cie1994(color0, color1)
            delta_e_cie2000s.append(d_e_cie2000)
            # d_e_cmc = delta_e_cmc(color0, color1)
            # delta_e_cmcs.append(d_e_cmc)
            
        delta_e_cie2000s = np.array(delta_e_cie2000s)
        # delta_e_cmcs = np.array(delta_e_cmcs)
        
        m = 10
        #获取色差最小的m个色卡的行号
        out1 = delta_e_cie2000s.argsort()[:m]
        # out2 = delta_e_cmcs.argsort()[:m]
        
        # 选取色卡列表中色差最小的m种RGB值  
        rgbs = []
        for i in range(m):
            rgbs.append(column[out1[i]][1])
        # print('rgbs:',rgbs)
            
        imgs = []
        similarColors =[]
        # 遍历m种rgb色号，并显示
        for i,rgb in enumerate(rgbs):
            rgb = rgb.split(':')
            rgb = [int(x) for x in rgb]
            # 将rgb值以纯色图像显示
            img = np.zeros((40,80,3),dtype=np.uint8)
            img[:,:,0] = rgb[2]
            img[:,:,1] = rgb[1]
            img[:,:,2] = rgb[0]
            imgs.append(img)
            im.append(img)

            # color = column[out1[i]][1]
            colorCode = column[out1[i]][0]
            # similarColors.append((color,colorCode))
            similarColors.append(colorCode)
        print('similar color%d:\n%s' % (j, similarColors),'\n')

        # 显示色卡中与测试图片主色最相似的色号，按行拼接
        for i in range(len(imgs)-1):
            imgs[0] = np.vstack((imgs[0],imgs[i+1]))
        im.append(imgs[0])
        
    for i in range(len(im)-1):
        im[0] = np.hstack((im[0],im[i+1]))

    skImg =  im[0] 
    return similarColors, skImg

def main(): 
    testImage = cv2.imread('D:/pictures/bpRecognition/cloth/bl12.jpg')
    if testImage is None:
        print("Error loading Image...")
        sys.exit()
        
    # 调用颜色比较函数
    similarColors, skImage = colorCompare(testImage)
    # print('similar color:%s' % similarColors)
    cv2.imshow('similar color', skImage)
    
    # for i in range(len(skImage)):
    #     cv2.imshow('color%d' % i,skImage[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ =='__main__':
    main()