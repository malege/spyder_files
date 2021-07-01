# -*- coding: utf-8 -*-
# 对一幅给定单背景图像，转换为RGBA模式，然后使背景像素透明度为0，方便添加各种背景
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

def bg_replace(pic, bg_file):
    
    # gray = np.array(pic.convert('L'))
    # thresh = threshold_otsu(gray)
    # gray[gray > thresh] = 255
    # gray[gray <= thresh] = 50
    # gray = Image.fromarray(gray)
    
    # pic = pic.convert('RGBA')
    w, h = pic.size
    array = pic.load() 
    for i in range(w):
        for j in range(h):
            pos = array[i,j]
            isEdit = (sum([1 for x in pos[0:3] if x > 180]) == 3)
            if isEdit:
                array[i,j] = (255,255,255,0)
                
    pic_copy = pic.copy()
    
    # bg_file_path = os.path.join('./imgs/bg', bg_file)
    # img = Image.open(bg_file_path)
    # w1, h1 = img.size
    # if w1<w or h1<h:
    #     return pic
    # elif w1==w and h1>h:
    #     x = 0
    #     y = np.random.randint(0, h1-h)
    #     bg_sample_img = img.crop((x, y, x+w, y+h))
    # elif w1>w and h1==h:
    #     x = np.random.randint(0, w1-w)
    #     y = 0
    #     bg_sample_img = img.crop((x, y, x+w, y+h))
    # else:
    #     x = np.random.randint(0, w1-w)
    #     y = np.random.randint(0, h1-h)
    #     bg_sample_img = img.crop((x, y, x+w, y+h))
    
    # r = np.random.randint(50, 70)
    # g = np.random.randint(50, 70)
    # b = np.random.randint(50, 70)
    # bg_sample_img.paste(Image.new('RGB', pic.size, (50,50,50)), (0,0), pic_copy)
    
    
    # ========================================================================
    # pic作为mask，处理后pic背景透明，前景文字部分作为mask从im上圈出作用区域
    # im  = Image.new('RGB', pic.size, 0)
    # pic_copy = pic.copy()
    # pic_copy.paste(im,(0,0),pic)
    # 
    # 
    # 
    # composite(image1, image2, mask)方法将image1粘贴到image2上
    # im = Image.composite(Image.new('RGB', pic.size, 255), pic, pic)
    # ========================================================================
    # return bg_sample_img
    return pic_copy

pic = Image.open('imgs/bg_bak/ch1.png')
bg_file = 'imgs/bg/001.jpg'
im = bg_replace(pic, bg_file)
# im = im.convert('RGB')
im.save('imgs/' + 'ch.png')



# bg_imgs = os.listdir(r'./imgs/bg')

# # bg_file = '001.jpg'

# files_dir = r'E:\python_projects\spyder_pyfile\imgs\test'
# files = os.listdir(files_dir)
# for file in files:
#     file_path = os.path.join(files_dir, file)
#     pic = Image.open(file_path)
#     bg_file = np.random.choice(bg_imgs)
#     im = bg_replace(pic, bg_file)
#     im = im.convert('RGB')
#     im.save('./imgs/math_t/' + file)
#     # plt.imshow(im)
#     # plt.show()


# =============================================================================
# import cv2
# img = cv2.imread(r'E:\python_projects\spyder_pyfile\imgs\1.png', -1)
# print(img.shape)
# cv2.imwrite('jian.png', img)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(gray.shape)
# cv2.imshow('gray',gray)
# cv2.waitKey(0)
# =============================================================================
