# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
import shutil
import random
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt

def bg_replace(pic, bg_file):
    # 对一幅给定单背景图像，转换为RGBA模式，然后使背景像素透明度为0，方便添加各种背景
    pic = pic.convert('RGBA')
    w, h = pic.size
    array = pic.load() 
    for i in range(w):
        for j in range(h):
            pos = array[i,j]
            isEdit = (sum([1 for x in pos[0:3] if x > 240]) == 3)
            if isEdit:
                array[i,j] = (255,255,255,0)
                
    pic_copy = pic.copy()
    
    # backgroud replace   
    bg_file_path = os.path.join('./imgs/bg', bg_file)
    img = Image.open(bg_file_path)
    w1, h1 = img.size
    if w1<w or h1<h:
        return pic.convert('RGB')
    elif w1==w and h1>h:
        x = 0
        y = np.random.randint(0, h1-h)
        bg_sample_img = img.crop((x, y, x+w, y+h))
    elif w1>w and h1==h:
        x = np.random.randint(0, w1-w)
        y = 0
        bg_sample_img = img.crop((x, y, x+w, y+h))
    else:
        x = np.random.randint(0, w1-w)
        y = np.random.randint(0, h1-h)
        bg_sample_img = img.crop((x, y, x+w, y+h))
        
    
    r = np.random.randint(50, 70)
    g = np.random.randint(50, 70)
    b = np.random.randint(50, 70)
    bg_sample_img.paste(Image.new('RGB', pic.size, (r,g,b)), (0,0), pic_copy)
    bg_sample_img = bg_sample_img.convert('RGB')
    
    return bg_sample_img   

if __name__=='__main__':
    
    bg_imgs = os.listdir(r'./imgs/bg')
    files_dir = r'E:\datasets\arithmetic_data\train_images'
    files = os.listdir(files_dir)
    for file in tqdm(files):
        file_path = os.path.join(files_dir, file)
        dst_path = r'E:\python_projects\text_renderer-master\example_data\output\arithmetic_data\trn_aug_images' + '/' + file
        
        bg_file = random.choice(bg_imgs)
        if bg_file=='001.jpg':
            shutil.copyfile(file_path, dst_path)
            # continue
        else:
            pic = Image.open(file_path)
            im = bg_replace(pic, bg_file)
            im.save(dst_path)
