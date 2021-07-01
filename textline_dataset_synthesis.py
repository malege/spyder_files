# -*- coding: utf-8 -*-
import os
import random
import cv2
import numpy as np
from collections import defaultdict

def preprocess(img, n, p, m, k = 4):
    # pad the img to make it squared
    pad_size = abs(img.shape[0] - img.shape[1]) // 2
    if img.shape[0] < img.shape[1]:
        pad_dims = ((pad_size, pad_size), (0, 0), (0, 0))
    else:
        pad_dims = ((0, 0), (pad_size//2, pad_size//2), (0, 0))
        # pad_dims = ((0, 0), (0, 0), (0, 0))
    img = np.pad(img, pad_dims, mode='constant', constant_values=255)

    # rescale and add empty border
    if p == 0:
        img = cv2.resize(img, (n//5, n//5))
        img = np.pad(img, ((int(n*0.875), k), (0, k), (0, 0)), mode='constant', constant_values=255)
        img = cv2.resize(img, (n//2, n))
    elif p == 1:
        img = cv2.resize(img, (n//2, n//2))
        img = np.pad(img, ((int(n*0.2), k//2), (0, 0), (0, 0)), mode='constant', constant_values=255)
        img = cv2.resize(img, (n//2, n)) 
    elif p == 2:
        img = cv2.resize(img, (n//2, n//2))
        img = np.pad(img, ((int(n*0.3), int(n*0.25)), (0, 0), (0, 0)), mode='constant', constant_values=255)
        img = cv2.resize(img, (n//2, n)) 
    elif p == 3:
        img = cv2.resize(img, (n//4, n//4))
        img = np.pad(img, ((k, int(n*0.6)), (int(n*0.2), k), (0, 0)), mode='constant', constant_values=255)
        img = cv2.resize(img, (n//2, n)) 
    elif p == 4:
        img = cv2.resize(img, (n//2, n//2))
        img = np.pad(img, ((k, int(n*0.6)), (k,int(n*0.2)), (0, 0)), mode='constant', constant_values=255)
        img = cv2.resize(img, (n//2, n)) 
    elif p ==5:
        img = cv2.resize(img, (n - 8*2, n - 8*2))
        img = np.pad(img, ((8, 8), (0, 0), (0, 0)), mode='constant', constant_values=255)
        img = cv2.resize(img, (n//2, n))
    elif p ==6:
        img = cv2.resize(img, (n - k*2, n - k*2))
        img = np.pad(img, ((k, k), (0, 0), (0, 0)), mode='constant', constant_values=255)
        img = cv2.resize(img, (n//2, n))
    elif p ==7:
        img = cv2.resize(img, (n - k*2, n - k*2))
        img = np.pad(img, ((k, k), (k, 0), (0, 0)), mode='constant', constant_values=255)
        img = cv2.resize(img, (n//2, n))
    elif p ==8:
        img = cv2.resize(img, (n - k*2, n - k*2))
        img = np.pad(img, ((k, k), (0, k), (0, 0)), mode='constant', constant_values=255)
        img = cv2.resize(img, (n//2, n))
    elif p == 9:
        img = cv2.resize(img, (n//7 - 0, n//7 - 0))
        img = np.pad(img, ((n-k-2-n//7, k+2), (2, 2), (0, 0)), mode='constant', constant_values=255)
    elif p == 10:
        img = cv2.resize(img, (n//2, n))
    else:
        img = cv2.resize(img, (n - k*2, n - k*2))
        img = np.pad(img, ((k, k), (m, m), (0, 0)), mode='constant', constant_values=255)
    return img


dict_file = 'e:/data/char_dict.txt'
d = defaultdict(str)
with open(dict_file, 'r', encoding = 'utf-8') as f:
    lines = f.readlines()
for i in range(len(lines)):
    k = lines[i].strip()
    d[k] = format(str(i), '0>5s')

file_path = 'e:/wiki_0.txt'
################################################################################
n = 32
with open(file_path, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
for r in range(4):
    print('current r: ', r)
    num = 0
    # lines = random.shuffle(lines)
    for line in lines:
        line = line.strip().replace('\0x00', '')
        
        im = []
        m = random.randint(0, 1)
        x = np.random.randint(r*10, (r+1)*10)
        for c in line:
            if c in ',，。、': p = 0
            elif c in ';；!！': p = 1
            elif c in ':：': p = 2
            elif c in '“‘': p = 3
            elif c in '”’"': p = 4
            elif c in 'abcdefghijklmnopqrstuvwxyz0123456789%$#&*+-/\|=<>？@~§±×÷…℃‰※℉℅№←↑→↓↖↗↘↙∈∏∑√∝∞∥∧∨∩∪∫∮∴∵∽≈≌≠≡≤≥⊕⊙⊥①②③④⑤⑥⑦⑧⑨⑩□△▽◇○◎☆々㎎㎏㎜㎝㎞㎡㏄￥￡': 
                p = 5
            elif c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ': p = 6
            elif c in '(（【[{《': p = 7
            elif c in '）)】]}》': p = 8
            elif c == '.': p = 9
            elif c == ' ': 
                p = 10
                img = np.zeros((32,16,3), np.uint8)
                img.fill(255)
                img = preprocess(img, n, p, m)
                im.append(img)
                continue
            else: p = -1
            
            ############################
            if c == '：': c = ':'
            elif c == '；': c = ';'
            elif c == '，': c = ','
            elif c == '！': c = '！'
            elif c == '（': c = '('
            elif c == '）': c = ')' 
            
            
            char_value = d[c]
            if char_value=='':
                line = line.replace(c, '')
                continue
            char_dir = os.path.join('e:/data/train_1.x', char_value)
            imglist = os.listdir(char_dir)
            
            # img_name = random.choice(imglist)
            # img_path = os.path.join(char_dir, img_name)
            
            img_path = os.path.join(char_dir, imglist[x])
            img = cv2.imread(img_path)
            img = preprocess(img, n, p, m)
            im.append(img)
        image = im[0]
        for i in range(len(im)-1):
            image = np.hstack((image,im[i+1]))
            
# =============================================================================
#         # # gamma 变化
#         # fi = image / 255.0
#         # gamma = random.uniform(1,2)
#         # image = np.power(fi, gamma)
#         
#         
#         # # 随机旋转
#         # rows, cols, _ = image.shape
#         # angle = random.randint(-1,1)
#         # print(angle)
#         # M = cv2.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), angle, 1)
#         # if angle >= 1:
#         #     cols = cols + 8 * angle
#         #     rows = rows + 8 * angle
#         # image = cv2.warpAffine(image, M, (cols,rows), borderValue = (255, 255, 255))
#         # cv2.imshow('image', image)
# =============================================================================
        
            
        # 使用的wiki文件不同，需要修改对应wiki_0字串名
        prefix_name = 'wiki_0' + '_' + format(str(r), '0>3s') + '_' + format(str(num), '0>6s')
        with open('e:/datasets/wiki_data/train_labels/' + prefix_name + '.txt', 'w', encoding = 'utf-8') as f1:
            f1.writelines(line)
        cv2.imwrite('e:/datasets/wiki_data/train_images/' + prefix_name + '.jpg', image)
        num += 1
        if num%1000==0:
            print('have write %s file' % num)
print('all %s samples write down!' % ((r + 1)*num))

