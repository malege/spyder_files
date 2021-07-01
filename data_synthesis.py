# -*- coding: utf-8 -*-
import os
import random
import cv2
import numpy as np
from collections import defaultdict

def preprocess(img, n, p, m, y, k = 4):
    
    # pad the img to make it squared
    # pad_size = abs(img.shape[0] - img.shape[1]) // 2
    # if img.shape[0] < img.shape[1]:
    #     pad_dims = ((pad_size, pad_size), (0, 0), (0, 0))
    # else:
    #     pad_dims = ((0, 0), (pad_size//2, pad_size//2), (0, 0))
    # img = np.pad(img, pad_dims, mode='constant', constant_values=255)
    
    
    pad_size = abs(img.shape[0] - img.shape[1]) // 2
    if img.shape[0] < img.shape[1]:
        pad_dims = ((pad_size, pad_size), (0, 0), (0, 0))
        img = np.pad(img, pad_dims, mode='constant', constant_values=255)
        img = cv2.resize(img, (n//2, n))
    else:
        h, w = img.shape[:2]
        # ratio = (n-8) / float(h)
        # resized_w = w * ratio
        # img = cv2.resize(img, (int(resized_w), (n-8)))
        
        ratio = n / float(h)
        resized_w = w * ratio
        img = cv2.resize(img, (int(resized_w), n))

    # rescale and add empty border
    if p == 0:
        img = cv2.resize(img, (n//5, n//5))
        pad_dims = ((int(n*0.875), k), (0, k), (0, 0)), 
        img = np.pad(img, pad_dims, mode='constant', constant_values=255)
        img = cv2.resize(img, (n//2, n))
    elif p == 1:
        img = cv2.resize(img, (n - k*2, n - k*2))
        img = np.pad(img, ((3*k//2, k//2), (0, 0), (0, 0)), constant_values=255)
        img = cv2.resize(img, (n//2, n)) 
    elif p == 2:
        img = cv2.resize(img, (n//2, n//2))
        img = np.pad(img, ((n-k*2-n//2, k*2), (0, 0), (0, 0)), constant_values=255)
        img = cv2.resize(img, (n//2, n)) 
    elif p == 3:
        img = cv2.resize(img, (n//4, n//4))
        img = np.pad(img, ((k, int(n*0.6)), (int(n*0.2), k), (0, 0)), constant_values=255)
        img = cv2.resize(img, (n//2, n)) 
    elif p == 4:
        img = cv2.resize(img, (n//2, n//2))
        img = np.pad(img, ((k, int(n*0.6)), (k, k), (0, 0)), constant_values=255)
        img = cv2.resize(img, (n//2, n)) 
    # elif p ==5:
    #     img = cv2.resize(img, (n - 4*2, n - 4*2))
    #     img = np.pad(img, ((4, 4), (0, 0), (0, 0)), constant_values=255)
    #     img = cv2.resize(img, (n//2, n))
    
    elif p ==5:
        # # img = cv2.resize(img, (n - 4*2, n - 4*2))
        # if y==0:
        #     img = np.pad(img, ((4, 4), (4, 0), (0, 0)), constant_values=255)
        # else:
        #     img = np.pad(img, ((4, 4), (0, 0), (0, 0)), constant_values=255)
        # # img = cv2.resize(img, (n//2, n))
        # h, w = img.shape[:2]
        # ratio = n / float(h)
        # resized_w = w * ratio
        # img = cv2.resize(img, (int(resized_w), n))
        pass
        
    elif p ==6:
        img = cv2.resize(img, (n - k*2, n - k*2))
        img = np.pad(img, ((k, k), (0, 0), (0, 0)), constant_values=255)
        img = cv2.resize(img, (n//2, n))
    elif p ==7:
        img = cv2.resize(img, (n - k*2, n - k*2))
        img = np.pad(img, ((k, k), (k, 0), (0, 0)), constant_values=255)
        img = cv2.resize(img, (n//2, n))
    elif p ==8:
        img = cv2.resize(img, (n - k*2, n - k*2))
        img = np.pad(img, ((k, k), (0, k), (0, 0)), constant_values=255)
        img = cv2.resize(img, (n//2, n))
    elif p == 9:
        img = cv2.resize(img, (n//5 - 0, n//5 - 0))
        img = np.pad(img, ((n-k-2-n//5, k+2), (2, 2), (0, 0)), constant_values=255)
    elif p == 10:
        img = cv2.resize(img, (n//2, n))
    else:
        img = cv2.resize(img, (n - k*2, n - k*2))
        img = np.pad(img, ((k, k), (m, m), (0, 0)), constant_values=255)
    return img

def image_joint(line):
    # 将文本行对应的字符用相关图片拼接
    x = random.randint(0, 224)
    im = []
    for y, c in enumerate(line):
        if c in ',，。、': p = 0
        elif c in ';；!！': p = 1
        elif c in ':：': p = 2
        elif c in '“‘': p = 3
        elif c in '”’"': p = 4
        elif c in 'abcdefghijklmnopqrstuvwxyz0123456789%$#*+-/\|=<>?@~§±×÷…℃\‰※℉℅№←↑→↓↖↗↘↙∈∏∑√∝∞∥∧∨∩∪∫∮∴∵∽≈≌≠≡≤≥⊕⊙⊥①②③④⑤⑥⑦⑧⑨⑩□△▽◇○◎☆々㎎㎏㎜㎝㎞㎡㏄￥￡': 
                p = 5
        elif c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ': p = 6
        elif c in '(（【[{《': p = 7
        elif c in '）)】]}》': p = 8
        elif c == '.': p = 9
        
        # elif c == ' ': 
        #     p = 10
        #     img = np.zeros((32,10,3), dtype = np.uint8)
        #     img.fill(255)
        #     # img = preprocess(img, n, p, m)
        #     im.append(img)
        #     continue
        
        else: p = -1
        
        # 从hwdb1.x解析出的字典不含部分中文标点符号
        if c == '：': c = ':'
        elif c == '；': c = ';'
        elif c == '，': c = ','
        elif c == '！': c = '！'
        elif c == '（': c = '('
        elif c == '）': c = ')' 
        
        word_value = d[c]
        if word_value=='':
            line = line.replace(c, '')
            # print(line)
            continue
        word_dir = os.path.join('e:/data/train_1.x', word_value)
        imglist = os.listdir(word_dir)
        
        # img_name = random.choice(imglist)
        # img_path = os.path.join(word_dir, img_name)
        
        img_path = os.path.join(word_dir, imglist[x])
        img = cv2.imread(img_path)
        img = preprocess(img, n, p, m, y)
        im.append(img)
    image = im[0]
    for i in range(len(im)-1):
        image = np.hstack((image,im[i+1]))
    
    # =============================================================================
    # gamma 变化
    # fi = image / 255.0
    # gamma = random.uniform(1,2)
    # image = np.power(fi, gamma)+200
    # 
    # # 随机旋转
    # rows, cols, _ = image.shape
    # angle = random.randint(-1,1)
    # print(angle)
    # M = cv2.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), angle, 1)
    # if angle >= 1:
    #     cols = cols + 8 * angle
    #     rows = rows + 8 * angle
    # image = cv2.warpAffine(image, M, (cols,rows), borderValue = (255, 255, 255))
    # =============================================================================
    
    return image

# 测试文本
t0 = '2020年5月12日，"China.Daily"，中国日报（人民论坛）：“说到就要做到；也一定能够做到!全年增长35.55%！”。'
t1 = '“开展‘不忘初心、牢记使命’主题教育，归根结底就是必须始终为中国人民谋幸福、为中华民族谋复兴”。'
t2 = '《吸血鬼的故事》Vikram and the Vampire'
t3 = 'eius）所著《金驴记》（The Golden Ass）出现的《爱神与美女》'
t4 = '埃及神话中大多数神祇都具有兽头人身的形象，这是埃及神话的一个显著特点。'
t5 = '未若锦囊收艳骨，一抔净土掩风流；质本洁来还洁去，强于污淖陷渠沟。'
t6 = '尔今死去侬收葬，未卜侬身何日丧；侬今葬花人笑痴，他年葬侬知是谁。'
t7 = '燕顺的喽啰正要剖开宋江的心肝，宋江悲叹大喊「想不到我宋江竟要命丧于此ㄕ'
t8 = '透过出色的攻击和守备表现，拿坡里力争上游，成为天使队正规的先发捕手。'
t9 = 'https://www.baidu.com'

s0 = '94 + 37 = 131'
s1 = '11.95 + 156.1 = 168.05'

if __name__ == '__main__':

    d_file = 'e:/data/char_dict.txt'
    d = defaultdict(str)
    with open(d_file, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        key = lines[i].strip()
        d[key] = format(str(i), '0>5s')
    
    n = 32
    m = 0 #random.randint(0, 1)
    
    image = image_joint(s1)
    
    cv2.imshow('image', image)
    cv2.imwrite('./imgs/a_f.jpg',image)
    # print('processed line:', line)
    # print('len of line:', len(line))
    cv2.waitKey(0)