import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def preprocess1(img, n, p, m, k = 4):
    # pad the img to make it squared
    pad_size = abs(img.shape[0] - img.shape[1]) // 2
    if img.shape[0] < img.shape[1]:
        pad_dims = ((pad_size, pad_size), (0, 0), (0, 0))
    else:
        pad_dims = ((0, 0), (pad_size//2, pad_size//2), (0, 0))
    img = np.pad(img, pad_dims, mode='constant', constant_values=255)

    # rescale and add empty border
    if p == 0:
        img = cv2.resize(img, (n//5, n//5))
        img = np.pad(img, ((int(n*0.875), k), (0, k), (0, 0)), constant_values=255)
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
    elif p ==5:
        img = cv2.resize(img, (n - 4*2, n - 4*2))
        img = np.pad(img, ((4, 4), (0, 0), (0, 0)), constant_values=255)
        img = cv2.resize(img, (n//2, n))
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
        img = np.pad(img, ((n-k-2-n//5, k+2), (0, 0), (0, 0)), constant_values=255)
    elif p == 10:
        img = cv2.resize(img, (n//2, n))
    else:
        img = cv2.resize(img, (n - k*2, n - k*2))
        img = np.pad(img, ((k, k), (m, m), (0, 0)), constant_values=255)
    return img


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
        ratio = (n) / float(h)
        resized_w = w * ratio
        img = cv2.resize(img, (int(resized_w), (n)))

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
        # img = cv2.resize(img, (n - 4*2, n - 4*2))
        # if y==0:
        #     img = np.pad(img, ((4, 4), (4, 0), (0, 0)), constant_values=255)
        # else:
        #     img = np.pad(img, ((4, 4), (0, 0), (0, 0)), constant_values=255)
        # img = cv2.resize(img, (n//2, n))
        
        # img = np.pad(img, ((4, 4), (0, 0), (0, 0)), constant_values=255)
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

# 将文本行对应的字符用相关图片拼接
def image_joint(line):
    im = []
    
    m = random.randint(0, 1)
    v = random.randint(8,16)
    x = np.random.randint(0, 140)
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
        elif c == ' ': 
            p = 10
            img = np.zeros((32,v,3), dtype = np.uint8)
            img.fill(255)
            # img = preprocess(img, n, p, m)
            im.append(img)
            continue
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
        word_dir = os.path.join('e:/data/test_1.x', word_value)
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
    return image

if __name__ == '__main__':
    
    n = 32
    m = 0 #random.randint(0, 1)
    
    d_file = 'e:/data/char_dict.txt'
    d = defaultdict(str)
    with open(d_file, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        k = lines[i].strip()
        d[k] = format(str(i), '0>5s')
    
    
    
    with open(r'E:\python_projects\text_renderer-master\example_data\text\answer2.txt', 'r') as f:
        a_s = f.readlines()  
    img_dir = 'E:/python_projects/text_renderer-master/example_data/output_test/print_hand_corpus/images'  
    files = os.listdir(img_dir)[15000:]
    # print(files)
    for i, file in enumerate(tqdm(files)):
        file_path = os.path.join(img_dir, file)
        print_img = cv2.imread(file_path)

        line = a_s[i].strip('\n')
        hd_img = image_joint(line)
        image = np.hstack((print_img, hd_img))
        cv2.imwrite('E:/python_projects/text_renderer-master/example_data/output_test/print_hand_corpus/' + 'imgs/' + file, image)
        




