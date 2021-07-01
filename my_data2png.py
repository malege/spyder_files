# -*- coding: utf-8 -*-
import os
import numpy as np
import struct
from tqdm import tqdm
from PIL import Image
from collections import defaultdict

data_dir = 'e:/Gnt1.0-1.2Test' 

def one_file(f):
    header_size = 10
    while True:
        header = np.fromfile(f, dtype='uint8', count=header_size)
        if not header.size: break
        sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
        tagcode = header[5] + (header[4] << 8)
        width = header[6] + (header[7] << 8)
        height = header[8] + (header[9] << 8)
        if header_size + width * height != sample_size:
            break
        image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
        yield image, tagcode
        
def read_from_gnt_dir(gnt_dir=data_dir):
    for file_name in os.listdir(gnt_dir):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dir, file_name)
            with open(file_path, 'rb') as f:
                for image, tagcode in one_file(f):
                    yield image, tagcode
# d = defaultdict(int)           
if os.path.exists('char_dict.txt'):
        print('found exist char_dict.txt...')
        with open('char_dict.txt', 'r', encoding='utf-8') as f:
            char_set = f.readlines()
            char_set = [i.strip() for i in char_set]
            char_dict = dict(zip(sorted(char_set), range(len(char_set))))
            
            # for i in range(len(char_set)):
            #     k = char_set[i].strip()
            #     d[k] = i
else:
    char_set = set()
    for _, tagcode in read_from_gnt_dir(gnt_dir=data_dir):
        tagcode_unicode = struct.pack('>H', tagcode).decode('gbk')
        tagcode_unicode = tagcode_unicode.replace('\x00', '')
        char_set.add(tagcode_unicode)
    char_set = sorted(list(char_set))
    
    char_dict = dict(zip(sorted(char_set), range(len(char_set)))) 
    
    with open('char_dict.txt', 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(char_set))
print('All got {} characters.'.format(len(char_set)))

counter = 0
for file_name in tqdm(os.listdir(data_dir)):
    print(file_name)
    if file_name.endswith('.gnt'):
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'rb') as f:
            for image, tagcode in one_file(f):
                tagcode_unicode = struct.pack('>H', tagcode).decode('gbk')
                tagcode_unicode = tagcode_unicode.replace('\x00', '')
                im = Image.fromarray(image)
                dir_name = './data/test_1.x/' + '%0.5d' % char_dict[tagcode_unicode]
                # dir_name = './data/train_1.x/' + '%0.5d' % d[tagcode_unicode]
                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
                im.convert('RGB').save(dir_name + '/' + format(str(counter), '0>4s') + '.png')
    counter += 1
