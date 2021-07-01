# -*- coding: utf-8 -*-
from collections import defaultdict

dict_file = 'e:/data/char_dict.txt'
with open(dict_file, 'r', encoding = 'utf-8') as f:
    lines = f.readlines()
hwdb_1x_dict = [x.strip() for x in lines]
 
# baidu_ppocr_dict_v1
bd_dict_file = 'e:/ppocr_keys_v1.txt'
with open(bd_dict_file, 'r', encoding = 'utf-8') as f:
    lines = f.readlines()
bd_dict_vl = [x.strip() for x in lines]
    
num = 0
s0 = set()
s1 = set()
d = defaultdict(int)
file_path = 'e:/wiki_0_new.txt'

with open(file_path, 'r', encoding = 'utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    for c in line:
        d[c] += 1 
        if c not in hwdb_1x_dict:
            s0.add(c)
            line = line.replace(c, '')
        else:
            if c not in bd_dict_vl: 
                s1.add(c)
    num += len(line)
    
# dt = sorted(d.items(), key = lambda x:x[1])
# print(dt)
print('Total character nums: %s' % num)  #1709134
print('len(d):', len(d))  #5852
print('len(s0):', len(s0))  #877
print('len(s1):', len(s1))  #302

s0 = list(s0)
s1 = list(s1)
with open('e:/extra_char_not_in_hwdb_1x_dict.txt', 'w', encoding = 'utf-8') as f1:
    for i in range(len(s0)):
        f1.writelines(s0[i] + '\t' + str(d[s0[i]]) + '\n')
        
s1 = list(s1)
with open('e:/extra_char_not_in_bd_dict.txt', 'w', encoding = 'utf-8') as f2:
    for i in range(len(s1)):
        f2.writelines(s1[i] + '\t' + str(d[s1[i]]) + '\n')
        
        
        
        
        