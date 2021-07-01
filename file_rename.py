# -*- coding: utf-8 -*-
import os
path = 'e:/datasets/wiki_data/train_labels/'

filelist = os.listdir(path)
total_num = len(filelist)

i = 0

for item in filelist:
    if item.endswith('_000.txt'):
        
        # print(item)
        # num = item.split('_')[0]
        
        src = os.path.join(os.path.abspath(path), item)
        dst = os.path.join(os.path.abspath(path),\
                            'wiki_0' + '_000_' + format(str(i), '0>6s') + '.txt')
        os.rename(src, dst)
        print('converting %s to %s ...' % (src, dst)) 
        i = i + 1
print('total %d to rename & converted %d jpgs' % (total_num, i)) 