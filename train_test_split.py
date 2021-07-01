# -*- coding: utf-8 -*-
import random
import os
work_dir = r'E:\python_projects\text_renderer-master\example_data\output\enum_corpus'
with open(os.path.join(work_dir, 'rec_labels.txt'), 'rt') as f:
    lines = f.readlines()
 
# lines = lines[20000:30000]
random.shuffle(lines)
pos = int(len(lines) *0.8)

train, test = lines[:pos], lines[pos:]
with open(os.path.join(work_dir, "trn_labels.txt"), 'a') as f1:
    f1.writelines(train)
with open(os.path.join(work_dir, "tst_labels.txt"), 'a') as f2:
    f2.writelines(test)
