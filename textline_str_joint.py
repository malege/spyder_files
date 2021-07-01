# -*- coding: utf-8 -*-
import random
import os
work_dir = 'E:/python_projects/text_renderer-master/example_data/output_test/print_hand_corpus'
with open(os.path.join(work_dir, 'rec_labels.txt'), 'rt') as f:
    lines1 = f.readlines()
dir1 = r'E:\python_projects\text_renderer-master\example_data\text'
with open(os.path.join(dir1, 'answer2.txt'), 'rt') as f:
    lines2 = f.readlines()
# if not len(lines1)==len(lines2):
#     exit()   
# else:
    
with open(os.path.join(work_dir, "joint_labels.txt"), 'a') as f:
    # for i in range(len(lines1)): 
    for i in range(15000, 22500): 
        new_line = lines1[i].strip('\n') + lines2[i-15000]
        f.writelines(new_line)




