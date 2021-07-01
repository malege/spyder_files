# -*- coding: utf-8 -*-
import os
# data_dir = r'E:\python_projects\text_renderer-master\example_data\output\print_hand_corpus'
# with open(os.path.join(data_dir, 'joint_labels.txt'), 'r') as f:
#     lines = f.readlines()
# with open(os.path.join(data_dir, 'trn_labels.txt'), 'w') as f:
#     for line in lines:
#         line = line.replace('images', 'trn_aug_images').replace('print_hand_corpus', 'print_hand')
        
#         f.writelines(line)
p = os.path.realpath(__file__)
print(p.split('.'))