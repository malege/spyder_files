# -*- coding: utf-8 -*-
import random
import os
work_dir = 'E:/python_projects/text_renderer-master/example_data/text'
with open(os.path.join(work_dir, '6_8.txt'), 'rt') as f:
    lines = f.readlines()


# with open(os.path.join(work_dir, "formula.txt"), 'w') as f1:
#     with open(os.path.join(work_dir, "answer.txt"), 'w') as f2:
#         for line in lines:
#             s1, s2 = line.split('=')
#             s1 = s1+'='+'\n'
#             f1.writelines(s1)
#             f2.writelines(s2)


with open(os.path.join(work_dir, "formula2.txt"), 'w') as f1:
    for line in lines:
        s1, s2 = line.split('=')
        s1 = s1+'='+'\n'
        f1.writelines(s1)
