#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:46:54 2021

@author: male
"""
import numpy as np
import matplotlib.pyplot as plt
text_file = '/home/male/datasets/w_h_num.txt'
with open(text_file, 'r', encoding='utf-8') as f:
        datalist = f.readlines()
nSamples = len(datalist)
print('nSamples:', nSamples)
w_h_ratios = []
line_char_nums = []
for i in range(nSamples):
    w_h, num = datalist[i].strip('\n').split('\t')
    w_h_ratios.append(float(w_h))
    line_char_nums.append(int(num))
print('w_h_ratios:', 'min:', min(w_h_ratios), 'max:', max(w_h_ratios))
print('line_char_nums:', 'min:', min(line_char_nums), 'max:', max(line_char_nums)) 

wh = [x for x in w_h_ratios if int(x) <= 25]
print(len(wh))

nu = [x for x in line_char_nums if int(x) <= 45]
print(len(nu))

hist1, bins1=np.histogram(w_h_ratios, 29, [0,29])
hist2, bins2=np.histogram(line_char_nums, 52, [0,52])
print(bins1)
print(bins2)
plt.plot(hist1, color='r')
plt.plot(hist2, color='g')
plt.show()
