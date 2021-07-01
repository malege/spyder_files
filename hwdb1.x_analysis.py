# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
data_dir = 'e:/data/train_1.x'
char_dirs = os.listdir(data_dir)
l = []
for char_dir in char_dirs:
    char_path = os.path.join(data_dir, char_dir)
    n = len(os.listdir(char_path))
    l.append(n)
print('min,max:', min(l), max(l))
# l = sorted(l)
# print(l)

hist, bins = np.histogram(l, 576, [0, 576])
plt.plot(hist,color='blue')