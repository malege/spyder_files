# -*- coding: utf-8 -*-

# import pdb
import numpy as np
import cv2
img = np.zeros((500,500,3), dtype = np.uint8)
# img.fill(255)
img[:,:,0] = 190
img[:,:,1] = 190
img[:,:,2] = 190
cv2.imshow('img', img)
cv2.imwrite('./imgs/bg/006.jpg', img)
# pdb.set_trace()
# print(img)
cv2.waitKey(0)