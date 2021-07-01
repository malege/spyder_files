# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:35:40 2021

@author: Yang
"""

import cv2
img = cv2.imread(r'C:\Users\Yang\Desktop\test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
cv2.imwrite('./test1.jpg', gray)
cv2.waitKey(0)