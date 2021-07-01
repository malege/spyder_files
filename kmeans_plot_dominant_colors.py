# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 20:58:25 2020
@author: Yang
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    print(hist,_)
    hist = hist.astype("float")
    hist /= hist.sum()
    # print(hist)
    centroids = np.uint8(clt.cluster_centers_)
    # print(centroids)
    # mul = np.dot(hist,centroids)
    # print(mul)
    dic = defaultdict(int)
    for i in range(len(np.unique(clt.labels_))):
        key = (centroids[i][0],centroids[i][1],centroids[i][2])
        dic[key] = hist[i]
    dic = dic.items()
    dic = sorted(dic,key=lambda x: -x[1])
    dominant_colors = [x[0] for x in dic]
    print(dominant_colors)
    return dic # hist
    

# =============================================================================
# def plot_colors(hist, cent):
#     start = 0
#     end = 0
#     myRect = np.zeros((50, 300, 3), dtype="uint8")
#     tmp = hist[0]
#     tmpC = cent[0]
#     for (percent, color) in zip(hist, cent):
#         if(percent > tmp):
#             tmp = percent
#             tmpC = color
#     end = start + (tmp * 300) # try to fit my rectangle 50*300 shape
#     cv2.rectangle(myRect, (int(start), 0), (int(end), 50),
#                   tmpC.astype("uint8").tolist(), -1)
#     start = end
#     #rest will be black. Convert to black
#     for (percent,color) in zip(hist, cent):
#         end = start + (percent * 300)  # try to fit my rectangle 50*300 shape
#         if(percent != tmp):
#             color = [0, 0, 0]
#             cv2.rectangle(myRect, (int(start), 0), (int(end), 50),
#                       color, -1) #draw in a rectangle
#             start = end
#     return myRect
# =============================================================================

# def plot_colors2(hist, centroids):
#     bar = np.zeros((40, 600, 3), dtype="uint8")
#     startX = 0
#     for (percent, color) in zip(hist, centroids):
#         # plot the relative percentage of each cluster
#         endX = startX + (percent * 800)
#         cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
#                       color.astype("uint8").tolist(), -1)
#         startX = endX
        
def plot_colors2(dic):
    bar = np.zeros((40, 600, 3), dtype="uint8")    
    startX = 0
    for x in dic:
        # plot the relative percentage of each cluster
        endX = startX + (x[1] * 600)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 40),
                      np.array(list(x[0])).astype("uint8").tolist(), -1)
        startX = endX
    return bar

img = cv2.imread("e:/pictures/nm.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, None,fx=0.2,fy=0.2,interpolation=cv2.INTER_CUBIC)
img = img.reshape((-1,3))
img = np.float32(img)
clt = KMeans(n_clusters=10, random_state=0).fit(img) #

# print(np.uint8(clt.cluster_centers_))

hist = find_histogram(clt)
bar = plot_colors2(hist)
# bar = plot_colors2(hist, clt.cluster_centers_)

# cv2.imshow('bar',bar)
# cv2.waitKey(0)
plt.figure(figsize = (9,9))
plt.title('dominant colors',fontsize=20)
plt.imshow(bar)
