import cv2
# import pdb
img = cv2.imread(r'C:\Users\Yang\Desktop\000029086.jpg')
# img1 = cv2.imread(r'E:\python_projects\spyder_pyfile\imgs\sg.jpg')
img = cv2.resize(img, (200, img.shape[0]))
# im = cv2.add(img,img1)
# im = cv2.resize(im, (im.shape[1]*5, im.shape[0]*5))
cv2.imwrite('./imgs/hah.jpg',img)
cv2.imshow('im',img)
cv2.waitKey(0)


# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw
# im1 = Image.open(r'E:\pictures\lena.jpg')#.convert('L')
# im2 = Image.open(r'E:\python_projects\spyder_pyfile\imgs\sg.jpg')
# im2 = im2.resize(im1.size)
# seq = im1.getdata()
# print(seq)
# # seq0 = list(seq)
# # print(seq0)
# print(im1.getextrema())
# l = im1.histogram()
# print(len(l))

# # fig = plt.figure()
# # fig.plt(x,l[x])
# # plt.show()
# im = Image.new('RGB', (500,500), 'white')
# draw = ImageDraw.Draw(im)
# draw.line([(i, im2.getextrema()[0][1] - l[i]) for i in np.arange(768)], fill = (255,0,0))
# im.show()



# t = im1.getbands()
# print(t)
# im = Image.blend(im1, im2, 0.3)
# im.show()


# import numpy as np
# arr1 = np.ones((3, 3))
# arr2 = np.array(arr1)
# arr3 = np.asarray(arr1)
# arr1[1] = 2
# print('arr1:\n', arr1)
# print('arr2:\n', arr2)
# print('arr3:\n', arr3)



# import cv2
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
#
# if __name__ == '__main__' :
#
#     # Read image
#     im = cv2.imread(r'C:\Users\Yang\Desktop\1.png')
#     # # im = cv2.resize(im, None, None,0.5,0.5)
#     gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#     imk =  Image.open(r'C:\Users\Yang\Desktop\1.png')
#     g = imk.convert('L')
#     g.save(r'C:\Users\Yang\Desktop\2.png')
#     # plt.show(g)
#     # print(gray.shape)
#     # cv2.imshow("i", gray)
#     # # Select ROI
#     # showCrosshair = False
#     # r = cv2.selectROI(im, showCrosshair)
#     # # Crop image
#     # imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
#     # # Display cropped image
#     # cv2.imshow("Image", imCrop)
#     # cv2.waitKey(0)
#
#     # dst = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 10)
#     _, dst = cv2.threshold(gray, 200,240,cv2.THRESH_BINARY)
#     img = np.zeros((im.shape[0],im.shape[1],3))
#     img[:,:,0] = dst
#     img[:,:,1] = dst
#     img[:,:,2] = dst
#     # binary = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
#     # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 10)
#     # contours, _ = cv2.findContours(dst, 2, 1)
#
#     # for c in contours[1:]:
#     #     cv2.drawContours(im, [c], 0, (0, 0, 255), 2)
#     #     x, y, w, h = cv2.boundingRect(c)  # 外接矩形
#     #     cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1)
#     # cv2.drawContours(im, contours, -1, (0, 0, 255), 3)
#     cv2.imshow("Image", im)
#     # cv2.imshow("dst", dst)
#     cv2.imshow("img", img)
#     # cv2.imshow("binary", binary)
#     cv2.waitKey(0)



# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.image as maping
# img = maping.imread('d:/pictures/025.png')
# plt.imshow(img)
# plt.show()