# -*- coding: utf-8 -*-
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
# def rotate_with_fill(img, magnitude):
#     rot = img.convert("RGBA").rotate(magnitude)
#     return Image.composite(rot,Image.new("RGBA", rot.size, (128, ) * 4),rot).convert(img.mode)

# im = Image.open(r'E:\python_projects\spyder_pyfile\imgs\1.jpg')
# im = rotate_with_fill(im, -1)
# im.show()

im = Image.new('RGB', (5,5), 100)#.convert('RGBA')
array = im.load()
print(array[0,0])
# im.show()
im = np.array(im)
print(im)


# =============================================================================
# img = Image.open(r'E:\python_projects\spyder_pyfile\imgs\1.jpg')
# img.show()
# # r,g,b = img.split()
# # r.show()
# # g.show()
# # b.show()
# 
# magnitude = 0.15
# rnd_ch_op = random.choice
# img = img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude * rnd_ch_op([-1, 1]), 1, 0),\
#               Image.BICUBIC, fillcolor=(128, 128, 128))
# img.show()
# =============================================================================
