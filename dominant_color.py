# -*- coding: utf-8 -*-
"obtain single dominant color of an image"
import colorsys
import PIL.Image as Image
import numpy as np
import cv2
def get_dominant_color(image): 
    
    image = image.convert('RGBA')
    image.thumbnail((200,200))
    max_score = 0.0001
    dominant_color = None
    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
        if a==0:
            continue
        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13,235)
        y = (y - 16.0) / (235 - 16)
        if y>0.9:
            continue
        score = (saturation+0.1)*count
        if score > max_score:
           max_score = score
           dominant_color = (r, g, b)
    print('dominant color in rgb is %s' % str(dominant_color))
    return dominant_color
    
def main():        
    testImage = Image.open('D:/pictures/bl0.jpg')  
    dominant_color = get_dominant_color(testImage)
    img = np.zeros((300,300,3),dtype=np.uint8)
    img[:,:,0] = dominant_color[2]
    img[:,:,1] = dominant_color[1]
    img[:,:,2] = dominant_color[0]  
    cv2.imshow('img',img)
    cv2.waitKey(0)

if __name__ =='__main__':
    main()   