# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 23:55:52 2021

@author: andre
"""

import cv2
import numpy as np 


def segmentar(imagen):
    ret,cierre = cv2.threshold(imagen,5,255,cv2.THRESH_BINARY_INV)
    # Copy the thresholded image.
    im_floodfill = cierre.copy()
    h, w = cierre.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    fill = cv2.floodFill(im_floodfill, mask, (0,0), 255);
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = cierre | im_floodfill_inv
    mgray = cv2.cvtColor(im_out, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(mgray, 250, 255, cv2.THRESH_BINARY)
    return binary
    
img = cv2.imread('mascara2.png')   
segmentado = segmentar (img)
cv2.imwrite('segmentada.png',segmentado)

