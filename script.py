# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 18:01:27 2020

@author: rejid4996
"""

# Python program to illustrate  
# template matching 
import cv2 
import numpy as np 
import os

os.chdir(r'C:\Users\rejid4996\OneDrive - ARCADIS\Desktop\Files\My Projects\Template matching')

# Read the main image 
img_rgb = cv2.imread('bill bakers.png')
  
# Convert it to grayscale 
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) 
  
# Read the template 
template = cv2.imread('bill bakers.png',0) 
  
# Store width and height of template in w and h 
w, h = template.shape[::-1] 
  
# Perform match operations. 
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED) 
  
# Specify a threshold 
threshold = 0.8
  
# Store the coordinates of matched area in a numpy array 
loc = np.where( res >= threshold)  
  
# Draw a rectangle around the matched region. 
for pt in zip(*loc[::-1]): 
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2) 
  
# Show the final image with the matched area. 
cv2.imshow('Detected',img_rgb) 

#%%
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img_rgb = cv.imread('bill bakers.png')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('dave.png',0)
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
cv.imwrite('res.png',img_rgb)
