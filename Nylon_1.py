# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 18:41:37 2022

@author: Freddie
"""

import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
from PIL import Image
import cv2
import numpy as np
from scipy.optimize import curve_fit

myvideo=cv2.VideoCapture('C:/Users/Freddie/Data/Nylon_1.mp4')
numberframes=0
while np.array(myvideo.read())[0]:
  numberframes+=1
myvideo=cv2.VideoCapture('C:/Users/Freddie/Data/Nylon_1.mp4')
print(numberframes)
print(myvideo.read())

def read_vid():    
    imageArray=np.array(myvideo.read())[1]
    return np.array(myvideo.read())[1]

def crop_image(imageArray,startx,endx,starty,endy):
    return imageArray[starty:endy,startx:endx,:]

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.3, 0.6, 0.1])

def coords():
    x=[]
    y=[]
    for i in range(0,y_lim):
        for j in range(0,x_lim):
            if white_bob[i][j]>0:
                x.append(j)
                y.append(i)
           
    x_ave=sp.mean(x)
    y_ave=sp.mean(y)
    
    Centre=x_ave,y_ave
    
    x_std=sp.std(x)
    y_std=sp.std(y)
    Std=x_std,y_std
    return x_ave,y_ave,Std

x_min=100      #left limit
x_max=1280   #right limit
y_min=0      #upper limit
y_max=720    #lower limit
x_lim=x_max-x_min  #total numbr of pixels in x direction
y_lim=y_max-y_min  #total numbr of pixels in y direction

displacement=[]
weights=[]
no_frames=0
for i in range(numberframes-1):
    no_frames+=1
    imageArray= np.array(myvideo.read())[1] #read_vid()
    plt.imshow(imageArray);                                 # Use this code to see the image before cropping. If you are using Spyder in Anaconda, you should use PIL.Image.fromarray(imageArray).show() 
    cr_image=crop_image(imageArray,x_min,x_max,y_min,y_max)  #imageArray,startx,endx,starty,endy
    new_image=np.array(cr_image)
    plt.imshow(new_image)                                    # Use this code to see the image after cropping.
    plt.show()
    grey_image=rgb2gray(new_image)                               
    thresh, blackAndWhiteImage = cv2.threshold(grey_image,180,255,cv2.THRESH_BINARY)
    white_bob=np.array(blackAndWhiteImage)
    plt.imshow(white_bob)                                    #You can use this code to see the black and white image.
    plt.show()
    x2, y2, Std = coords()
    displacement.append(np.sqrt(x2**2+y2**2))
    error=np.sqrt((2*x2*Std[0])**2+(2*y2*Std[1])**2)
    weights.append(error)
frame=np.arange(0,no_frames)
print(myvideo.read())
time=frame/24.94
plt.plot(time,displacement,color='red')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (pixels)')
plt.title("Aluminium Bob 1")
#%%
def sin(t,A,w,f,c):
    return A*np.sin(w*t+f)+c
fit,cov=curve_fit(sin,time,displacement,p0=[100,7,0,560],sigma=weights)
fit_eq=sin(time,fit[0],fit[1],fit[2],fit[3])
plt.plot(time,fit_eq)         #This plots the best fit curve alonside the actual data
plt.show()
print(cov)
print("The amplitude is",fit[0],"+-",np.sqrt(cov[0][0]))
print("The time period is",2*np.pi*1/(fit[1]),"+-",np.sqrt(cov[1][1]))