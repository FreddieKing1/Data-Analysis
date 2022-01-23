# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 17:03:21 2022

@author: Freddie
"""
import matplotlib.pyplot as plt
import pandas as pd              #Imports all relevant modules
import numpy as np

bubble=pd.read_csv("C:/Users/Freddie/Data/Interferometry.csv") #Reads in the pixel measurement data
shift=bubble["Fringe Shift (Pixels)"]   #Assigns the fringe shift values to 'shift'
width=bubble["Fringe Width (Pixels)"]   #Assigns the fringe width values to 'width'
w_std=np.std(width)
w_mean=np.mean(width)
w_err=w_std/len(width)
s_std=np.std(shift)
s_mean=np.mean(shift)
s_err=s_std/len(shift)
F=s_mean/w_mean
F_err=np.sqrt(((1/w_mean)*s_err)**2+(((s_mean/(w_mean)**2)*w_err))**2) 
#The means, standard deviations and standard errors on the mean are calculated and plugged into the
#error propagation formula for F
N=1.33
dN=0.005
L=532e-9    #All values and uncertainties assigned to variables
dL=0.5e-9
x=(F*L)/((N-1)*2) #x is found using the formula, divided by 2 to account for the fact that the light passed through 2 layers of the film
dx=np.sqrt(((L/(N-1))*F_err)**2+((F/(N-1))*dL)**2+(((F*L)/((N-1)**2))*dN)**2)
print("The value of F is",F,"+-",F_err)
print("The bubble's film thickness is",x,"+-",dx)
#%%
flame=pd.read_csv("C:/Users/Freddie/Data/Interferometry_2.csv") #Reads in the flame pixel measurement data
d_x_s=abs(flame["x2"]-flame["x1"]) #Finds the change in x and y
d_y_s=abs(flame["y2"]-flame["y1"])
shifts=np.sqrt(d_x_s**2+d_y_s**2)  #Finds the overall pixel distance
d_x_w=abs(flame["x2.1"]-flame["x1.1"]) #Finds the change in x and y
d_y_w=abs(flame["y2.1"]-flame["y1.1"])
widths=np.sqrt(d_x_w**2+d_y_w**2) #Finds the overall pixel distance
w_std_1=np.std(widths)
w_mean_1=np.mean(widths)
w_err_1=w_std_1/len(widths)
s_std_1=np.std(shifts)
s_mean_1=np.mean(shifts)
s_err_1=s_std_1/len(shifts)
F_1=s_mean_1/w_mean_1
F_err_1=np.sqrt(((1/w_mean_1)*s_err_1)**2+(((s_mean_1/(w_mean_1)**2)*w_err_1))**2)
x_1=0.6e-2
dx_1=0.5e-2
N_f=((F_1*L)/x_1)+1.0003
dN_f=np.sqrt(((F_1/x_1)*dL)**2+((L/x_1)*F_err_1)**2+(((F_1*L)/(x_1)**2)*dx_1)**2)
print("The value of F_1 is",F_1,"+-",F_err_1)
print("The refractive index of the flame is",N_f,"+-",dN_f)
