# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 10:02:09 2021

@author: Freddie
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.image as mpimg  #Imports every module that may be used in this analysis
import scipy as spi
from scipy.optimize import curve_fit

doub_sing_slit=pd.read_csv("C:/Users/Freddie/Data/Diffraction Labs.csv",skiprows=2) #Reads in the data that includes our measurements from the single and double slit experiments
doub_sing_slit=pd.DataFrame(doub_sing_slit)  #Converts the data to a pandas dataframe
single_160=doub_sing_slit.iloc[:4,:2]        #The next few lines split up the data into
double_160=doub_sing_slit.iloc[:4,3:5]       #the different focal lengths and slit numbers
single_500=doub_sing_slit.iloc[:3,7:9]  #It is decided that this part of the data will be removed as there are too few data points, and the method was inappropriate
single_160["Distance"]=single_160["Distance"]/1000      #These lines convert the data from 
double_160["Distance.1"]=double_160["Distance.1"]/1000  #millimeters into meters
single_500["Distance.2"]=single_500["Distance.2"]/1000
single_160["Distance"]=np.abs(single_160["Distance"]-single_160.iloc[0,1])      #This standardises the distance relative
double_160["Distance.1"]=np.abs(double_160["Distance.1"]-double_160.iloc[0,1])  #to the reference point
single_500["Distance.2"]=np.abs(single_500["Distance.2"]-single_500.iloc[0,0])
print(single_160)
print(single_500)
print(double_160)
plt.scatter(single_160.iloc[1:,0],single_160.iloc[1:,1],label="Single Slit, f=160mm")  #The data is plotted separately for the single and double slits
plt.scatter(double_160.iloc[1:,0],double_160.iloc[1:,1],label="Double Slit, f=160mm")  #The first data point is omitted since this is simply a reference with respect to the zero order
plt.grid()
plt.legend()
plt.title("Realative Distance Plotted Against the Order of Minima")  #This code creates the labels and graphical feature of the graph
plt.xlabel("Order of Minimum")
plt.ylabel("Distance (m)")
fit_sing,cov_sing=np.polyfit(single_160.iloc[1:,0],single_160.iloc[1:,1],1,cov=True) #Using the polyfit function, a linear fit is made for the data
fit_eq_sing=np.poly1d(fit_sing) 
plt.plot(single_160.iloc[1:,0],fit_eq_sing(single_160.iloc[1:,0])) #This plots the linear on the graph along with our data
fit_doub,cov_doub=np.polyfit(double_160.iloc[1:,0],double_160.iloc[1:,1],1,cov=True) #Using the polyfit function, a linear fit is made for the data
fit_eq_doub=np.poly1d(fit_doub) 
plt.plot(double_160.iloc[1:,0],fit_eq_doub(double_160.iloc[1:,0])) #This plots the linear on the graph along with our data
plt.show()
g_s=fit_sing[0]
g_d=fit_doub[0]
dg_s=np.sqrt(cov_sing[0][0])
dg_d=np.sqrt(cov_doub[0][0])
f=160e-3
df=0.5e-3    #This set of variables are used in the calculations of the slit widths and uncertainties
l=670e-9
dl=1e-9
a=(l*f)/g_s
d=(l*f)/g_d
da=np.sqrt(((f*dl)/g_s)**2+((l*df)/g_s)**2+((l*f*dg_s)/g_s**2)**2)    #This is the error propagation formula
dd=np.sqrt(((f*dl)/g_d)**2+((l*df)/g_d)**2+((l*f*dg_d)/g_d**2)**2)
print("Our value for a, the single slit width, is",a,"+-",da,"m")   #Prints the final values along with uncertainties
print("Our value for d, the double slit separation is",d,"+-",dd,"m")
#%%

sing_cmos=pd.read_csv("C:/Users/Freddie/Data/Values_Single_2.csv")
doub_cmos=pd.read_csv("C:/Users/Freddie/Data/Values_Double_2.csv")  #Reads in the data from the CMOS camera for the single and double slit
print(sing_cmos)
print(doub_cmos)
s_d=sing_cmos["Distance_(pixels)"]*5.203125e-6    #This assigns the distance and intensity columns to 
s_i=sing_cmos["Gray_Value"]                       #shorthand variables and converts the distance
d_d=doub_cmos["Distance_(pixels)"]*5.203125e-6    #from pixel length to meters
d_i=doub_cmos["Gray_Value"]
def single(x,a,i0,s):
    t=(np.pi*a*(x-s))/(l*f)   #The function for the general waveform produced by single slit diffraction
    sin=(np.sin(t))/t
    return i0*sin**2
fit,cov=curve_fit(single,s_d,s_i,p0=[a,6,0.0035],maxfev=1000)  #This finds the best fit curve
print(fit[0])
sing_cmos_fit=single(s_d,fit[0],fit[1],fit[2])
plt.plot(s_d,sing_cmos_fit)         #This plots the best fit curve alonside the actual data
plt.plot(s_d,s_i)
plt.show()
def double(x,a,i0,s,d):
    t=(np.pi*a*(x-s))/(l*f)   #The function for the general waveform produced by double slit diffraction
    sin=(np.sin(t))/t
    cos=(np.cos(np.pi*d*x))/(l*f)
    return 4*i0*(sin**2)*cos**2
fit_1,cov_1=curve_fit(double,d_d,d_i,p0=[a,6,0.0035,d],maxfev=1000)
doub_cmos_fit=double(d_d,fit_1[0],fit_1[1],fit_1[2],fit_1[3])
plt.plot(d_d,doub_cmos_fit)       #This plots the best fit curve alongside the actual data
plt.plot(d_d,d_i)
plt.show()
print("Using the CMOS camera, our estimate for a, the slit width is",fit[0],"and our estimate for d, the slit separation is",fit_1[3])
#%%

