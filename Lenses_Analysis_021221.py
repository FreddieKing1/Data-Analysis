# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:03:16 2021

@author: Freddie
"""
import numpy as np
import matplotlib.pyplot as plt   #Imports all the relevent modules
import pandas as pd
import scipy as sp
from scipy.optimize import curve_fit

lens_data=pd.read_csv("Lens Exp Data 30.11.21.csv") #Reads in the three datasets as one
print(lens_data)    #Prints the read-in data for inspection
exp_1=lens_data.iloc[:,:11] #Separates out the data from the first experiment
print(exp_1)
exp_2=lens_data.iloc[:2,12:17] #Separates out the data from the second experiment
print(exp_2)
exp_3=lens_data.iloc[:2,18:23] #Separates out the data from the third experiment
print(exp_3)

s=exp_1.iloc[:,0]
s_prime=exp_1.iloc[:,1]  #This extracts the lines of data we want to plot
plt.scatter(1/s,1/s_prime,label="Actual Data") #This plots 1/s against 1/s'
fit,cov=np.polyfit(1/s,1/s_prime,1,cov=True)
fit_line=np.poly1d(fit)
plt.plot(1/s,fit_line(1/s),label="Best Fit")   #Here, a best fit line is created and plotted alongside the data
plt.xlabel("1/s (m^-1)")
plt.ylabel("1/s' (m^-1)")
plt.title("A graph of 1/s against 1/s' to find the focal length")
plt.legend()   #This code customises our graph, adding a title, axis labels, a grid and a legend
plt.grid()
plt.show()
y_int=fit[1]
y_int_err=np.sqrt(cov[1][1])
f=1/y_int          #This calculates our focal length value and its uncertainty from the fit and covariance matrix
f_err=y_int_err/y_int
print("The value of f is",f*1000,"+-",f_err,"mm")

ho=0.024   #This was the height of the object
hi=exp_1["ho (ave)"] #These are the average heights of the images formed
d_h=hi/ho
plt.scatter(s,d_h,label="True Data")   #This plots d_h against s
def M(s,f):         #We define a function that is supposed to fit our curve according to the theory
    return f/(s-f)
fit,cov_1=curve_fit(M,s,d_h,p0=0.1)  #This finds the best fit line based off off of the theoretical equation
fit_eq=M(s,fit[0])                 #and plots it against the data
plt.plot(s,fit_eq,label="Best Fit")
plt.title("A graph of magnification against distance between the lens and light source")
plt.ylabel("Magnification (ho/hi)")
plt.xlabel("Distance (m)")     #Here, we cutomise the graph similar to before
plt.grid()
plt.legend()
plt.show()
f_1=fit[0]
print("Our second value for f is",f_1*1000,"+-",np.sqrt(cov_1[0][0]),"mm")


