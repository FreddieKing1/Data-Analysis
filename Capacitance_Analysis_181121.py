# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 10:03:35 2021

@author: Freddie
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
 #Imports all relevant modules
electro=pd.read_csv("C:/Users/Freddie/Data/largeCV.csv")
ceram_1=pd.read_csv("C:/Users/Freddie/Data/CERAMIC.CSV")
ceram_2=pd.read_csv("C:/Users/Freddie/Data/Phase Shift.csv")
#Reads in the data for the electrolytic and ceramic capacitors
electro=pd.DataFrame(electro)
ceram_1=pd.DataFrame(ceram_1)
ceram_2=pd.DataFrame(ceram_2)
print(electro)
#Converts all the data to datframes to make them easier to manipulate
plt.plot(electro.iloc[:,0],electro.iloc[:,1])
plt.title("Initial Check")
plt.show()
#Initial plot of the voltage against time to ensure we have the correct form
V_0=max(electro.iloc[:,1])        #This finds the initial voltage
print("The value for V_0 is", V_0)
y_t=[]                       #Initialises empty lists for the values of voltage and time
t_t=[]                     
y=np.log(electro.iloc[:,1])
t=electro.iloc[:,0]            
for m in range(0,len(y)):      #Iterates through the logs of the voltages, since there is 
    if y[m]>=-1:               #massive variation in the later values which distorts our best fit
        y_t.append(y[m])       #line, we will only consider values greater or equal to -1 to reduce 
        t_t.append(t[m])       #uncertainty.
plt.plot(t_t,y_t,label="Actual Data")
fit,cov=np.polyfit(t_t,y_t,1,cov=True)
fit_eq=np.poly1d(fit)                                     #This code makes a graph with a title,
plt.plot(t_t,fit_eq(t_t),color="red",label="Best Fit")    #x and y axis labels, a grid, and a legend
plt.title("Log graph of voltage against time")            #which distinguishes the real data from the
plt.xlabel("Time (s)")                                    #fit line.
plt.ylabel("Voltage (V)")
plt.grid()
plt.legend()
plt.show()
uncertainty=np.sqrt(cov[0][0])      #The square root of the first entry in the covariance matrix
grad=fit[0]                         #gives a good estimate of the uncertainty
resistance=10000                    #The resistor used in the circuit had a value of 10 kiloohms
capacitance=-1/(grad*resistance)
first_error=np.sqrt((1/((grad**4)*(resistance**2)))*(uncertainty**2)+(1/((grad**4)*(resistance**2)))*(0.1*(resistance**2)))    #This calculates the capacitance
print("The capacitance is",capacitance,"+-",first_error,"Farads" )
#%%
print(ceram_1)
plt.plot(ceram_1.iloc[:,0],ceram_1.iloc[:,1]) #Plots an initial voltage against time graph to ensure
plt.title("Initial Check")                    #the form is as expected
plt.show()
y_1=ceram_1.iloc[:,1]  #This assigns the time data to t_1 and the voltage data to y_1
t_1=ceram_1.iloc[:,0]
y_t_1=[]
t_t_1=[]
for m in range(0,len(t_1)-1):  #This appends all the data between t=0 and t=0.0005 to new lists
    if 0<=t_1[m]<0.0005:       #so that we can isolate the first charging period of the capacitor
        y_t_1.append(y_1[m]) 
        t_t_1.append(t_1[m])
plt.plot(t_t_1,y_t_1)       #The first charging period is plotted.
plt.title("Ceramic Capacitor Charge 1")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.show()
y_t_11=[]                       #Initialises empty lists for the values of voltage and time
t_t_11=[]                     
y_log_1=np.log(max(y_t_1)+0.0000001-y_t_1) #Takes the log but adds a tiny value to prevent 0 error.
t=electro.iloc[:,0]            
for m in range(0,len(y_t_1)):      #Iterates through the logs of the voltages, since there is 
    if y_log_1[m]>=-1:               #massive variation in the later values which distorts our best fit
        y_t_11.append(y_log_1[m])       #line, we will only consider values greater or equal to -1 to reduce 
        t_t_11.append(t_t_1[m])       #uncertainty.
plt.plot(t_t_11,y_t_11,label="Actual Data")
fit_1,cov_1=np.polyfit(t_t_11,y_t_11,1,cov=True)
fit_eq_1=np.poly1d(fit_1)                                     #This code makes a graph with a title,
plt.plot(t_t_11,fit_eq_1(t_t_11),color="red",label="Best Fit")    #x and y axis labels, a grid, and a legend
plt.title("Log graph of voltage against time")            #which distinguishes the real data from the
plt.xlabel("Time (s)")                                    #fit line.
plt.ylabel("Voltage (V)")
plt.grid()
plt.legend()
plt.show()
uncertainty_1=np.sqrt(cov_1[0][0])      #The square root of the first entry in the covariance matrix
grad_1=fit_1[0]                         #gives a good estimate of the uncertainty
resistance_1=100000                   #The resistor used in the circuit had a value of 100 kiloohms
capacitance_1=-1/(grad_1*resistance_1)
error_1=np.sqrt((1/((grad_1**4)*(resistance_1**2)))*(uncertainty_1**2)+(1/((grad_1**4)*(resistance_1**2)))*(0.1*(resistance_1**2)))
print("The capacitance is",capacitance_1,"+-",error_1,"Farads") 
y_t_2=[]                     
t_t_2=[]                      
for m in range(0,len(t_1)-1):  #This appends all the data between t=0.001 and t=0.0015 to new lists
    if 0.001<=t_1[m]<0.0015:    #so that we can isolate the second charging period of the capacitor.
        y_t_2.append(y_1[m]) 
        t_t_2.append(t_1[m])
plt.plot(t_t_2,y_t_2)       #The second charging period is plotted.
plt.title("Ceramic Capacitor Charge 1")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.show()
y_t_22=[]                       #Initialises empty lists for the values of voltage and time
t_t_22=[]                     
y_log_2=np.log(max(y_t_2)+0.0000001-y_t_2) #Takes the log but adds a tiny value to prevent 0 error.
t=electro.iloc[:,0]            
for m in range(0,len(y_t_2)):      #Iterates through the logs of the voltages, since there is 
    if y_log_2[m]>=-1:               #massive variation in the later values which distorts our best fit
        y_t_22.append(y_log_2[m])       #line, we will only consider values greater or equal to -1 to reduce 
        t_t_22.append(t_t_2[m])       #uncertainty.
plt.plot(t_t_22,y_t_22,label="Actual Data")
fit_2,cov_2=np.polyfit(t_t_22,y_t_22,1,cov=True)
fit_eq_2=np.poly1d(fit_2)                                     #This code makes a graph with a title,
plt.plot(t_t_22,fit_eq_2(t_t_22),color="red",label="Best Fit")    #x and y axis labels, a grid, and a legend
plt.title("Log graph of voltage against time")            #which distinguishes the real data from the
plt.xlabel("Time (s)")                                    #fit line.
plt.ylabel("Voltage (V)")
plt.grid()
plt.legend()
plt.show()
uncertainty_2=np.sqrt(cov_2[0][0])      #The square root of the first entry in the covariance matrix
grad_2=fit_2[0]                         #gives a good estimate of the uncertainty
resistance_2=100000                   #The resistor used in the circuit had a value of 100 kiloohms
capacitance_2=-1/(grad_2*resistance_2)
error_2=np.sqrt((1/((grad_2**4)*(resistance_2**2)))*(uncertainty_2**2)+(1/((grad_2**4)*(resistance_2**2)))*(0.1*(resistance_2**2)))
print("The capacitance is",capacitance_2,"+-",error_2,"Farads") 
y_t_3=[]                     
t_t_3=[]
for m in range(0,len(t_1)-1):  #This appends all the data between t=0.001 and t=0.0015 to new lists
    if 0.0005<=t_1[m]<0.001:    #so that we can isolate the second charging period of the capacitor.
        y_t_3.append(y_1[m]) 
        t_t_3.append(t_1[m])
plt.plot(t_t_3,y_t_3)       #The second charging period is plotted.
plt.title("Ceramic Capacitor Discharge 1")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.show()
y_t_33=[]                       #Initialises empty lists for the values of voltage and time
t_t_33=[]                     
y_log_3=np.log(y_t_3) #Takes the log but adds a tiny value to prevent 0 error.
t=electro.iloc[:,0]            
for m in range(0,len(y_t_3)):      #Iterates through the logs of the voltages, since there is 
    if y_log_3[m]>=-1:               #massive variation in the later values which distorts our best fit
        y_t_33.append(y_log_3[m])       #line, we will only consider values greater or equal to -1 to reduce 
        t_t_33.append(t_t_3[m])       #uncertainty.
plt.plot(t_t_33,y_t_33,label="Actual Data")
fit_3,cov_3=np.polyfit(t_t_33,y_t_33,1,cov=True)
fit_eq_3=np.poly1d(fit_3)                                     #This code makes a graph with a title,
plt.plot(t_t_33,fit_eq_3(t_t_33),color="red",label="Best Fit")    #x and y axis labels, a grid, and a legend
plt.title("Log graph of voltage against time")            #which distinguishes the real data from the
plt.xlabel("Time (s)")                                    #fit line.
plt.ylabel("Voltage (V)")
plt.grid()
plt.legend()
plt.show()
uncertainty_3=np.sqrt(cov_3[0][0])      #The square root of the first entry in the covariance matrix
grad_3=fit_3[0]                         #gives a good estimate of the uncertainty
resistance_3=100000                   #The resistor used in the circuit had a value of 100 kiloohms
capacitance_3=-1/(grad_3*resistance_3)
error_3=np.sqrt((1/((grad_3**4)*(resistance_3**2)))*(uncertainty_3**2)+(1/((grad_3**4)*(resistance_3**2)))*(0.1*(resistance_3**2)))
print("The capacitance is",capacitance_3,"+-",error_3,"Farads")
y_t_4=[]                     
t_t_4=[]
for m in range(0,len(t_1)-1):  #This appends all the data between t=0.001 and t=0.0015 to new lists
    if 0.0015<=t_1[m]<0.002:    #so that we can isolate the second charging period of the capacitor.
        y_t_4.append(y_1[m]) 
        t_t_4.append(t_1[m])
plt.plot(t_t_4,y_t_4)       #The second charging period is plotted.
plt.title("Ceramic Capacitor Discharge 2")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.show()
y_t_44=[]                       #Initialises empty lists for the values of voltage and time
t_t_44=[]                     
y_log_4=np.log(y_t_4) #Takes the log but adds a tiny value to prevent 0 error.
t=electro.iloc[:,0]            
for m in range(0,len(y_t_4)):      #Iterates through the logs of the voltages, since there is 
    if y_log_4[m]>=-1:               #massive variation in the later values which distorts our best fit
        y_t_44.append(y_log_4[m])       #line, we will only consider values greater or equal to -1 to reduce 
        t_t_44.append(t_t_4[m])       #uncertainty.
plt.plot(t_t_44,y_t_44,label="Actual Data")
fit_4,cov_4=np.polyfit(t_t_44,y_t_44,1,cov=True)
fit_eq_4=np.poly1d(fit_4)                                     #This code makes a graph with a title,
plt.plot(t_t_44,fit_eq_4(t_t_44),color="red",label="Best Fit")    #x and y axis labels, a grid, and a legend
plt.title("Log graph of voltage against time")            #which distinguishes the real data from the
plt.xlabel("Time (s)")                                    #fit line.
plt.ylabel("Voltage (V)")
plt.grid()
plt.legend()
plt.show()
uncertainty_4=np.sqrt(cov_4[0][0])      #The square root of the first entry in the covariance matrix
grad_4=fit_4[0]                         #gives a good estimate of the uncertainty
resistance_4=100000                   #The resistor used in the circuit had a value of 100 kiloohms
capacitance_4=-1/(grad_4*resistance_4)
error_4=np.sqrt((1/((grad_4**4)*(resistance_4**2)))*(uncertainty_4**2)+(1/((grad_4**4)*(resistance_4**2)))*(0.1*(resistance_4**2)))
print("The capacitance is",capacitance_4,"+-",error_4,"Farads")
mean=(capacitance_3+capacitance_4)/2            #Takes the mean of the two discharge results.
final_err=0.25*np.sqrt(error_3**2+error_4**2)   #We discounted the charge results since the values were quite different and the uncertainties were large.
print("Our overall estimate for the capacitance of this capacitor is",mean,"+-",final_err)
#%%
ceram_2["α (Degrees)"]=ceram_2["α (Degrees)"]*(3.14159265358979/180) #Converts the angles in the data to radians
print(ceram_2)
capacitance=[]
uncertainties=[]                               #Initialises the empty lists for the capacitances and uncertainties
for i in range(0,len(ceram_2.iloc[:,0])):
    v_g=ceram_2.iloc[i,0]
    v_x=ceram_2.iloc[i,1]
    alpha=ceram_2.iloc[i,2]          #This iterates through the ceram_2 dataset and calculates the value
    vg_std=ceram_2.iloc[i,3]         #of capacitance for each row using the formulas given in the lab book.
    vx_std=ceram_2.iloc[i,4]
    alpha_std=ceram_2.iloc[i,5]
    R_1=6800    #These are the known values of the resistance and driving frequency
    d_freq=50000
    V_R1=np.sqrt((v_g*np.cos(alpha)-v_x)**2+(v_g*np.sin(alpha))**2)
    i_g=V_R1/R_1
    phi=np.arccos((v_g*np.sin(alpha))/V_R1)   
    x_c=v_x/(i_g*np.cos(phi))                  
    c_t=1/(2*np.pi*x_c*d_freq)
    capacitance.append(c_t)
    tm_1=(((np.sin(alpha))**2)/(4*(np.pi**2)*(d_freq**2)*(v_x**2)*(R_1**2)))*(vg_std**2)                 #This chunk of code uses the error propagation formula
    tm_2=(((v_g**2)*((np.cos(alpha))**2))/(4*(np.pi**2)*(d_freq**2)*(v_x**2)*(R_1**2)))*(alpha_std**2)   #shown in the lab book to calculate uncertainties
    tm_3=((v_g**2)*((np.sin(alpha))**2))/(4*((np.pi)**2)*(d_freq**2)*(v_x**4)*(R_1**2))*(vx_std**2)
    err=np.sqrt(tm_1+tm_2+tm_3)
    uncertainties.append(err)
mean=np.mean(capacitance)   #This takes the mean of the capacitances to find the estimate
final_error=0.25*np.sqrt(uncertainties[0]**2+uncertainties[1]**2+uncertainties[2]**2+uncertainties[3]**2)  #This combines the uncertainty terms in quadrature
print("Our final value for the capacitance of the ceramic capacitance is",mean-20e-12,"+-",final_error) 
