# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:58:07 2022

@author: Freddie
"""

import pandas as pd
import numpy as np      #Imports all relevant modules
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

lcr_r1=pd.read_csv("C:/Users/Freddie/Data/LCR_Circuit_1ohm.csv")
lcr_r2=pd.read_csv("C:/Users/Freddie/Data/LCR_Circuit_2ohm.csv") #Reads in the data and assigns to variable LCR_r followed by the resistance of the resistor
lcr_r3=pd.read_csv("C:/Users/Freddie/Data/LCR_Circuit_3ohm.csv")
print(lcr_r1)
def resonance(f,N,w,y):
    root=np.sqrt(((f**2)-(w**2))**2+(y*f)**2)    #This is the resonance curve function that will be used for fitting
    return N/root

def resonance_r(f,w,y,r):
    t_1=(1+r)*(f**2-w**2)               #This is the resonance curve function that will be used for fitting, takes into account internal impedance
    t_2=f*y*r
    root=np.sqrt(t_1**2+t_2**2)
    t_3=2*1*w**2
    return t_3/root

def phase(f,w,a,d,y):
    top=-y*(f-a)                #This is the function for the phase curve which will be used for fitting
    bottom=w**2-(f-a)**2
    return np.arctan2(top,bottom)+d

def phase_r(f,w,a,d,y,r,R):
    top=-y*(f-a)*r                #This is the function for the phase curve which will be used for fitting, takes into account internal impedance
    bottom=(w**2-(f-a)**2)*(R+r)
    return -np.arctan2(top,bottom)+d

freq_1=lcr_r1["Frequency (Hz)"]*0.001*2*np.pi
print(freq_1)
amp_1=lcr_r1["Amplitude (mV)"]*0.001
amp_1_u=lcr_r1["Amplitude STD"]                     #Assigns each useful column to different variables
phase_1=-lcr_r1["Phase (degrees)"]*((np.pi)/180)   #Degrees is converted to radians
phase_1_u=lcr_r1["Phase STD"]
plt.scatter(freq_1,phase_1,label="Data",color="blue")
fit1,cov1=curve_fit(phase_r,freq_1,phase_1,sigma=1/phase_1_u,p0=[16394,16394,-1.5,1000,1,1],maxfev=10000)
fit_eq1=phase_r(freq_1,fit1[0],fit1[1],fit1[2],fit1[3],fit1[4],fit1[5])  #This creates an arctan fit to the phase data 
plt.plot(freq_1,fit_eq1,label="Fit",color="blue")
plt.title("Driving Frequency Against Signal Phase of Capacitor (R=1 ohm)")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Phase Difference (radians)")
plt.grid()               #The best fit is plotted alongside the data
plt.legend()             #We will use the best fit to find the resonant frequency
plt.show()
print(fit1[0],fit1[1],fit1[2],fit1[3],fit1[4],fit1[5])
print(np.sqrt(np.diag(cov1)))

#f_test=np.linspace(60000,120000,1000000)
#a_test=resonance_r(f_test,100000,1000,35)
#plt.plot(f_test,a_test)

plt.scatter(freq_1,amp_1,label="Data",color="blue")
fit2,cov2=curve_fit(resonance_r,freq_1,amp_1,sigma=1/amp_1_u,p0=[100,10,5])
fit_eq2=resonance_r(freq_1,fit2[0],fit2[1],fit2[2])
plt.plot(freq_1,fit_eq2,label="Fit",color="blue")
plt.title("Driving Frequency Against Signal Amplitude of Capacitor (R=1 ohm)")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Amplitude (V)")
plt.plot(freq_1,resonance_r(freq_1,100,10,29.295),label="Expected")
plt.legend()
plt.grid()  
plt.show()
print(fit2[0],fit2[1],fit2[2])
print(np.diag(cov2))

freq_2=lcr_r2["Frequency (Hz)"]*2*np.pi*0.001
amp_2=lcr_r2["Amplitude (mV)"]*0.001
amp_2_u=lcr_r2["Amplitude STD"]                     #Assigns each useful column to different variables
phase_2=-lcr_r2["Phase (degrees)"]*((np.pi)/180)   #Degrees is converted to radians
phase_2_u=lcr_r2["Phase STD"]
plt.scatter(freq_2,phase_2,label="Data",color="blue")
fit3,cov3=curve_fit(phase_r,freq_2,phase_2,sigma=1/phase_2_u,p0=[16394,16394,-1.5,1000,1,1],maxfev=10000)
fit_eq3=phase_r(freq_2,fit3[0],fit3[1],fit3[2],fit3[3],fit3[4],fit3[5])  #This creates an arctan fit to the phase data 
plt.plot(freq_2,fit_eq3,label="Fit",color="blue")
plt.title("Driving Frequency Against Signal Phase of Capacitor (R=2 ohm)")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Phase Difference (radians)")
plt.grid()               #The best fit is plotted alongside the data
plt.legend()             #We will use the best fit to find the resonant frequency
plt.show()
plt.scatter(freq_2,amp_2,label="Data",color="blue")
fit4,cov4=curve_fit(resonance_r,freq_2,amp_2,sigma=1/amp_2_u,p0=[103,10,2],maxfev=1000)
print(fit3[0],fit3[1],fit3[2],fit3[3],fit3[4],fit3[5])
print(np.sqrt(np.diag(cov3)))

#f_test=np.linspace(800,1200,1000000)
#a_test=resonance_r(f_test,1030,100,2)
#plt.plot(f_test,a_test)

fit_eq4=resonance_r(freq_2,fit4[0],fit4[1],fit4[2])
plt.plot(freq_2,fit_eq4,label="Fit",color="blue")
plt.title("Driving Frequency Against Signal Amplitude of Capacitor (R=2 ohm)")
plt.xlabel("Frequency (rads^-1)")
plt.ylabel("Amplitude (V)")
plt.legend()
plt.grid()
plt.show()
print(fit4[0],fit4[1],fit4[2])
print(np.diag(cov4))

freq_3=lcr_r3["Frequency (Hz)"]*2*np.pi*0.001
amp_3=lcr_r3["Amplitude (mV)"]*0.001
amp_3_u=lcr_r3["STD Amplitude"]                     #Assigns each useful column to different variables
phase_3=-lcr_r3["Phase (degrees)"]*((np.pi)/180)   #Degrees is converted to radians
phase_3_u=lcr_r3["STD Phase"]
plt.scatter(freq_3,phase_3,label="Data",color="blue")
fit5,cov5=curve_fit(phase_r,freq_3,phase_3,sigma=1/phase_3_u,p0=[16394,16394,-1.5,3000,1,1],maxfev=10000)
fit_eq5=phase_r(freq_3,fit5[0],fit5[1],fit5[2],fit5[3],fit5[4],fit5[5])  #This creates an arctan fit to the phase data 
plt.plot(freq_3,fit_eq5,label="Fit",color="blue")
plt.title("Driving Frequency Against Signal Phase of Capacitor (R=3 ohm)")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Phase Difference (radians)")
plt.grid()               #The best fit is plotted alongside the data
plt.legend()             #We will use the best fit to find the resonant frequency
plt.show()
print(fit5[0],fit5[1],fit5[2],fit5[3],fit5[4],fit5[5])
print(np.sqrt(np.diag(cov5)))

plt.scatter(freq_3,amp_3,label="Data",color="blue")
fit6,cov6=curve_fit(resonance_r,freq_3,amp_3,sigma=1/amp_3_u,p0=[103,10,1],maxfev=1000)
fit_eq6=resonance_r(freq_3,fit6[0],fit6[1],fit6[2])
plt.plot(freq_3,fit_eq6,label="Fit",color="blue")
plt.title("Driving Frequency Against Signal Amplitude of Capacitor (R=3 ohm)")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Amplitude (V)")
plt.legend()
plt.grid()
plt.show()
print(fit6[0],fit6[1],fit6[2])
print(np.diag(cov6))

print(fit2[0],fit4[0],fit6[0])

def res_r(f,w,R,r,y,V,d):
    t_1=(R+r)
    t_2=(f*y*r)/(f**2-w**2)
    root=np.sqrt(t_1**2+t_2**2)
    return (V*R)/root+d

v_in=lcr_r3["Input (mV)"]*0.001
v_in_u=lcr_r3["STD Input "]
plt.scatter(freq_3,v_in,label="Data",color="blue")
fit7,cov7=curve_fit(res_r,freq_3,v_in,sigma=1/v_in_u,p0=[103,3,1,300,1.91,0.08])
fit_eq7=res_r(freq_3,fit7[0],fit7[1],fit7[2],fit7[3],fit7[4],fit7[5])
plt.plot(freq_3,fit_eq7,color="blue",label="Fit")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Load Voltage (V)")
plt.title("Load Voltage Against Frequency")
plt.legend()
plt.grid()
plt.show()
print(fit7[0],fit7[1],fit7[2],fit7[3],fit7[4])
print((np.diag(cov7)))

print(fit2[0],fit4[0],fit6[0])
print(fit2[2],fit4[2],fit6[2],fit7[2])

int_r=[fit2[2],fit4[2],fit6[2]]
int_r_u=np.sqrt(cov2[2][2]+cov4[2][2]+cov6[2][2])
res_f=[fit2[0],fit4[0],fit6[0],fit7[0]]
res_f_u=np.sqrt(cov2[0][0]+cov4[0][0]+cov6[0][0])
#%%
plt.scatter(freq_1,phase_1,label="1 Ohm",color="blue")
fit1,cov1=curve_fit(phase_r,freq_1,phase_1,sigma=1/phase_1_u,p0=[16394,16394,-1.5,1000,1,1],maxfev=10000)
fit_eq1=phase_r(freq_1,fit1[0],fit1[1],fit1[2],fit1[3],fit1[4],fit1[5])  #This creates an arctan fit to the phase data 
plt.plot(freq_1,fit_eq1,label="1 Ohm Fit",color="blue")
plt.scatter(freq_2,phase_2,label="2 Ohm",color="red")
fit3,cov3=curve_fit(phase_r,freq_2,phase_2,sigma=1/phase_2_u,p0=[16394,16394,-1.5,1000,1,1],maxfev=10000)
fit_eq3=phase_r(freq_2,fit3[0],fit3[1],fit3[2],fit3[3],fit3[4],fit3[5])  #This creates an arctan fit to the phase data 
plt.plot(freq_2,fit_eq3,label="2 Ohm Fit",color="red")
plt.scatter(freq_3,phase_3,label="3 Ohm",color="purple")
fit5,cov5=curve_fit(phase_r,freq_3,phase_3,sigma=1/phase_3_u,p0=[16394,16394,-1.5,3000,1,1],maxfev=10000)
fit_eq5=phase_r(freq_3,fit5[0],fit5[1],fit5[2],fit5[3],fit5[4],fit5[5])  #This creates an arctan fit to the phase data 
plt.plot(freq_3,fit_eq5,label="3 Ohm Fit",color="purple")
plt.title("Driving Frequency Against Signal Phase of Capacitor")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Phase Difference (radians)")
plt.grid()               #The best fit is plotted alongside the data
plt.legend()
plt.show()

plt.scatter(freq_1,amp_1,label="1 Ohm",color="blue",s=10)
fit2,cov2=curve_fit(resonance_r,freq_1,amp_1,sigma=1/amp_1_u,p0=[100,10,5])
fit_eq2=resonance_r(freq_1,fit2[0],fit2[1],fit2[2])
plt.plot(freq_1,fit_eq2,label="1 Ohm Fit",color="blue")
plt.scatter(freq_2,amp_2,label="2 Ohm",color="purple",s=10)
fit4,cov4=curve_fit(resonance_r,freq_2,amp_2,sigma=1/amp_2_u,p0=[103,10,2],maxfev=1000)
fit_eq4=resonance_r(freq_2,fit4[0],fit4[1],fit4[2])
plt.plot(freq_2,fit_eq4,label="2 Ohm Fit",color="purple")
plt.scatter(freq_3,amp_3,label="3 Ohm",color="brown",s=10)
fit6,cov6=curve_fit(resonance_r,freq_3,amp_3,sigma=1/amp_3_u,p0=[103,10,1],maxfev=1000)
fit_eq6=resonance_r(freq_3,fit6[0],fit6[1],fit6[2])
plt.plot(freq_3,fit_eq6,label="3 Ohm Fit",color="brown")
plt.title("Driving Frequency Against Signal Amplitude of Capacitor")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Amplitude (V)")
plt.legend()
plt.grid() 

frequency=np.linspace(60,130,1000000)
max_1=resonance_r(res_f[0],fit2[0],fit2[1],fit2[2])
max_2=resonance_r(res_f[1],fit4[0],fit4[1],fit4[2])
max_3=resonance_r(res_f[2],fit6[0],fit6[1],fit6[2])
print(max_1,max_2,max_3)
f1=(max_1/(np.sqrt(2)))-0.00001
f2=((max_1/np.sqrt(2)))+0.00001
print(f1,f2)
delta_w_1=[]
n_1=0
for i in range(0,len(frequency)-1):
    if (f1<=resonance_r(frequency[n_1],fit2[0],fit2[1],fit2[2])<=f2):
        delta_w_1.append(frequency[n_1])
        n_1+=1
    else:
        n_1+=1
f3=(max_2/(np.sqrt(2)))-0.00001
f4=((max_2/np.sqrt(2)))+0.00001
delta_w_2=[]
n_2=0
for i in range(0,len(frequency)-1):
    if (f3<=resonance_r(frequency[n_2],fit4[0],fit4[1],fit4[2])<=f4):
        delta_w_2.append(frequency[n_2])
        n_2+=1
    else:
        n_2+=1
f5=(max_3/(np.sqrt(2)))-0.00001
f6=((max_3/np.sqrt(2)))+0.00001
delta_w_3=[]
n_3=0
for i in range(0,len(frequency)-1):
    if (f5<=resonance_r(frequency[n_3],fit6[0],fit6[1],fit6[2])<=f6):
        delta_w_3.append(frequency[n_3])
        n_3+=1
    else:
        n_3+=1
print(delta_w_1,delta_w_2,delta_w_3)    #We iterate through all the frequencies in order to find where the curve goes below 1/sqrt(2) of the maximum amplitude
print(max_3/np.sqrt(2))

dw1=max(delta_w_1)-min(delta_w_1)
dw2=max(delta_w_2)-min(delta_w_2)
dw3=max(delta_w_3)-min(delta_w_3)
print(dw1,dw2,dw3)
Q=[res_f[0]/dw1,res_f[1]/dw2,res_f[2]/dw3]
Q_u=[res_f[0]*(np.sqrt(cov2[0][0])/(dw1)**2),res_f[1]*(np.sqrt(cov4[0][0])/(dw2)**2),res_f[2]*(np.sqrt(cov6[0][0])/(dw3)**2)]
print("The Q factors for each circuit are",Q,"+-",Q_u,"respectively")
print("From our set of experiments we can deduce that the internal resistance is, on average,",np.mean(int_r),"+-",int_r_u,"Ohms")
print("We can also deduce that the average natural frequency across our experiments was",np.mean(res_f),"+-",res_f_u,"kHz")
print(max_1,max_2,max_3)