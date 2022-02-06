# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 13:27:47 2022

@author: Freddie
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

amp_tim=pd.read_csv("C:/Users/Freddie/Data/Amplitude_Period_Data.csv")

print(amp_tim)
angle=amp_tim["Angle"]
a_err=amp_tim["Angle Uncertainty"]
period=amp_tim["Time Period (s)"]/0.97866
p_err=amp_tim["Time Uncertainty"]
plt.scatter(angle,period,color="red",label="Data")
plt.title("Amplitude against Time Period")
plt.xlabel("Time Period (s)")
plt.ylabel("Amplitude (Pixels)")
fit,cov=np.polyfit(angle,period,deg=2,cov=True,w=1/a_err)
fit_eq=np.poly1d(fit)
plt.plot(angle,fit_eq(angle),label="Best Fit")
plt.grid()
plt.legend()
plt.show()
print("The value of a is",fit[0],"+-",np.sqrt(cov[0][0]))
print("The value of b is",fit[1],"+-",np.sqrt(cov[1][1]))
print("The value of c is",fit[2],"+-",np.sqrt(cov[2][2]))
#%%
time=pd.read_csv("C:/Users/Freddie/Data/Pend_Mass_Period.csv")
mass=pd.read_csv("C:/Users/Freddie/Data/Masses_Pendulum.csv")

al_p=time["Al Period"]
al_err=time["Al Uncertainty"]
ny_p=time["Ny Period"]
ny_err=time["Ny Uncertainty"]
br_p=time["Br Period"]
br_err=time["Br Uncertainty"]
tu_p=time["Tu Period"]
tu_err=time["Tu Uncertainty"]
print(mass)
masses=mass["Mass (g)"]*(1/1000)
root_mass=[]
for m in masses:
    root_mass.append(1/m)
    
al=np.mean(al_p)
al_u=np.sqrt((al_err[0])**2+(al_err[1])**2+(al_err[2])**2)
ny=np.mean(ny_p)
ny_u=np.sqrt((ny_err[0])**2+(ny_err[1])**2+(ny_err[2])**2)
br=np.mean(br_p)
br_u=np.sqrt((br_err[0])**2+(br_err[1])**2+(br_err[2])**2)
tu=np.mean(tu_p)
tu_u=np.sqrt((tu_err[0])**2+(tu_err[1])**2+(tu_err[2])**2)
periods=[br**2,al**2,tu**2,ny**2]

plt.scatter(root_mass,periods,color="red",label="Data")
weights=[0.5/br_u,0.5/al_u,0.5/tu_u,0.5/ny_u]
fit_1,cov_1=np.polyfit(root_mass,periods,deg=1,cov=True,w=weights)
fit_eq_1=np.poly1d(fit_1)
plt.plot(root_mass,fit_eq_1(root_mass),label="Best Fit")
plt.title("Mass of Pendulum Bob Against Time Period")
plt.ylabel("(Time Period)^2 (s^2)")
plt.xlabel("1/Mass (kg^-1)")
plt.grid()
plt.legend()
del_m=9.80665/(4*((np.pi)**2)*0.227)
del_m_u=np.sqrt((9.80665/(4*((np.pi)**2)*(0.227**2)))*0.003)**2+((1/(4*((np.pi)**2)*0.227))*np.sqrt(cov_1[0][0]))**2
print("The change in the ratio of gravitational and inertial mass is given by",fit_1[0],"+-",del_m_u,"multiplied by 1/gravitational mass")
