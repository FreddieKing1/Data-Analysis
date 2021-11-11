# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:01:38 2021

@author: Freddie
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
def i_output(V,f,n1,n2,e,t):
    m=-(V*n1*e)/(n2*f)
    cos=np.cos(f*t)
    return m*cos
f0=500
V0=10
n10=1000
n20=1
e0=8.55e-4
def dif_func(I,t):
    dI_dt=V0*np.cos(f0*t)
    return dI_dt
t=np.linspace(0,10,100)
soln=spi.odeint(dif_func,0,t)
y=i_output(V0,f0,n10,n20,e0,t)
plt.plot(t,soln,label="Primary Coil")
plt.plot(t,y,label="Secondary Coil")
plt.xlabel("Time/s")
plt.ylabel("Current/A")
plt.title("Primary and Secondary Currents with a Ferrite Core")
plt.legend()
plt.subplot(1,2,2)
def i_output(V,f,n1,n2,e,t):
    m=-(V*n1*e)/(n2*f)
    cos=np.cos(f*t)
    return m*cos
f0=500
V0=10
n10=1000
n20=1
e0=3.09e-5
def dif_func(I,t):
    dI_dt=V0*np.cos(f0*t)
    return dI_dt
t=np.linspace(0,10,100)
soln=spi.odeint(dif_func,0,t)
y=i_output(V0,f0,n10,n20,e0,t)
plt.plot(t,soln,label="Primary Coil")
plt.plot(t,y,label="Secondary Coil")
plt.xlabel("Time/s")
plt.ylabel("Current/A")
plt.title("Primary and Secondary Currents with no core")
plt.legend()
plt.show()

