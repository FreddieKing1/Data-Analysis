# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import pandas as pd                #Imports all the relevant modules
import numpy as np

rydberg=pd.read_csv(r"C:\Users\Freddie\Data\RydbergDATA.csv",skiprows=1) #Reads in the data set, assigns to variable 'rydberg'
d=1.25e-5                      #Gives the slit width from our diffraction grating with 
red_80=rydberg.iloc[:,1:4]
aqua=rydberg.iloc[:,6:9]
purple=rydberg.iloc[:,11:14]
red_300=rydberg.iloc[:4,17:20] #Assigns each colour to a different dataset
red_80_ang=2*np.pi*(((red_80.iloc[1:9,1]+red_80.iloc[1:9,2]/60)-169.95)/360)
red_300_ang=2*np.pi*(((red_300.iloc[1:9,1]+red_300.iloc[1:9,2]/60)-169.95)/360)
aqua_ang=2*np.pi*(((aqua.iloc[1:9,1]+(aqua.iloc[1:9,2])/60)-169.95)/360)
purple_ang=2*np.pi*(((purple.iloc[1:7,1]+purple.iloc[1:7,2]/60)-169.95)/360)  #Converts all the angles to radians and separates them into separate datasets
red_80_ord=red_80.iloc[1:,0]
red_300_ord=red_300.iloc[1:,0]
aqua_ord=aqua.iloc[1:,0]
purple_ord=purple.iloc[1:7,0]   #Assigns the order numbers for each colour to different datasets

plt.scatter(red_80_ord,np.sin(red_80_ang),color="red",label="Data")
fit_r,cov_r=np.polyfit(red_80_ord,np.sin(red_80_ang),1,cov=True)
fit_eq_r=np.poly1d(fit_r)
plt.plot(red_80_ord,fit_eq_r(red_80_ord),color="blue",label="Best Fit")
plt.title("A Graph of Order Against sin of the Angle to the Zeroth Order for the Red fringes (80 lpm)")
plt.ylabel("sin(Angle (Radians))")
plt.xlabel("Order")
grad_r=fit_r[0]
wvl_r=grad_r*1.25e-5         #This plots a graph of order against sin(angle) for the red fringes and find the linear best fit 
err_r=np.sqrt(cov_r[0][0])   #The wavelength is then given by the best fit gradient times the slit width d
plt.grid()
plt.legend()
plt.show()
print(wvl_r,err_r)

plt.scatter(red_300_ord,np.sin(red_300_ang),color="red",label="Data")
fit_r2,cov_r2=np.polyfit(red_300_ord,np.sin(red_300_ang),1,cov=True)
fit_eq_r2=np.poly1d(fit_r2)
plt.plot(red_300_ord,fit_eq_r2(red_300_ord),color="blue",label="Best Fit")
plt.title("A Graph of Order Against sin of the Angle to the Zeroth Order for the Red fringes (300 lpm)")
plt.ylabel("sin(Angle (Radians))")
plt.xlabel("Order")
grad_r2=fit_r2[0]
wvl_r2=grad_r2*1.25e-5         #Same as above is applied to the red fringes at 300 lpm 
err_r2=np.sqrt(cov_r2[0][0])   
plt.grid()
plt.legend()
plt.show()                     
print(wvl_r2,err_r2)

plt.scatter(aqua_ord,np.sin(aqua_ang),color="red",label="Data")
fit_a,cov_a=np.polyfit(aqua_ord,np.sin(aqua_ang),1,cov=True)
fit_eq_a=np.poly1d(fit_a)
plt.plot(aqua_ord,fit_eq_a(aqua_ord),color="blue",label="Best Fit")
plt.title("A Graph of Order Against sin of the Angle to the Zeroth Order for the Aqua fringes")
plt.ylabel("sin(Angle (Radians))")
plt.xlabel("Order")               #Same as above is applied to the aqua fringes
grad_a=fit_a[0]
wvl_a=grad_a*1.25e-5
err_a=np.sqrt(cov_a[0][0])
plt.grid()
plt.legend()
plt.show()
print(wvl_a,err_a)

plt.scatter(purple_ord,np.sin(purple_ang),color="red",label="Data")
fit_p,cov_p=np.polyfit(purple_ord,np.sin(purple_ang),1,cov=True)
fit_eq_p=np.poly1d(fit_p)
plt.plot(purple_ord,fit_eq_p(purple_ord),color="blue",label="Best Fit")
plt.title("A Graph of Order Against sin of the Angle to the Zeroth Order for the Purple fringes")
plt.ylabel("sin(Angle (Radians))")
plt.xlabel("Order")               #Same as above is applied to the purple fringes
grad_p=fit_p[0]
wvl_p=grad_p*1.25e-5
err_p=np.sqrt(cov_p[0][0])
plt.grid()
plt.legend()
plt.show()
print(wvl_p,err_p)

recip_wvl=[1/wvl_r,1/wvl_a,1/wvl_p]
shells=[-5/36,-3/16,-21/100]             #This plots the reciprocal of the wavelengths against the difference in the reciprocals of the squares of the change in energy levels 
weights=[1/(err_r)**2,1/(err_a)**2,1/(err_p)**2]
plt.scatter(shells,recip_wvl,color="red",label="Data")
fit_R,cov_R=np.polyfit(shells,recip_wvl,deg=1,w=weights,cov=True) #The gradient of the graph is the negative of the Rydberg constant
fit_eq_R=np.poly1d(fit_R)
plt.plot(shells,fit_eq_R(shells),color="blue",label="Best Weighted Fit")
plt.title("A graph of 1/wavelength against the difference between the square reciprocals of the energy levels")
plt.ylabel("1/n1^2-1/n2^2")
plt.xlabel("1/wavelength")
plt.grid()
plt.legend()
plt.show() 
print("Our estimate for the Rydberg constant is",-fit_R[0],"+-",np.sqrt(cov_R[0][0]))