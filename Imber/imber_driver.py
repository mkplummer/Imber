# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 15:35:43 2022

@author: plummer.323
"""
get_ipython().run_line_magic('matplotlib', 'qt')
import matplotlib.pyplot as plt
import numpy as np

import instrument_list
import target_list
import Imber

from SecretColors import Palette
p = Palette("ibm") # 'ibm' 'brewer' 'material' 'clarity

import matplotlib.gridspec as gridspec

'Import target & instrument'
target1 = target_list.VHS1256b_MODHIS
instrument = instrument_list.MODHIS

'Set Spots (Truth)'
lat,lon,radius,contrast = [30],[30],[30],[0.25]
# lat,lon,radius,contrast = [30,60],[-60,60],[30,30],[0.25,-0.25]
target1.set_spots(lat,lon,radius,contrast)

'Model Settings'
length_of_observation = target1.period
time_steps = 10
n =  51 # 51 (VHS1256b), 97 (BetaPicb), 57 (HR8799e), 203 (SIMP0136), 7 (TRAPPIST-1), 37 (HR8799d)
eps = 0.4 #Linear LDC
spectral_resolution = instrument.spectral_resolution

'Numerical Model'
model_numerical = Imber.Numerical_Model(target1,length_of_observation,time_steps,n,eps,spectral_resolution)
model_numerical.run()

'Spectroscopic Observation (Simulated)'
texp = int(length_of_observation*3600/time_steps)
# texp = int(length_of_observation*3600/time_steps)-60 # Exposure time [s], Assumes continuous obervation
nexp = 1 # Number of exposures
obs1 = Imber.Simulated_Observation(instrument,model_numerical,texp,nexp,host_star = False)
Lambda = 1 # Least Squares Deconvolution Regularization Parameter (higher >> smoother)
obs1.create_line_profile(Lambda)

'Analytical Model'
model_analytical = Imber.Analytical_Model(target1, length_of_observation, time_steps, n, eps, spectral_resolution)
model_analytical.create_line_profile()
model_analytical.rotational_plus_instrument_kernel()
model_analytical.create_lightcurve()

'Plots'
fig,ax = plt.subplots()
plt.plot(obs1.line_profile[0],'r--')
plt.plot(model_numerical.line_profile[0],'b')
plt.plot(model_analytical.line_profile[0],'g')


'Photometric Observation (Simulated)'
texp = 60
nexp = 1
instrument_photo = instrument_list.METISimager
obs2 = Imber.Simulated_Observation(instrument_photo,model_numerical,texp,nexp,host_star = False)
obs2.create_lightcurve()


'Deviation Plots'
deviation_plots_on = False
if deviation_plots_on == True:
    'Input Inferred Values'
    lat,lon,radius,contrast = [28.26],[29.39],[30.38],[0.25]
    target1.set_spots(lat,lon,radius,contrast)
    'Recompute Analytical Model'
    model_analytical = Imber.Analytical_Model(target1, length_of_observation, time_steps, n, eps, spectral_resolution)
    model_analytical.create_line_profile()
    model_analytical.rotational_plus_instrument_kernel()
    model_analytical.rotational_broadening_kernel()
    'Deviation Plot Calculations'
    LPanalytical = model_analytical.line_profile
    Zstd = model_analytical.LPrb_plus_instrument
    obs_dev = np.zeros((time_steps,n)) #Empty 2D Array
    for i in range(time_steps): #Filling the 2D Array
        # temp_dev = np.std(Zobs[i,:])
        obs_dev[i] = (obs1.line_profile[i,:]-Zstd)#/temp_dev 
    model_dev = np.zeros((time_steps,n)) #Empty 2D Array
    for i in range(time_steps): #Fill the 2D Array
        model_dev[i] = (LPanalytical[i,:]-Zstd)#/temp_dev 
    residual = obs_dev-model_dev
    
    'Deviation Plots'
    color = 'inferno'
    # color = cmaps.teal
    extent = [model_analytical.rv_array[0],model_analytical.rv_array[n-1],360,0,]
    fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(10,8),sharex=True,sharey=True)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    axarr[0].imshow(obs_dev,cmap = color,extent = extent)
    axarr[0].set_title('Simulated Observation',fontsize = 16)
    axarr[0].set_ylabel('Phase (deg)',fontsize = 16)
    axarr[0].set_xlabel('Radial Velocity (km/s)',fontsize = 16)
    # axarr[0].set_xticks([0,int(n/4),int(n/2),3*int(n/4),n])
    # axarr[0].set_xticklabels([-14,-10,-6,-2,2,6,10,14])
    # axarr[0].set_yticklabels([-45,0,45,90,135,180,225,270,315,360],fontsize = 12)
    axarr[1].imshow(model_dev,cmap = color,extent = extent)
    axarr[1].set_xlabel('Radial Velocity (km/s)',fontsize = 16)
    axarr[1].set_title('Inferred Model',fontsize = 16)
    axarr[2].imshow(residual,cmap = color,extent = extent)
    axarr[2].set_xlabel('Radial Velocity (km/s)',fontsize = 16)
    axarr[2].set_title('Residuals',fontsize = 16)
    
    'Residual Plot'
    color = p.aqua()
    fig,ax = plt.subplots()
    residual_array = residual.reshape(915,1)
    plt.hist(residual_array,25,color = color,label = 'Residuals')
    plt.xlabel('Residual Deviations',fontsize = 16)
    plt.ylabel('Counts (#)',fontsize = 16)
    
    from scipy.stats import norm
    import matplotlib.mlab as mlab
    (mu, sigma) = norm.fit(residual_array)
    print('New S/N =',1/sigma)
    x_axis = np.arange(-0.1,0.1, 0.0001)
    plt.plot(x_axis,0.65*norm.pdf(x_axis,mu,sigma),color = p.black(),linewidth = 2.3,label = 'Gaussian Fit')
    plt.xlim([-0.015,0.015])
    plt.legend()


'Setup Photometric Observation Class'
perform_nested_sampling = True
if perform_nested_sampling == True:
    truths = [lat,lon,radius,contrast]
    # truths = [lat[0],lon[0],radius[0],contrast[0],lat[1],lon[1],radius[1],contrast[1]]
    time_samples = 500
    inference = Imber.NestedSampling(target1, time_samples,spectral_observation = obs1,photo_observation = obs2,\
                 free_contrast = False,free_radius = True,sampling_mode='dynamic',\
                         truths = truths,evolution_spot_number = False,evolution_rotation = False)
    inference.run()








