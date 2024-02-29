#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:26:28 2022

@author: plummer.323
"""
import numpy as np
import scipy
from astropy.io import fits
from astropy import units as u
from specutils.fitting import fit_generic_continuum
from specutils.spectra import Spectrum1D, SpectralRegion
import matplotlib.pyplot as plt
import pandas as pd
import dynesty
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot

from tqdm import tqdm

plate_scale = 206265
c = 2.9979E5

class Target:
    ''' Base object for Doppler Imaging Code. Each target (star, brown dwarf, exoplanet)
    is created as a class. 
    
    Inputs:
        vsini: Rotational velocity as a function of inclination [km/s]"
        vstar: Radial velocity of star system w.r.t. solar system [km/s], Default to 0"
        inclination: Orbital axis of target w.r.t. solar system [deg]"
        period: Target rotational period [hrs]
    '''
    
    def __init__(self,vsini,inclination,period,vstar = 0):
        " Here we instantiate the target"
        self.vsini = vsini
        self.inclination = inclination
        self.period = period
        self.vstar = vstar
        self.spot_evolution_boolean = False
        self.prior_rotation = False
        
    def set_spots(self,lat,lon,radius,contrast):
        ''' Here we add spots to the target. These spots are later added to the analytical
        and numerical line profiles'''
        self.lat = lat
        self.lon = lon
        self.radius = radius
        self.contrast = contrast
    
    def spot_evolution(self,spot_number,rotation,radius,contrast):
        self.spot_evolution_boolean = True
        self.spot_evolution_number = spot_number
        self.spot_evolution_rotation = rotation
        self.spot_evolution_radius = radius
        self.spot_evolution_contrast = contrast
        
    def observed_properties(self,spectra_file,apparent_magnitude,magnitude_band,zero,file_type = 'FITS',host_magnitude_band = 0,host_magnitude = 0):
        ''' Necessary inputs to use Observation-class Object
        
        Inputs:
            spectra_file: Spectrum filename
            file_type: File format (normally FITS or CSV)
            temperature: Only used if using the blackbody mode (usually for diagnostics)
            apparent_magnitude: Needs to match the associated magnitude band
            magnitude_band: Band corresponding to supplied apparent magnitude
            zero: Magnitude system zero (either AB or Vega)'''
        
        self.spectra_file = spectra_file
        self.apparent_magnitude = apparent_magnitude
        self.magnitude_band = magnitude_band
        self.zero = zero
        self.file_type = file_type
        self.host_magnitude_band = host_magnitude_band 
        self.host_magnitude = host_magnitude


        
class Simulated_Observation:
    '''
    Observation class is used to simulate synthetic light curves, spectra, and line profiles
    
    Required Parameters:
        texp: Exposure time [sec]
        nexp: Number of exposures [#]
        length_of_observations: Time period over which target is observed
        target: Target-class object
        telescope: Telescope-class object
        instrument: Instrument-class object
    '''
    
    def __init__(self,instrument,model,texp,nexp,host_star = False):
        self.target = model.target # Takes target from model
        self.instrument = instrument # Input Instrument-class object
        self.model = model
        self.texp = texp #[secs]
        self.nexp = nexp # number of exposures
        self.time_hours = np.linspace(0,model.length_of_observation,model.time_steps)
        self.host_star = host_star
        
    def create_lightcurve(self):
        if self.instrument.data_type == 'Photometric':
            self.lightcurve_generate()
        elif self.instrument.data_type == 'Spectroscopic':
            print('Need to select photometric instrument for light curve generation.')
    
    def create_line_profile(self,Lambda):
        self.n = self.model.n
        self.eps = self.model.eps
        self.spectral_resolution = self.model.spectral_resolution
        if self.instrument.data_type == 'Photometric':
            print('Need to select photometric instrument for light curve generation.')
        elif self.instrument.data_type == 'Spectroscopic':
            self.create_spectra(Lambda)
            
    def create_spectra(self,Lambda):
        ''' Creates spectra a line profiles based on selected target,
        regularization paramater, and broadening mechanisms.
        
        Inputs:
            Lambda: Regularization parameter to smooth LSD result
            broadening: Add instrument broadening by setting to 'instrument' 
        '''
     
        ''' Based on spectral band & target spectrum, create normalized spectra
        within desired band.
        '''
        observed_wavelength_temp,band_flux_temp = build_band_spectra(self.instrument.spectral_band,self.target,self.instrument.lambda_ref,normalize = True)
        self.observed_wavelength = observed_wavelength_temp
        self.band_flux = band_flux_temp
        ''' Generate numerical line profile.'''    
        self.model.run()
        Z = self.model.line_profile
        temp_spectra = np.zeros((self.model.time_steps,len(self.band_flux)))
        ''' Compute SNR and spectral noise based on target.'''
        self.compute_snr_spectral()
        # self.snr = 500*np.ones((len(self.band_flux)))
        noise_level = 1/self.snr
        noiseless_spectra = np.zeros((self.model.time_steps,len(self.band_flux)))
        noise = np.zeros((self.model.time_steps,len(self.band_flux)))
        observed_LP = np.zeros((self.model.time_steps,self.model.n))
        for k in tqdm(range(self.model.time_steps)):
            for j in range(len(self.observed_wavelength)):
                ''' Generate random Gaussian noise for each wavelength.'''
                noise_temp = np.random.normal(0,noise_level[j],1)
                if abs(noise_temp) > 1:
                    noise[k,j]= 1
                else:
                    noise[k,j] = noise_temp
            ''' Create simulated observed spectra by convolving template spectra  
                with numerically computed line profile and adding Gaussian noise.
            '''
            noiseless_spectra_temp = np.convolve(self.band_flux,Z[k,:],'same')
            noiseless_spectra[k] = noiseless_spectra_temp
            observed_spectra_temp = noiseless_spectra_temp+noise[k,:]
            temp_spectra[k] = observed_spectra_temp
            S = 0#make_S(sigma,len(self.band_flux))
            ''' subspectra_length limits wavelength range over which LSD is performed.'''
            subspectra_length = 10000
            if len(temp_spectra[k]) > subspectra_length:
                ''' Each subspectra is used to create a line profile. These subspectra are 
                weighted by how closely they match a rotationally broadened kernel. A
                weighted average is then computed.'''
                number_of_subspectra = int(len(self.observed_wavelength)/subspectra_length)+1
                Zobs = np.zeros((number_of_subspectra,self.model.n))
                weights = np.zeros(number_of_subspectra)
                start_index = 0
                for i in range(number_of_subspectra):
                    ''' Checks to see if subspectra length would exceed total length
                    of the spectra. If so, the end_index is set to the length of the 
                    entire spectra-1.'''
                    if start_index+subspectra_length<len(self.observed_wavelength):
                        end_index = start_index+int(subspectra_length)
                    else:
                        end_index = len(self.observed_wavelength)-1
                    Ztemp = leastSqDeconvolution(observed_spectra_temp[start_index:end_index],self.band_flux[start_index:end_index],S,self.model.n,Lambda)
                    weights[i] = 1/(sum(abs(Ztemp-Z[k,:])))
                    Zobs[i,:] = weights[i]*Ztemp
                    start_index = end_index
                observed_LP[k] = number_of_subspectra*np.mean(Zobs,axis = 0)/sum(weights)
            else:
                ''' LSD performed to find simulated observed line profiles.'''
                Ztemp = leastSqDeconvolution(observed_spectra_temp,self.band_flux,S,self.model.n,Lambda)
                observed_LP[k] = Ztemp
        self.noise = noise
        self.noiseless_spectra = noiseless_spectra
        self.observed_spectra = temp_spectra # Spectra with noise
        self.line_profile = observed_LP # Deconvolve observed line profile
        
    def compute_snr_spectral(self):
        ''' Compute S/N for each wavelength in instrument's spectral band
            based on target apparent_magnitude, band, and spectra.
        '''
        self.instrument.initial_calculations(self.target)
        apparent_magnitude = self.target.apparent_magnitude
        magnitude_band = self.target.magnitude_band
        zero = self.target.zero
        ''' Magnitude Bands corresponding to the apparent magnitude input.'''
        mbands = ['U','B','V','R','I','J','H','K','L','M','N','Q']
        mband_index = mbands.index(magnitude_band)
        if self.host_star == False:
            self.target.host_magnitude_band = 'J'
        mband_host_index = mbands.index(self.target.host_magnitude_band)
        central_wavelength_photo = [0.36,0.44,0.55,0.64,0.79,1.25,1.65,2.16,3.5,4.8,10.2,21.0] # [microns] (Liske ELT Spectroscopic ETC)
        dlambda = self.instrument.lambda_ref/self.instrument.spectral_resolution
        lambda_ref_photo = central_wavelength_photo[mband_index]
        lambda_ref_photo_host = central_wavelength_photo[mband_host_index]
        ''' Compute flux in photometric band and use that to find corresponding
            expected flux in instrument band. The ratio is used to adjust the 
            measured flux accordingly.
        '''
        # if self.mode == 'blackbody':
        #     self.instrument_wave,instrument_flux = blackbody(self.spectral_band,target.temperature,dlambda,lambda_ref)
        #     photometric_wave,photometric_flux = blackbody(target.magnitude_band,target.temperature,dlambda_photo,lambda_ref_photo)
        # else:
        photometric_wave,photometric_flux = build_band_spectra(magnitude_band,self.target,lambda_ref_photo)
        host_photometric_wave,host_photometric_flux = build_band_spectra(self.target.host_magnitude_band,self.target,lambda_ref_photo_host)
        integrated_instrument_flux = sum(self.instrument.instrument_wave*self.instrument.instrument_flux)/sum(self.instrument.instrument_wave)
        integrated_photometric_flux = sum(photometric_wave*photometric_flux)/sum(photometric_wave)
        integrated_host_photometric_flux = sum(host_photometric_wave*host_photometric_flux)/sum(host_photometric_wave)
        flux_ratio = integrated_instrument_flux/integrated_photometric_flux
        flux_ratio_host = integrated_instrument_flux/integrated_host_photometric_flux
        
        ''' Atmospheric Transmission & Sky Background Flux '''
        if zero == 'Vega':
            Zref = [7.38,7.18,7.44,7.64,7.91,8.51,8.94,9.40,10.32,10.69,11.91,13.17] # (Liske ELT Spectroscopic ETC)
        elif zero == 'AB':
            Zref = [7.09,7.25,7.44,7.58,7.77,8.14,8.39,8.64,9.04,9.32,10.00,10.57] #(Liske ELT Spectroscopic ETC)
        else:
            print('Did not input valid photometric zero.')
            exit()
        Z = Zref[mband_index]
        Zhost = Zref[mband_host_index]
        F0 = 10**(-Z)
        F0host = 10**(-Zhost)
        target_flux = F0*10**(-0.4*apparent_magnitude)*flux_ratio
        host_flux = F0host*10**(-0.4*self.target.host_magnitude)*flux_ratio_host
        if self.instrument.spectral_band == 'GCLEF-Red':
            self.target.Iband_apparent_magnitude = -2.5*np.log10(target_flux/(10**(-7.91)))
        encircled_energy = self.instrument.encircled_energy*0.01
        E_gamma = (1.985E-19)/self.instrument.lambda_ref #Photon Energy [J/gamma]
        conversion_factor = self.instrument.efficiency*dlambda*self.instrument.collecting_area/E_gamma
        ''' Computing Signal Flux'''
        Fobj_hat = self.instrument.atm_transmission*target_flux
        Fhost_hat = self.instrument.atm_transmission*host_flux
        ''' Number of electrons from objec in S/N ref area per exposure '''
        Nobj = Fobj_hat*encircled_energy*conversion_factor*self.texp
        'Compute Host Electrons'
        Nhost = Fhost_hat*encircled_energy*conversion_factor*self.texp
        ''' Computing Noise '''
        sky_solid_angle = np.pi*(self.instrument.Rref**2)/(10**6) # Omega in Liske ETC
        Npix = 2*self.instrument.Nspec #Nyquist Criterion
        Nsky = self.instrument.Fsky*self.texp*dlambda*self.instrument.collecting_area*sky_solid_angle
        if self.host_star == False:
            starlight_suppression = 0
        else:
            starlight_suppression = self.instrument.starlight_suppression
        self.snr = (self.nexp**0.5)*Nobj/((Nobj+starlight_suppression*Nhost+Nsky+\
                                           Npix*(self.instrument.readout_noise**2)+Npix*self.instrument.dark_current*self.texp)**0.5)
        # fig, ax = plt.subplots()
        # plt.plot(self.instrument_wave,self.snr)
        # plt.xlabel('Wavelength [microns]')
        # plt.ylabel('S/N')
        return(self.snr)
    
    def lightcurve_generate(self):
        'Creates photometric light curve w/ noise based on S/N'
        self.snr = self.compute_snr_photo()
        self.model.run(noiseon=True,sigma = 1/self.snr)
    
    def compute_snr_photo(self):
        ' Compute sky aperture per pixel'
        omega_pixel = self.instrument.pixel_scale**2 # [mas^2/pixel]

        ' Size of S/N reference area'
        omega = (10**(-6))*self.instrument.Npix*omega_pixel

        ' Magnitude Bands corresponding to the apparent magnitude input.'
        mbands = ['U','B','V','R','I','J','H','K','L','M','N','Q']
        ibands = ['U','B','V','R','I','J','H','K','L','M','N','Q']
        mband_index = mbands.index(self.target.magnitude_band)
        if self.host_star == False:
            self.target.host_magnitude_band = 'J'
        mband_host_index = mbands.index(self.target.host_magnitude_band)
        iband_index = ibands.index(self.instrument.instrument_band)
        central_wavelength_photo_array = [0.36,0.44,0.55,0.64,0.79,1.25,1.65,2.16,3.45,4.8,10.2,21.0] # [microns] (Liske ELT Spectroscopic ETC)
        central_wavelength_photo = central_wavelength_photo_array[mband_index]

        'Sky Extinction & brightness'

        if self.instrument.site == 'Paranal-like':
            k_array = [0.46,0.2,0.11,0.07,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
            msky_array = [21.87,22.43,21.62,20.95,19.71,16.0,14.0,13.0,5.3,1.3,-3.7,-6.5]
        elif self.instrument.site == 'High&Dry':
            k_array = [0.28,0.12,0.07,0.04,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
            msky_array = [21.87,22.43,21.62,20.95,19.71,16.0,14.0,13.0,6.2,2.3,-3.2,-5.8]
        atmospheric_extinction = 10**(-0.4*self.instrument.airmass*k_array[iband_index])

        ' Compute Photometric Zeropoint'
        if self.target.zero == 'Vega':
            Zref = [7.38,7.18,7.44,7.64,7.91,8.51,8.94,9.40,10.32,10.69,11.91,13.17] # (Liske ELT Spectroscopic ETC)
        elif self.target.zero == 'AB':
            Zref = [7.09,7.25,7.44,7.58,7.77,8.14,8.39,8.64,9.04,9.32,10.00,10.57] #(Liske ELT Spectroscopic ETC)
        else:
            print('Did not input valid photometric zero.')
            exit()    
        Z = Zref[mband_index]
        F0 = 10**(-Z)
        Zhost = Zref[mband_host_index]
        F0host = 10**(-Zhost)

        'Photon Energy at central wavelength'
        Egamma = 1.985*10**(-19)/central_wavelength_photo #To verify vs. ETC use photo OTW use instrument
        conversion_factor = self.instrument.efficiency*self.instrument.instrument_band_range\
            *self.instrument.collecting_area/Egamma

        ' Object Flux'
        Fobj = F0*10**(-0.4*self.target.apparent_magnitude)
        Fobj_hat = atmospheric_extinction*Fobj
        
        'Host Flux'
        Fhost = F0host*10**(-0.4*self.target.host_magnitude)
        Fhost_hat = atmospheric_extinction*Fhost
        
        ' Number of photons detected in S/N reference area per exposure'
        Nobj = Fobj_hat*self.instrument.encircled_energy*conversion_factor*self.texp
        
        Nhost = Fhost_hat*self.instrument.encircled_energy*conversion_factor*self.texp
        
        'Sky Flux'
        Fsky_hat = atmospheric_extinction*F0*10**(-0.4*msky_array[iband_index])

        'Number of photons detected in sky background'
        Nsky = Fsky_hat*omega*conversion_factor*self.texp

        'S/N ratio'
        if self.host_star == False:
            starlight_suppression = 0
        else:
            starlight_suppression = self.instrument.starlight_suppression
        self.snr = (self.nexp**0.5)*Nobj/((Nobj+starlight_suppression*Nhost+Nsky+self.instrument.Npix\
                                           *(self.instrument.readout_noise**2)+\
                                               self.instrument.Npix*self.instrument.dark_current*\
                                                   self.texp)**0.5)
        return(self.snr)
    
class Spectroscopic_Instrument:
    ''' Instantiate Instrument class. This class represents telescope instruments
    and can compute S/N and create simulated spectra.
    
    Inputs:
    
        spectral_band: Instrument's band, often just the name as they may be specific 
            and not tied directly to an established photometric band.
        sky_file: Filename containing sky brightness/emission
        transmission_file: Filename containing sky tranmission
        speectral_resolution: Instrument spectral resolution
        efficiency: Instrument Throughput
        Area: Collecting area of telescope
        Nspec: # of spectra over which collected light is disseminated
        encircled_energies: Percentage of PSF w/in S/N reference area
        Rref: Radius of S/N reference area
        dark_current: Dark current in electrons/pixel
        readout_noise: # of electrons'''
    def __init__(self,aperture_diameter,collecting_area,\
                                              sky_file,file_type,name,spectral_resolution,\
                                                  efficiency,Nspec,lambda_ref,dark_current,\
                                                      readout_noise,transmission_file=0,\
                                                          starlight_suppression = 0):
        self.data_type = 'Spectroscopic'
        self.aperture_diameter = aperture_diameter 
        self.collecting_area = collecting_area
        self.sky_file = sky_file
        self.file_type = file_type
        self.spectral_band = name
        self.spectral_resolution = spectral_resolution
        self.efficiency = efficiency 
        self.Nspec = Nspec
        self.lambda_ref = lambda_ref
        self.dark_current = dark_current
        self.readout_noise = readout_noise
        self.transmission_file = transmission_file
        self.starlight_suppression = starlight_suppression 
       
    def initial_calculations(self,target):
        self.instrument_wave,self.instrument_flux = build_band_spectra(self.spectral_band,target,self.lambda_ref)
        if self.spectral_band == 'MODHIS':
            names = ['nm_wave','nm_flux','Wave','flux']
            data = pd.read_csv(self.sky_file,names = names)
            wave = data['Wave']
            flux = data['nm_flux']
            names = ['wave','transmission']
            data1 = pd.read_csv(self.transmission_file,names = names)
            wave_transmission = data1['wave']
            transmission = data1['transmission']
            ''' Using Rref = 2.4*(lambda/D)*206265 & Table 6 ELT Spectral ETC,
                and interpolating
            '''
            self.Rref = 2.4*((self.lambda_ref*1E-6)/self.aperture_diameter)*plate_scale*1000
            rvector = [20.0,21.0,29.0,50.0] #From ELT Spectroscopic ETC Documentation
            evector = [33.9,34.1,35.5,39.4] #From ELT Spectroscopic ETC Documentation
            self.encircled_energy = np.interp(self.Rref,rvector,evector)
            sky_emission = np.interp(self.instrument_wave,wave,flux)
            self.atm_transmission= np.interp(self.instrument_wave,wave_transmission,transmission)
            self.Fsky = sky_emission
        elif self.spectral_band == 'METIS':
            fits_file = fits.open(self.sky_file)
            sky_data = fits_file[1].data
            wave = sky_data['lam']/1000
            transmission_list = sky_data['trans']
            Fsky_list = sky_data['flux']
            self.Rref = 2.4*((self.lambda_ref*1E-6)/self.aperture_diameter)*plate_scale*1000
            rvector = [21.0, 29.0,50.0,63.0] #From ELT Spectroscopic ETC Documentation
            evector = [53.7,62.1,72.0,73.2] #From ELT Spectroscopic ETC Documentation
            self.encircled_energy = np.interp(self.Rref,rvector,evector)
            self.atm_transmission = np.interp(self.instrument_wave,wave,transmission_list)
            self.Fsky = np.interp(self.instrument_wave,wave,Fsky_list)
        elif self.spectral_band == 'GCLEF-Red':
            fits_file = fits.open(self.sky_file)
            sky_data = fits_file[1].data
            wave = sky_data['lam']/1000
            transmission_list = sky_data['trans']
            Fsky_list = sky_data['flux']
            self.Rref = 2.4*((self.lambda_ref*1E-6)/self.aperture_diameter)*plate_scale*1000
            seeing = 350 # [mas] GMT Science Book/Single Target GLAO
            self.encircled_energy = energy_contained(self.Rref, seeing)
            self.atm_transmission = np.interp(self.instrument_wave,wave,transmission_list)
            self.Fsky = np.interp(self.instrument_wave,wave,Fsky_list)
        elif self.spectral_band == 'JWST':
            self.Rref = 2.4*((self.lambda_ref*1E-6)/self.aperture_diameter)*plate_scale*1000
            self.encircled_energy = 100
            self.atm_transmission = 1
            self.Fsky = 0
            
class Photometric_Instrument():
    def __init__(self,aperture_diameter,collecting_area,\
                                            sky_file,file_type,airmass,\
                                                name,instrument_band,\
                                                    central_wavelength_instrument,\
                                                        instrument_band_range,\
                                                            efficiency, site, Npix,\
                                                                encircled_energy,\
                                                                    dark_current,\
                                                                        readout_noise,\
                                                                            pixel_scale,\
                                                                                starlight_suppression = 0):
        self.data_type = 'Photometric'
        self.aperture_diameter = aperture_diameter
        self.collecting_area = collecting_area
        self.sky_file = sky_file
        self.file_type = file_type
        self.airmass = airmass
        self.name = name
        self.instrument_band = instrument_band
        self.central_wavelength_instrument = central_wavelength_instrument
        self.instrument_band_range = instrument_band_range
        self.efficiency = efficiency
        self.site = site
        self.Npix = Npix
        self.encircled_energy = encircled_energy 
        self.dark_current = dark_current
        self.readout_noise = readout_noise
        self.pixel_scale = pixel_scale
        self.starlight_suppression = starlight_suppression 
        
class Photometric_Observation():
    def __init__(self,flux,time_hours,snr):
        self.light_curve = flux
        self.time_hours = time_hours
        self.snr = snr

class NestedSampling():
    def __init__(self,target,time_samples,spectral_observation = 0,photo_observation = 0,\
                 free_contrast = False,free_radius = False,sampling_mode='dynamic',\
                         truths = 0,evolution_spot_number = False,evolution_rotation = False):
        self.target = target
        self.time_samples = time_samples #Should be more than observations for interpolation
        self.free_radius = free_radius
        self.free_contrast = free_contrast
        self.spectral_observation = spectral_observation
        self.photo_observation = photo_observation
        self.sampling_mode = sampling_mode
        self.truths = truths
        self.target.spot_evolution_number = evolution_spot_number
        self.target.spot_evolution_rotation = evolution_rotation
        if evolution_spot_number == False:
            self.evolution_boolean = False
        else:
            self.evolution_boolean = True
    
    def run(self):
        if self.spectral_observation != False :
            if self.photo_observation != False:
                data_mode = 'Spectroscopic&Photometric'
                length_of_observation = max(self.spectral_observation.time_hours[-1],\
                                            self.photo_observation.time_hours[-1])
                n = self.spectral_observation.n
                eps = self.spectral_observation.eps
                spectral_resolution = self.spectral_observation.spectral_resolution
                'Define our 3-D correlated multivariate normal likelihood'
                sigma_spectral = 1/np.mean(self.spectral_observation.snr)
                sigma_photo = 1/(self.photo_observation.snr)
                sig2_spectral = (sigma_spectral**2)*np.ones(len(self.spectral_observation.time_hours))
                sig2_photo = (sigma_photo**2)*np.ones(len(self.photo_observation.time_hours))
                sig2 = np.append(sig2_spectral,sig2_photo)
                C = np.diag(sig2)  
                Cinv = np.linalg.inv(C)  # define the inverse (i.e. the precision matrix)
                lnorm = -0.5 * (np.log(2 * np.pi) * (len(self.spectral_observation.time_hours)+\
                                len(self.photo_observation.time_hours)) +\
                                np.linalg.slogdet(C)[1])  # ln(normalization)
            else:
                data_mode = 'Spectroscopic'
                length_of_observation = self.spectral_observation.time_hours[-1]
                n = self.spectral_observation.n
                eps = self.spectral_observation.eps
                spectral_resolution = self.spectral_observation.spectral_resolution
                'Define our 3-D correlated multivariate normal likelihood'
                sigma_spectral = 1/np.mean(self.spectral_observation.snr)
                C = np.diag((sigma_spectral**2)*np.ones(len(self.spectral_observation.time_hours)))  
                Cinv = np.linalg.inv(C)  # define the inverse (i.e. the precision matrix)
                lnorm = -0.5 * (np.log(2 * np.pi) * len(self.spectral_observation.time_hours) +
                                np.linalg.slogdet(C)[1])  # ln(normalization)
        else:
            if self.photo_observation != False:
                data_mode = 'Photometric'
                length_of_observation = self.photo_observation.time_hours[-1]
                n = 101
                eps = 0.5 #Default
                spectral_resolution = 100E3
                'Define our 3-D correlated multivariate normal likelihood'
                sigma_photo = 1/(self.photo_observation.snr)
                C = np.diag((sigma_photo**2)*np.ones(len(self.photo_observation.time_hours)))  
                Cinv = np.linalg.inv(C)  # define the inverse (i.e. the precision matrix)
                lnorm = -0.5 * (np.log(2 * np.pi) * len(self.photo_observation.time_hours) +
                                np.linalg.slogdet(C)[1])  # ln(normalization)
            else:
                print('You need to input an observeration')
                exit()
        numspot = len(self.truths[0])
        if self.free_contrast == True:
            if self.free_radius == True:
                ndim = numspot*4
            else:
                ndim = numspot*3
        else:
            if self.free_radius == True:
                ndim = numspot*3
            else:
                ndim = numspot*2
        Star2 = self.target
        if self.evolution_boolean == True:
            spot_number = Star2.spot_evolution_number
            rotation = Star2.spot_evolution_rotation
            if self.free_radius == True:
                if self.free_contrast == True:
                    ndim = ndim+2*len(spot_number)
                else:
                    ndim = ndim+len(spot_number)
            else:
                if self.free_contrast == True:
                    ndim = ndim+len(spot_number)
                else:
                    ndim = ndim
        
        def loglike(x):
            """The log-likelihood function."""
            lats = np.zeros(numspot)
            lons = np.zeros(numspot)
            rads = np.zeros(numspot)
            contrasts = np.zeros(numspot)
            if self.evolution_boolean == True:
                radius_evo = np.zeros(numspot)
                contrast_evo = np.zeros(numspot)
            'Build the free parameter array.'
            'For now, this assumes only one evolution per spot.'
            j = 0
            for i in range(numspot):
                lats[i] = x[i+j]
                lons[i] = x[i+j+1]
                if self.free_radius == True:
                    rads[i] = x[i+j+2]
                    if self.evolution_boolean == True:
                        radius_evo[i] = x[i+j+3]
                    if self.free_contrast == True:
                        if self.evolution_boolean == True:
                            contrasts[i] = x[i+j+4]
                            contrast_evo[i] = x[i+j+5]
                            j = j+5
                        else:
                            contrasts[i] = x[i+j+3]
                            j = j+3
                    else:
                        if self.evolution_boolean == True:
                            contrasts[i] = self.truths[3][i]
                            contrast_evo[i] = self.truths[5][i]
                            j = j+3
                        else:
                            contrasts[i] = self.truths[3][i]
                            j = j+2
                else:
                    rads[i] = self.truths[2][i]
                    if self.evolution_boolean == True:
                        radius_evo[i] = self.truths[4][i]
                    if self.free_contrast == True:
                        if self.evolution_boolean == True:
                            contrasts[i] = x[i+j+2]
                            contrast_evo[i] = x[i+j+3]
                            j = j+3
                        else:
                            contrasts[i] = x[i+j+2]
                            j = j+2
                    else:
                        if self.evolution_boolean == True:
                            contrast_evo[i] = self.truths[5][i]
                        contrasts[i] = self.truths[3][i]
                        j = j+1
            'Add Spots'
            Star2.set_spots(lats,lons,rads,contrasts)
            if self.evolution_boolean == True:
                Star2.spot_evolution(spot_number, rotation, radius_evo, contrast_evo)
            if data_mode == 'Spectroscopic':
                spectral_model = Analytical_Model(Star2,length_of_observation,self.time_samples,n,eps,spectral_resolution)
                spectral_model.create_line_profile()
                spectral_model.rotational_plus_instrument_kernel()
                interp_line_profile = np.zeros((n,len(self.spectral_observation.time_hours)))
                for i in range(n):
                    interp_line_profile[i] = np.interp(self.spectral_observation.time_hours,spectral_model.time_hours,spectral_model.line_profile[:,i])
                interp_line_profile = interp_line_profile.T
                baseline = spectral_model.LPrb_plus_instrument
                obs_dev = self.spectral_observation.line_profile-baseline
                model_dev = interp_line_profile-baseline
                diff = np.sum(obs_dev-model_dev,axis = 1)
            elif data_mode == 'Photometric':
                photo_model = Analytical_Model(Star2,length_of_observation,self.time_samples,n,eps,spectral_resolution)
                photo_model.create_lightcurve()
                interp_light_curve = np.interp(self.photo_observation.time_hours,photo_model.time_hours,photo_model.light_curve)
                diff = self.photo_observation.model.light_curve-interp_light_curve
            elif data_mode == 'Spectroscopic&Photometric':
                spectral_model = Analytical_Model(Star2,length_of_observation,self.time_samples,n,eps,spectral_resolution)
                spectral_model.create_line_profile()
                spectral_model.rotational_plus_instrument_kernel()
                interp_line_profile = np.zeros((n,len(self.spectral_observation.time_hours)))
                for i in range(n):
                    interp_line_profile[i] = np.interp(self.spectral_observation.time_hours,spectral_model.time_hours,spectral_model.line_profile[:,i])
                interp_line_profile = interp_line_profile.T
                baseline = spectral_model.LPrb_plus_instrument
                obs_dev = self.spectral_observation.line_profile-baseline
                model_dev = interp_line_profile-baseline
                diff1 = np.sum(obs_dev-model_dev,axis = 1)
                photo_model = Analytical_Model(Star2,length_of_observation,self.time_samples,n,eps,spectral_resolution)
                photo_model.create_lightcurve()
                interp_light_curve = np.interp(self.photo_observation.time_hours,photo_model.time_hours,photo_model.light_curve)
                diff2 = self.photo_observation.model.light_curve-interp_light_curve
                diff = np.append(diff1,diff2)
            lp = -0.5 * np.dot(diff, np.dot(Cinv, diff)) + lnorm
            return lp

        # Define our uniform prior.
        def ptform(u):
            """Transforms samples `u` drawn from the unit cube to samples to those
            from our uniform prior within [-10., 10.) for each variable."""
            x = np.array(u)
            j = 0
            if numspot == 1:
                if self.free_contrast == True:
                    if self.free_radius == True:
                        if self.evolution_boolean == True:
                            x[0] = 90. * (2. * u[0] - 1.) #Latitude
                            x[1] = 180 * (2. * u[1] - 1.) #Longitude
                            x[2] = 30. + 25 * (2. * u[2] - 1.) #Radius
                            x[3] = 30. + 25 * (2. * u[3] - 1.) #Radius Evolved
                            x[4] = 1.0 * (2. * u[4] - 1.) #Contrast
                            x[5] = 1.0 * (2. * u[5] - 1.) #Contrast Evolved
                        else:
                            x[0] = 90. * (2. * u[0] - 1.) #Latitude
                            x[1] = 180 * (2. * u[1] - 1.) #Longitude
                            x[2] = 30. + 25 * (2. * u[2] - 1.) #Radius 
                            x[3] = 1.0 * (2. * u[3] - 1.) #Contrast
                    else:
                        if self.evolution_boolean == True:
                            x[0] = 90. * (2. * u[0] - 1.) #Latitude
                            x[1] = 180 * (2. * u[1] - 1.) #Longitude
                            x[2] = 1.0 * (2. * u[2] - 1.) #Contrast
                            x[3] = 1.0 * (2. * u[3] - 1.) #Contrast Evolved
                        else:
                            x[0] = 90. * (2. * u[0] - 1.) #Latitude
                            x[1] = 180 * (2. * u[1] - 1.) #Longitude
                            x[2] = 1.0 * (2. * u[2] - 1.) #Contrast
                else:
                    if self.free_radius == True:
                        if self.evolution_boolean == True:
                            x[0] = 90. * (2. * u[0] - 1.) #Latitude
                            x[1] = 180 * (2. * u[1] - 1.) #Longitude
                            x[2] = 30. + 25 * (2. * u[2] - 1.) #Radius
                            x[3] = 30. + 25 * (2. * u[3] - 1.) #Radius Evolved
                        else:
                            x[0] = 90. * (2. * u[0] - 1.) #Latitude
                            x[1] = 180 * (2. * u[1] - 1.) #Longitude
                            x[2] = 30. + 25 * (2. * u[2] - 1.) #Radius
                    else:
                        x[0] = 90. * (2. * u[0] - 1.) #Latitude
                        x[1] = 180 * (2. * u[1] - 1.) #Longitude
            elif numspot == 2:
                if self.free_contrast == True:
                    if self.free_radius == True:
                        if self.evolution_boolean == True:
                            x[0] = 90. * (2. * u[0] - 1.) #Latitude
                            x[1] = 180 * (2. * u[1] - 1.) #Longitude
                            x[2] = 30. + 25 * (2. * u[2] - 1.) #Radius
                            x[3] = 30. + 25 * (2. * u[3] - 1.) #Radius
                            x[4] = 0.0 + 1.0 * (2. * u[4] - 1.) #Contrast
                            x[5] = 0.0 + 1.0 * (2. * u[5] - 1.) #Contrast Evolved
                            x[6] = 90 * (2. * u[6] - 1.) #Latitude
                            x[7] = 180 * (2. * u[7] - 1.) #Longitude
                            x[8] = 30. + 25 * (2. * u[8] - 1.) #Radius 
                            x[9] = 30. + 25 * (2. * u[9] - 1.) #Radius Evolved
                            x[10] = 0.0 + 1.0 * (2. * u[10] - 1.) #Contrast
                            x[11] = 0.0 + 1.0 * (2. * u[11] - 1.) #Contrast Evolved
                        else:
                            x[0] = 90. * (2. * u[0] - 1.) #Latitude
                            x[1] = 180 * (2. * u[1] - 1.) #Longitude
                            x[2] = 30. + 25 * (2. * u[2] - 1.) #Radius 
                            x[3] = 0.5 + 0.5 * (2. * u[3] - 1.) #Contrast
                            x[4] = 90 * (2. * u[4] - 1.) #Latitude
                            x[5] = 180 * (2. * u[5] - 1.) #Longitude
                            x[6] = 30. + 25 * (2. * u[6] - 1.) #Radius 
                            x[7] = -0.5 +0.5 * (2. * u[7] - 1.) #Contrast
                    else:
                        if self.evolution_boolean == True:
                            x[0] = 90. * (2. * u[0] - 1.) #Latitude
                            x[1] = 180 * (2. * u[1] - 1.) #Longitude
                            x[2] = 0.5 + 0.5 * (2. * u[2] - 1.) #Contrast
                            x[3] = 0.5 + 0.5 * (2. * u[3] - 1.) #Contrast Evolved
                            x[4] = 90 * (2. * u[4] - 1.) #Latitude
                            x[5] = 180 * (2. * u[5] - 1.) #Longitude
                            x[6] = -0.5 +0.5 * (2. * u[6] - 1.) #Contrast 
                            x[7] = -0.5 +0.5 * (2. * u[7] - 1.) #Contrast Evolved
                        else:
                            x[0] = 90. * (2. * u[0] - 1.) #Latitude
                            x[1] = 180 * (2. * u[1] - 1.) #Longitude
                            x[2] = 0.5 + 0.5 * (2. * u[2] - 1.) #Contrast
                            x[3] = 90 * (2. * u[3] - 1.) #Latitude
                            x[4] = 180 * (2. * u[4] - 1.) #Longitude
                            x[5] = -0.5 +0.5 * (2. * u[5] - 1.) #Contrast 
                else:
                    if self.free_radius == True:
                        if self.evolution_boolean == True:
                            x[0] = 90. * (2. * u[0] - 1.) #Latitude
                            x[1] = 180 * (2. * u[1] - 1.) #Longitude
                            x[2] = 30. + 25 * (2. * u[2] - 1.) #Radius 
                            x[3] = 30. + 25 * (2. * u[3] - 1.) #Radius Evolved
                            x[4] = 90 * (2. * u[4] - 1.) #Latitude
                            x[5] = 180 * (2. * u[5] - 1.) #Longitude
                            x[6] = 30. + 25 * (2. * u[6] - 1.) #Radius 
                            x[7] = 30. + 25 * (2. * u[7] - 1.) #Radius Evolved
                        else:
                            x[0] = 90. * (2. * u[0] - 1.) #Latitude
                            x[1] = 180 * (2. * u[1] - 1.) #Longitude
                            x[2] = 30. + 25 * (2. * u[2] - 1.) #Radius 
                            x[3] = 90 * (2. * u[3] - 1.) #Latitude
                            x[4] = 180 * (2. * u[4] - 1.) #Longitude
                            x[5] = 30. + 25 * (2. * u[5] - 1.) #Radius 
                    else:
                        x[0] = 90. * (2. * u[0] - 1.) #Latitude
                        x[1] = 180 * (2. * u[1] - 1.) #Longitude
                        x[2] = 90 * (2. * u[2] - 1.) #Latitude
                        x[3] = 180 * (2. * u[3] - 1.) #Longitude
            elif numspot == 3:
                if self.free_contrast == True:
                    if self.free_radius == True:
                        if self.evolution_boolean == True:
                            x[0] = 90. * (2. * u[0] - 1.) #Latitude
                            x[1] = 180 * (2. * u[1] - 1.) #Longitude
                            x[2] = 30. + 25 * (2. * u[2] - 1.) #Radius
                            x[3] = 30. + 25 * (2. * u[3] - 1.) #Radius
                            x[4] = 0.0 + 0.5 * (2. * u[4] - 1.) #Contrast
                            x[5] = 0.0 + 0.5 * (2. * u[5] - 1.) #Contrast Evolved
                            x[6] = 90 * (2. * u[6] - 1.) #Latitude
                            x[7] = 180 * (2. * u[7] - 1.) #Longitude
                            x[8] = 30. + 25 * (2. * u[8] - 1.) #Radius 
                            x[9] = 30. + 25 * (2. * u[9] - 1.) #Radius Evolved
                            x[10] = 0.0 + 0.5 * (2. * u[10] - 1.) #Contrast
                            x[11] = 0.0 + 0.5 * (2. * u[11] - 1.) #Contrast Evolved
                            x[12] = 90 * (2. * u[12] - 1.) #Latitude
                            x[13] = 180 * (2. * u[13] - 1.) #Longitude
                            x[14] = 30. + 25 * (2. * u[14] - 1.) #Radius 
                            x[15] = 30. + 25 * (2. * u[15] - 1.) #Radius Evolved
                            x[16] = 0.0 + 0.5 * (2. * u[16] - 1.) #Contrast
                            x[17] = 0.0 + 0.5 * (2. * u[17] - 1.) #Contrast Evolved
                        else:
                            x[0] = 90. * (2. * u[0] - 1.) #Latitude
                            x[1] = 180 * (2. * u[1] - 1.) #Longitude
                            x[2] = 30. + 25 * (2. * u[2] - 1.) #Radius 
                            x[3] = 0.5 + 0.5 * (2. * u[3] - 1.) #Contrast
                            x[4] = 90 * (2. * u[4] - 1.) #Latitude
                            x[5] = 180 * (2. * u[5] - 1.) #Longitude
                            x[6] = 30. + 25 * (2. * u[6] - 1.) #Radius 
                            x[7] = -0.5 +0.5 * (2. * u[7] - 1.) #Contrast
                            x[8] = 90 * (2. * u[8] - 1.) #Latitude
                            x[9] = 180 * (2. * u[9] - 1.) #Longitude
                            x[10] = 30. + 25 * (2. * u[10] - 1.) #Radius 
                            x[11] = 0.0 +0.5 * (2. * u[11] - 1.) #Contrast
                    else:
                        if self.evolution_boolean == True:
                            x[0] = 90. * (2. * u[0] - 1.) #Latitude
                            x[1] = 180 * (2. * u[1] - 1.) #Longitude
                            x[2] = 0.5 + 0.5 * (2. * u[2] - 1.) #Contrast
                            x[3] = 0.5 + 0.5 * (2. * u[3] - 1.) #Contrast Evolved
                            x[4] = 90 * (2. * u[4] - 1.) #Latitude
                            x[5] = 180 * (2. * u[5] - 1.) #Longitude
                            x[6] = -0.5 +0.5 * (2. * u[6] - 1.) #Contrast 
                            x[7] = -0.5 +0.5 * (2. * u[7] - 1.) #Contrast Evolved
                            x[8] = 90 * (2. * u[8] - 1.) #Latitude
                            x[9] = 180 * (2. * u[9] - 1.) #Longitude
                            x[10] = 0.0 +0.5 * (2. * u[10] - 1.) #Contrast 
                            x[11] = 0.0 +0.5 * (2. * u[11] - 1.) #Contrast Evolved
                        else:
                            x[0] = 90. * (2. * u[0] - 1.) #Latitude
                            x[1] = 180 * (2. * u[1] - 1.) #Longitude
                            x[2] = 0.5 + 0.5 * (2. * u[2] - 1.) #Contrast
                            x[3] = 90 * (2. * u[3] - 1.) #Latitude
                            x[4] = 180 * (2. * u[4] - 1.) #Longitude
                            x[5] = -0.5 +0.5 * (2. * u[5] - 1.) #Contrast 
                            x[6] = 90 * (2. * u[6] - 1.) #Latitude
                            x[7] = 180 * (2. * u[7] - 1.) #Longitude
                            x[8] = 0.0 +0.5 * (2. * u[8] - 1.) #Contrast 
                else:
                    if self.free_radius == True:
                        if self.evolution_boolean == True:
                            x[0] = 90. * (2. * u[0] - 1.) #Latitude
                            x[1] = 180 * (2. * u[1] - 1.) #Longitude
                            x[2] = 30. + 25 * (2. * u[2] - 1.) #Radius 
                            x[3] = 30. + 25 * (2. * u[3] - 1.) #Radius Evolved
                            x[4] = 90 * (2. * u[4] - 1.) #Latitude
                            x[5] = 180 * (2. * u[5] - 1.) #Longitude
                            x[6] = 30. + 25 * (2. * u[6] - 1.) #Radius 
                            x[7] = 30. + 25 * (2. * u[7] - 1.) #Radius Evolved
                            x[8] = 90 * (2. * u[8] - 1.) #Latitude
                            x[9] = 180 * (2. * u[9] - 1.) #Longitude
                            x[10] = 30. + 25 * (2. * u[10] - 1.) #Radius 
                            x[11] = 30. + 25 * (2. * u[11] - 1.) #Radius Evolved
                        else:
                            x[0] = 90. * (2. * u[0] - 1.) #Latitude
                            x[1] = 180 * (2. * u[1] - 1.) #Longitude
                            x[2] = 30. + 25 * (2. * u[2] - 1.) #Radius 
                            x[3] = 90 * (2. * u[3] - 1.) #Latitude
                            x[4] = 180 * (2. * u[4] - 1.) #Longitude
                            x[5] = 30. + 25 * (2. * u[5] - 1.) #Radius 
                            x[6] = 90 * (2. * u[6] - 1.) #Latitude
                            x[7] = 180 * (2. * u[7] - 1.) #Longitude
                            x[8] = 30. + 25 * (2. * u[8] - 1.) #Radius 
                    else:
                        x[0] = 90. * (2. * u[0] - 1.) #Latitude
                        x[1] = 180 * (2. * u[1] - 1.) #Longitude
                        x[2] = 90 * (2. * u[2] - 1.) #Latitude
                        x[3] = 180 * (2. * u[3] - 1.) #Longitude 
                        x[4] = 90 * (2. * u[4] - 1.) #Latitude
                        x[5] = 180 * (2. * u[5] - 1.) #Longitude 
            elif numspot == 4:
                if self.free_contrast == True:
                    if self.free_radius == True:
                        if self.evolution_boolean == True:
                            x[0] = 90. * (2. * u[0] - 1.) #Latitude
                            x[1] = 180 * (2. * u[1] - 1.) #Longitude
                            x[2] = 30. + 25 * (2. * u[2] - 1.) #Radius
                            x[3] = 30. + 25 * (2. * u[3] - 1.) #Radius
                            x[4] = 0.0 + 0.5 * (2. * u[4] - 1.) #Contrast
                            x[5] = 0.0 + 0.5 * (2. * u[5] - 1.) #Contrast Evolved
                            x[6] = 90 * (2. * u[6] - 1.) #Latitude
                            x[7] = 180 * (2. * u[7] - 1.) #Longitude
                            x[8] = 30. + 25 * (2. * u[8] - 1.) #Radius 
                            x[9] = 30. + 25 * (2. * u[9] - 1.) #Radius Evolved
                            x[10] = 0.0 + 0.5 * (2. * u[10] - 1.) #Contrast
                            x[11] = 0.0 + 0.5 * (2. * u[11] - 1.) #Contrast Evolved
                            x[12] = 90 * (2. * u[12] - 1.) #Latitude
                            x[13] = 180 * (2. * u[13] - 1.) #Longitude
                            x[14] = 30. + 25 * (2. * u[14] - 1.) #Radius 
                            x[15] = 30. + 25 * (2. * u[15] - 1.) #Radius Evolved
                            x[16] = 0.0 + 0.5 * (2. * u[16] - 1.) #Contrast
                            x[17] = 0.0 + 0.5 * (2. * u[17] - 1.) #Contrast Evolved
                            x[18] = 90 * (2. * u[18] - 1.) #Latitude
                            x[19] = 180 * (2. * u[19] - 1.) #Longitude
                            x[20] = 30. + 25 * (2. * u[20] - 1.) #Radius 
                            x[21] = 30. + 25 * (2. * u[21] - 1.) #Radius Evolved
                            x[22] = 0.0 + 0.5 * (2. * u[22] - 1.) #Contrast
                            x[23] = 0.0 + 0.5 * (2. * u[23] - 1.) #Contrast Evolved
            return x

        if self.sampling_mode == 'static':
            # "Static" nested sampling.
            sampler = dynesty.NestedSampler(loglike, ptform, ndim)
            sampler.run_nested()
            self.results = sampler.results
        elif self.sampling_mode == 'dynamic':
            # "Dynamic" nested sampling.
            dsampler = dynesty.DynamicNestedSampler(loglike, ptform, ndim,sample = 'rwalk')
            dsampler.run_nested()
            self.results = dsampler.results
        else:
            # "Static" nested sampling.
            sampler = dynesty.NestedSampler(loglike, ptform, ndim)
            sampler.run_nested()
            sresults = sampler.results
            
            # "Dynamic" nested sampling.
            dsampler = dynesty.DynamicNestedSampler(loglike, ptform, ndim)
            dsampler.run_nested()
            dresults = dsampler.results
            
            # Combine results from "Static" and "Dynamic" runs.
            self.results = dyfunc.merge_runs([sresults, dresults])


        # Plot a summary of the run.
        rfig, raxes = dyplot.runplot(self.results)

        # Plot traces and 1-D marginalized posteriors.
        tfig, taxes = dyplot.traceplot(self.results)

        # Plot the 2-D marginalized posteriors.
        if numspot == 1:
            if self.free_contrast == True:
                if self.free_radius == True:
                    if self.evolution_boolean == True:
                        labels = ['Lat','Lon', 'Rad','Rad B','Con', 'Con B']
                    else:
                        labels = ['Lat','Lon', 'Rad','Con']
                else:
                    if self.evolution_boolean == True:
                        labels = ['Lat','Lon','Con','Con B']
                    else:
                        labels = ['Lat','Lon','Con']
            else:
                if self.free_radius == True:
                    if self.evolution_boolean == True:
                        labels = ['Lat','Lon', 'Rad','Rad B']
                    else:
                        labels = ['Lat','Lon', 'Rad']
                else:
                    labels = ['Lat','Lon']
        elif numspot == 2:
            if self.free_contrast == True:
                if self.free_radius == True:
                    if self.evolution_boolean == True:
                        labels = ['Lat 1','Lon 1', 'Rad 1','Rad 1B','Con 1','Con 1B','Lat 2','Lon 2', 'Rad 2','Rad 2B','Con 2','Con 2B']
                    else:
                        truths = [self.truths[0][0],self.truths[1][0],self.truths[2][0],self.truths[3][0],\
                                  self.truths[0][1],self.truths[1][1],self.truths[2][1],self.truths[3][1]]
                        labels = ['Lat 1','Lon 1', 'Rad 1','Con 1','Lat 2','Lon 2', 'Rad 2','Con 2']
                else:
                    if self.evolution_boolean == True:
                        labels = ['Lat 1','Lon 1','Con 1','Con 1B','Lat 2','Lon 2','Con 2','Con 2B']
                    else:
                        labels = ['Lat 1','Lon 1','Con 1','Lat 2','Lon 2','Con 2']
            else:
                if self.free_radius == True:
                    if self.evolution_boolean == True:
                        labels = ['Lat 1','Lon 1', 'Rad 1','Rad 1B','Lat 2','Lon 2', 'Rad 2','Rad 2B']
                    else:
                        labels = ['Lat 1','Lon 1', 'Rad 1','Lat 2','Lon 2', 'Rad 2']
                else:
                    labels = ['Lat 1','Lon 1','Lat 2','Lon 2']
        elif numspot == 3:
            if self.free_contrast == True:
                if self.free_radius == True:
                    if self.evolution_boolean == True:
                        labels = ['Lat 1','Lon 1', 'Rad 1','Rad 1B','Con 1','Con 1B','Lat 2','Lon 2', 'Rad 2','Rad 2B','Con 2','Con 2B','Lat 3','Lon 3', 'Rad 3','Rad 3B','Con 3','Con 3B']
                    else:
                        labels = ['Lat 1','Lon 1', 'Rad 1','Con 1','Lat 2','Lon 2', 'Rad 2','Con 2','Lat 3','Lon 3', 'Rad 3','Con 3']
                else:
                    if self.evolution_boolean == True:
                        labels = ['Lat 1','Lon 1','Con 1','Con 1B','Lat 2','Lon 2','Con 2','Con 2B','Lat 3','Lon 3','Con 3','Con 3B']
                    else:
                        labels = ['Lat 1','Lon 1','Con 1','Lat 2','Lon 2','Con 2','Lat 3','Lon 3','Con 3']
            else:
                if self.free_radius == True:
                    if self.evolution_boolean == True:
                        labels = ['Lat 1','Lon 1', 'Rad 1','Rad 1B','Lat 2','Lon 2', 'Rad 2','Rad 2B','Lat 3','Lon 3', 'Rad 3','Rad 3B']
                    else:
                        labels = ['Lat 1','Lon 1', 'Rad 1','Lat 2','Lon 2', 'Rad 2','Lat 3','Lon 3', 'Rad 3']
                else:
                    labels = ['Lat 1','Lon 1','Lat 2','Lon 2','Lat 1','Lon 1']
        elif numspot == 4:
            if self.free_contrast == True:
                if self.free_radius == True:
                    if self.evolution_boolean == True:
                        labels = ['Lat 1','Lon 1', 'Rad 1','Rad 1B','Con 1','Con 1B','Lat 2','Lon 2', 'Rad 2','Rad 2B','Con 2','Con 2B','Lat 3','Lon 3', 'Rad 3','Rad 3B','Con 3','Con 3B','Lat 4','Lon 4', 'Rad 4','Rad 4B','Con 4','Con 4B']
                    else:
                        labels = ['Lat 1','Lon 1', 'Rad 1','Con 1','Lat 2','Lon 2', 'Rad 2','Con 2','Lat 3','Lon 3', 'Rad 3','Con 3','Lat 4','Lon 4', 'Rad 4','Con 4']
        quantiles = [0.16,0.50,0.84]
        # cfig, caxes = dyplot.cornerplot(self.results,labels = labels,show_titles=True,quantiles = quantiles,title_quantiles = (0.16,0.5,0.84),truths = self.truths)
        cfig, caxes = dyplot.cornerplot(self.results,labels = labels,show_titles=True,quantiles = quantiles,title_quantiles = (0.16,0.5,0.84))
        # else:
        #     cfig, caxes = dyplot.cornerplot(self.results,labels = labels,truths = self.truths,show_titles=True,quantiles = quantiles,title_quantiles = (0.16,0.5,0.84))

class Analytical_Model:
    '''
    Object class used to generate analytical line profiles and light curves.
    
        Inputs:
            length_of_observation: Total time for observations [hrs]
            time_steps: # of times steps for 1 revolution of target
            n: # of pixels in line profile
            eps: Linear limb darkening coefficient
            target: Target-class object
    '''
    
    def __init__(self,target,length_of_observation,time_steps,n,eps,spectral_resolution,time_stamps = 0):
        self.length_of_observation = length_of_observation
        self.n = n
        self.eps = eps
        self.spectral_resolution = spectral_resolution
        self.target = target
        self.rv_array = np.linspace(-2*self.target.vsini,2*self.target.vsini,n)
        'Assign Target Data to Model'
        " Zrb is the rotational broadening kernel used for analtical model."
        Zrb = rotationalbroadening(self.rv_array, self.target.vsini, self.target.vstar, self.eps)
        self.Zrb = Zrb/sum(Zrb) #Normalize the rotational broadening kernel
        self.time_steps = time_steps
        self.time_hours = np.linspace(0,length_of_observation,time_steps)
        
    def rotational_broadening_kernel(self):
        ''' The rotational broadening kernel has either bright or dark Gaussian spots
        either added or subtracted (respectively).'''
        LPrb = np.copy(self.Zrb)
        for i in range(len(self.target.lat)):
            LPrb = LPrb + self.spots_generator(self.target.lat[i],self.target.lon[i],self.target.radius[i],self.target.contrast[i],i)
        self.LPrb = LPrb
        
    def instrument_broadening_kernel(self):
        " This function creates an instrument profile alone."
        c = 2.9979E5 #[km/s]
        sig = c/self.spectral_resolution/2
        " The instrument profile is a Gaussian distribution with std dev = c/R."
        LPinstrument = np.exp(-0.5 * np.power(self.rv_array/sig, 2.)) / (np.sqrt(2*np.pi)*sig)
        " Normalize the instrument profile."
        LPinstrument = LPinstrument/sum(LPinstrument)
        self.LPinstrument = LPinstrument
    
    def rotational_plus_instrument_kernel(self):
        ''' Here we convolve an instrument profile with the rotational broadening
        kernel to create our analytical line profile which can then be modified
        by either bright (additive) or dark (subtractive) Gaussian spots.'''
        LPrb = np.copy(self.Zrb)
        " The std dev is c/R"
        sig = c/self.spectral_resolution/2
        " Here we can create an instrument broadening kernel using a Gaussian distribution."
        LPinstrument = np.exp(-0.5 * np.power(self.rv_array/sig, 2.)) / (np.sqrt(2*np.pi)*sig)
        LPinstrument = LPinstrument/sum(LPinstrument) 
        ''' The instrument and rotational broadening kernels are convolved. This new line
        profile is used for generating the line profiles with Gaussian spots.'''
        Za = np.convolve(LPrb,LPinstrument,'same')
        self.LPrb_plus_instrument = Za/sum(Za) # Analytical line profile is normalized 
    
        
    def create_line_profile(self):
        ''' This function creates a convolved instrument & rotationally broadened
        line profile and then adds Gaussian spots via spots_generator.
        
        The instrument input here needs to be an Instrument class.'''
        
        c = 2.9979E5 #[km/s]
        " Import rotationally broadened kernel."
        LPrb = np.copy(self.Zrb)
        " Create Gaussian instrument line profile with std dev = c/R"
        sig = c/self.spectral_resolution/2
        LPinstrument = np.exp(-0.5 * np.power(self.rv_array/sig, 2.)) / (np.sqrt(2*np.pi)*sig)
        LPinstrument = LPinstrument/sum(LPinstrument)
        self.LPinstrument = LPinstrument
        " Convolve rotationally broadened & instrument kernel and normalize."
        Za = np.convolve(LPrb,LPinstrument,'same')
        Z = Za/sum(Za)
        " Add Gaussian Spots"
        for i in range(len(self.target.lat)):
            Z = Z + self.spots_generator(self.target.lat[i],self.target.lon[i],self.target.radius[i],self.target.contrast[i],i)
        self.line_profile = Z
        
    def create_lightcurve(self):
        self.create_line_profile()
        self.light_curve = np.sum(self.line_profile,axis = 1)
    
    def spots_generator(self,lat,lon,radius,contrast,spot_number):
        ''' This function accepts a single spot (with lat,lon,radius,contrast) and
        creates a line profile (in RV space) which can be added (bright spot)
        or subtracted (dark spot) to create the full analytical model.'''
        
        degrees_of_observation = (self.length_of_observation/self.target.period)*360
        if degrees_of_observation <= 360:
            lon_array = np.linspace(lon,lon+degrees_of_observation,self.time_steps)
            ''' rotation_inclination uses the Euler-Rodrigues formula to perform a coordinate
            transformation which takes into account a target's inclination vs. the observer.'''
            latspot,lonspot = rotation_inclination(lat,lon_array,self.target.inclination)
            " a & b are the longitudinal and latitudinal (respectively) spot radii corrected"
            " for the Lambertian cosine law (the grow smaller as they approach the target's"
            " limbs.)"
            a = radius*np.ones(self.time_steps)*np.abs(np.cos(np.radians(lonspot)))
            b = radius*np.ones(self.time_steps)*np.abs(np.cos(np.radians(latspot)))
            " Convert from degree to RV space."
            va,vb = a*self.target.vsini/90,b*self.target.vsini/90
            # va = self.vsini*np.sin(np.radians(radius))*np.cos(np.radians(lonspot))
            " Spot area as an ellipse on the stellar disk"
            spot_area = (a*b)/(90**2)
            ''' Intensity computes the spot's magnitude based on area, contrast, & a factor
            to account for the line profile being a 1D represenation of a 2D phenomenon.'''
            intensity = 2.2807*spot_area*contrast #Value may be np.sqrt(5*np.pi/3) instead of 2.2807
            ''' Viewing accounts for whether or not a spot is visible at each time_stepsstep.
            In many instances (& inclinations), the spot will be on the non-viewable
            hemisphere.'''
            view = viewing(lat,latspot,lonspot,self.time_steps,self.target.inclination)
            ''' We use matrix methods to create the Gaussian distribution for the spots
            for each timestep in RV space.'''
            rv_temp = np.tile(lonspot*self.target.vsini/90,self.n).reshape(self.n,self.time_steps).T
            # rv_temp = np.tile(self.vsini*np.sin(np.radians(lonspot)),self.n).reshape(self.n,self.time_steps).T
            rv_halfwidth = np.tile(va,self.n).reshape(self.n,self.time_steps).T
            rvsub = (self.rv_array-rv_temp)/rv_halfwidth
            Z = np.exp(-0.5*rvsub*rvsub).T#/(2*np.pi*va)#/np.sqrt(np.pi)
            ''' We ensure the star spot has been normalized prior to being multipled
            by its intensity.'''
            sumZ = np.sum(Z,axis=0)
            sumZ = np.where(sumZ<1,1,sumZ)
            Z = Z/sumZ
            " The final spot line profile is computed."
            Zspot = -(Z*intensity*view).T
            return(Zspot) 
        else:
            time_samples = int(self.time_steps*360/degrees_of_observation)
            if self.target.prior_rotation == True:
                Zspot = np.zeros((self.time_steps+time_samples,self.n))
                ' 1 additional rotation to round up and another to account for prior rotation'
                number_of_rotations = int(degrees_of_observation/360)+2
            else:
                Zspot = np.zeros((self.time_steps,self.n))
                if degrees_of_observation%360 == 0:
                    number_of_rotations = int(degrees_of_observation/360)
                else:
                    number_of_rotations = int(degrees_of_observation/360)+1
            current_rotation = 0
            sample_start = 0
            longitude = lon
            for i in range(number_of_rotations):
                if current_rotation+1 < number_of_rotations:
                    observational_phase = 360
                else:
                    observational_phase = np.mod(degrees_of_observation,360*(number_of_rotations-1))
                    if observational_phase != 360.0:
                        if observational_phase != 0.0:
                            time_samples = np.mod(self.time_steps,time_samples)
                    if observational_phase == 0.0:
                        observational_phase = 360.0
                        
                if isinstance(longitude,np.ndarray) == True:
                    dlongitude = longitude[-1]-longitude[-2]
                    longitude = longitude[-1]+dlongitude
                if isinstance(radius,np.ndarray) == True:
                    radius = radius[-1]
                if isinstance(contrast,np.ndarray) == True:
                    contrast = contrast[-1]

                if self.target.spot_evolution_boolean == True:
                    index = np.where((np.array(self.target.spot_evolution_rotation)==current_rotation+1) & \
                                     (np.array(self.target.spot_evolution_number) == spot_number))[0]
                    for i in range(len(index)):
                        radius = np.linspace(radius,self.target.spot_evolution_radius[index[i]],time_samples)
                        contrast = np.linspace(contrast,self.target.spot_evolution_contrast[index[i]],time_samples)

                longitude = np.linspace(longitude,longitude+observational_phase,time_samples)
                ''' rotation_inclination uses the Euler-Rodrigues formula to perform a coordinate
                transformation which takes into account a target's inclination vs. the observer.'''
                latspot,lonspot = rotation_inclination(lat,longitude,self.target.inclination)
                " a & b are the longitudinal and latitudinal (respectively) spot radii corrected"
                " for the Lambertian cosine law (the grow smaller as they approach the target's"
                " limbs.)"
                a = radius*np.abs(np.cos(np.radians(lonspot)))
                b = radius*np.abs(np.cos(np.radians(latspot)))
                " Convert from degree to RV space."
                va,vb = a*self.target.vsini/90,b*self.target.vsini/90
                " Spot area as an ellipse on the stellar disk"
                spot_area = (a*b)/(90**2)
                ''' Intensity computes the spot's magnitude based on area, contrast, & a factor
                to account for the line profile being a 1D represenation of a 2D phenomenon.'''
                intensity = 2.2807*spot_area*contrast #Value may be np.sqrt(5*np.pi/3) instead of 2.2807
                ''' Viewing accounts for whether or not a spot is visible at each time_stepsstep.
                In many instances (& inclinations), the spot will be on the non-viewable
                hemisphere.'''
                view = viewing(lat,latspot,lonspot,time_samples,self.target.inclination)
                ''' We use matrix methods to create the Gaussian distribution for the spots
                for each timestep in RV space.'''
                rv_temp = np.tile(lonspot*self.target.vsini/90,self.n).reshape(self.n,time_samples).T
                rv_halfwidth = np.tile(va,self.n).reshape(self.n,len(va)).T
                rvsub = (self.rv_array-rv_temp)/rv_halfwidth
                Z = np.exp(-0.5*rvsub*rvsub).T#/(2*np.pi*va)#/np.sqrt(np.pi)
                ''' We ensure the star spot has been normalized prior to being multipled
                # by its intensity.'''
                sumZ = np.sum(Z,axis=0)
                sumZ = np.where(sumZ<1,1,sumZ)
                Z = Z/sumZ
                " The final spot line profile is computed."
                Zspot[sample_start:sample_start+time_samples,:] = -(Z*intensity*view).T
                current_rotation += 1
                sample_start = sample_start+time_samples
            return(Zspot) 

class Numerical_Model:
    '''
    Object class used to generate numerical line profiles, light curves, and star maps.
    
        Inputs:
            length_of_observation: Total time for observations [hrs]
            time_steps: # of times steps for 1 revolution of target
            n: # of pixels in line profile
            eps: Linear limb darkening coefficient
            target: Target-class object
        '''
        
    def __init__(self,target,length_of_observation,time_steps,n,eps,spectral_resolution):
        self.length_of_observation = np.copy(length_of_observation)
        self.n = n
        self.eps = eps
        self.spectral_resolution = spectral_resolution
        self.target = target
        self.time_steps = time_steps
        self.rv_array = np.linspace(-2*self.target.vsini,2*self.target.vsini,n)
        self.time_hours = np.linspace(0,length_of_observation,time_steps)
    
    def run(self,noiseon = False, sigma = 0):
        ''' Runs numerical code based on input instrument. Instrument is used so that
        instrument broadening can be taken into account.'''
        time_steps = self.time_steps
        ''' L and l are the # of longitudinal and latitudinal cells'''
        L = 500
        l = int(L/2)
        ''' rv_num provides an RV mapping based on target longitude.'''
        rv_num = np.linspace(-2*self.target.vsini,2*self.target.vsini,L)
        ''' We create a grid map for the entire surface map.'''
        latcell = np.linspace(90,-90,l)*np.ones((L,1))
        latcell = latcell.T
        loncell = np.linspace(-180,180,L)*np.ones((l,1))
        ''' mu computes the distance of each cell from the sub-observer point.'''
        # mu2 = 1-(latcell**2+loncell**2)/90**2
        # mu2_bad_index = np.where(mu2<0)
        # mu2[mu2_bad_index] = 0
        mu = np.sqrt(1-(latcell**2+loncell**2)/90**2)
        ''' starflux defines the area viewable by the oberverer.'''
        starflux = 1-self.eps*(1-mu)
        notnumbers = np.isnan(mu)
        ''' Non-viewable regions of the target are set to zero.'''
        starflux[notnumbers] = 0
        ''' starflux is summed to create a reference for normalization'''
        starfluxsum = sum(sum(starflux))
        ''' Flux is summed along the longitudinal axis to create an initial line profile'''
        starLP = np.sum(starflux,axis=0)/starfluxsum
        ''' The spatial line profile is interpolated onto the line profile desired 
        (based on value of n)'''
        starLP_interp_temp = np.interp(self.rv_array,rv_num,starLP)
        ''' The line profile is then convolved with an instrument profile.'''
        starLPi = add_instrument_broadening(starLP_interp_temp,self.spectral_resolution,self.rv_array,self.time_steps,self.n)       
        self.line_profile_unspotted = starLPi
        ''' The ratio between the maximum of these two lines profiles is the key to properly
        adjusting the magnitude of the Gaussian spots.'''
        self.ratio = max(starLPi)/max(starLP)
        degrees_of_observation = (self.length_of_observation/self.target.period)*360
        if degrees_of_observation <= 360:
            '''' Properly formats period and time'''
            if isinstance(time_steps,list):
                period = time_steps[-1]
                time_array = time_steps
                time_steps = len(time_steps)
            else:
                time_array = 0
            '''' Phase array accounts for the phases being considered through 1 rotation'''
            phase_array = np.zeros((len(self.target.lat),time_steps))
            ''' latspot and lonspot track the lat/lon for each spot after Euler-Rodrigues
            transformation.'''
            latspot = np.zeros((len(self.target.lat),time_steps))
            lonspot = np.zeros((len(self.target.lat),time_steps))
            ''' frontside tracks whether or not a spot is viewable at each epoch'''
            frontside = np.zeros((len(self.target.lat),time_steps))
            ''''In the loop below, the transformed lat/lon are computed using
            the Euler-Rodrigues formula, whether or not the spot is in the viewable
            hemisphere is also tracked with frontside() function
            '''
            for i in range(len(self.target.lat)):
                if isinstance(time_array,list):
                    phase_array[i] = self.target.lon[i]+np.multiply(time_array,1/period)*360 #Period of 5.05 hrs (Karalidi et al., 2016) 
                else:
                    phase_array[i] = np.linspace(self.target.lon[i],self.target.lon[i]+360,time_steps)
                latspot[i],lonspot[i] = rotation_inclination(self.target.lat[i],phase_array[i],self.target.inclination)
                frontside[i] = viewing(self.target.lat[i],latspot[i],lonspot[i],time_steps,self.target.inclination)
            ''' LPtime_steps is the array storing the numerical line profile for each epoch.'''
            LPtime = np.zeros((time_steps,self.n))
            'Numerical flux stores photometric flux for light curves'
            numerical_flux = np.zeros(time_steps)
            'Save Flux Map for Each Time Step'
            FluxSave = np.zeros((l,L),np.float_)
            for i in range(time_steps):
                spot = np.zeros((l,L))
                for k in range(len(self.target.lat)):
                    ''' Compute 2D Spot Gaussian for each cell'''
                    sigx = self.target.radius[k]*np.abs(np.cos(np.radians(lonspot[k,i])))
                    sigy = self.target.radius[k]*np.abs(np.cos(np.radians(latspot[k,i])))
                    subx2 = ((loncell-lonspot[k,i])/sigx)**2
                    suby2 = ((latcell-latspot[k,i])/sigy)**2
                    intensity = np.exp(-(subx2+suby2)/2)
                    spot = spot-intensity*self.target.contrast[k]*frontside[k,i]
                ''' Normalize the spotmap and sum to create inital line profile (lon space)'''
                spotmap = spot/starfluxsum
                spotLP = np.sum(spotmap,axis=0)
                ''' Interpolate line profile into RV space & account for flux ratio'''
                spotLP_interp = self.ratio*np.interp(self.rv_array,rv_num,spotLP)
                ''' Add spot line profile to stellar line profile and record'''
                tempLP = starLPi+spotLP_interp
                'Save Star Map, Line Profile, and Photometric Flux for each time step'
                spot[notnumbers] = 0
                starmap = (starflux+spot)/starfluxsum
                FluxSave = np.dstack((FluxSave,starmap))
                LPtime[i] = tempLP
                numerical_flux[i] = sum(tempLP)
            if noiseon == True:
                numerical_flux = numerical_flux+max(numerical_flux)*np.random.normal(0,sigma,time_steps)
            self.StarMap = FluxSave[:,125:375,1:]
            self.line_profile = LPtime
            self.light_curve = numerical_flux
        else:
            if degrees_of_observation%360 == 0:
                number_of_rotations = int(degrees_of_observation/360)
            else:
                number_of_rotations = int(degrees_of_observation/360)+1
            current_rotation = 0
            sample_start = 0
            FluxSave = np.zeros((l,L),np.float_)
            LPtime = np.zeros((self.time_steps,self.n))
            numerical_flux = np.zeros(self.time_steps)
            radius = list(np.copy(self.target.radius))
            contrast = list(np.copy(self.target.contrast))
            longitude = list(np.copy(self.target.lon))
            for m in range(number_of_rotations):
                if current_rotation+1 < number_of_rotations:
                    observational_phase = 360
                    time_samples = int(self.time_steps*360/degrees_of_observation)
                else:
                    observational_phase = np.mod(degrees_of_observation,360*(number_of_rotations-1))
                    check_if_zero = np.mod(self.time_steps,time_samples)
                    if check_if_zero ==0:
                        time_samples = time_samples
                    else:
                        time_samples = check_if_zero
                    # time_samples = np.mod(self.time_steps,(int(self.time_steps*360/degrees_of_observation)+1)*(number_of_rotations-1))
                # if isinstance(self.time_steps,list):
                #     period = self.time_steps[-1]
                #     time_array = self.time_steps
                #     time_steps = len(self.time_steps)
                # else:
                #     time_steps = self.time_steps
                #     time_array = 0
                # phase_array = np.zeros((len(self.target.lat),time_samples))
                latspot = np.zeros((len(self.target.lat),time_samples))
                lonspot = np.zeros((len(self.target.lat),time_samples))
                frontside = np.zeros((len(self.target.lat),time_samples))
                for i in range(len(self.target.lat)):
                    # if isinstance(time_array,list):
                    #     phase_array[i] = self.lon[i]+np.multiply(time_array,1/period)*360 #Period of 5.05 hrs (Karalidi et al., 2016) 
                    # else:
                    if isinstance(longitude[i],np.ndarray) == True:
                        dlongitude = longitude[i][-1]-longitude[i][-2]
                        longitude[i] = longitude[i][-1]+dlongitude
                    if isinstance(radius[i],np.ndarray) == True:
                        dradius = 0 # radius[i][-1]-radius[i][-2]
                        radius[i] = radius[i][-1]+dradius
                    if isinstance(contrast[i],np.ndarray) == True:
                        dcontrast = 0 #contrast[i][-1]-contrast[i][-2]
                        contrast[i] = contrast[i][-1]+dcontrast
                    longitude[i] = np.linspace(longitude[i],longitude[i]+observational_phase,time_samples)
                    latspot[i],lonspot[i] = rotation_inclination(self.target.lat[i],longitude[i],self.target.inclination)
                    frontside[i] = viewing(self.target.lat[i],latspot[i],lonspot[i],time_samples,self.target.inclination)

                if self.target.spot_evolution_boolean == True:
                    index = np.where(np.array(self.target.spot_evolution_rotation)==current_rotation+1)[0]
                    for i in range(len(index)):
                        radius[i] = np.linspace(radius[i],self.target.spot_evolution_radius[index[i]],time_samples)
                        contrast[i] = np.linspace(contrast[i],self.target.spot_evolution_contrast[index[i]],time_samples)

                for i in range(time_samples):
                    spot = np.zeros((l,L))
                    for k in range(len(self.target.lat)):
                        # else:
                        #     radius = self.target.radius[k]
                        #     contrast = self.target.contrast[k]
                        if isinstance(radius[k],np.ndarray) == True:
                            sigx = radius[k][i]*(np.abs(np.cos(np.radians(lonspot[k,i]))))
                            sigy = radius[k][i]*np.abs(np.cos(np.radians(latspot[k,i])))
                            subx2 = ((loncell-lonspot[k,i])/sigx)**2
                            suby2 = ((latcell-latspot[k,i])/sigy)**2
                            intensity = np.exp(-(subx2+suby2)/2)
                            spot = spot-intensity*contrast[k][i]*frontside[k,i]
                        else:
                            sigx = radius[k]*(np.abs(np.cos(np.radians(lonspot[k,i]))))
                            sigy = radius[k]*np.abs(np.cos(np.radians(latspot[k,i])))
                            subx2 = ((loncell-lonspot[k,i])/sigx)**2
                            suby2 = ((latcell-latspot[k,i])/sigy)**2
                            intensity = np.exp(-(subx2+suby2)/2)
                            spot = spot-intensity*contrast[k]*frontside[k,i]
                    ''' Normalize the spotmap and sum to create inital line profile (lon space)'''
                    spotmap = spot/starfluxsum
                    spotLP = np.sum(spotmap,axis=0)
                    ''' Interpolate line profile into RV space & account for flux ratio'''
                    spotLP_interp = self.ratio*np.interp(self.rv_array,rv_num,spotLP)
                    ''' Add spot line profile to stellar line profile and record'''
                    tempLP = starLPi+spotLP_interp
                    'Save Star Map, Line Profile, and Photometric Flux for each time step'
                    spot[notnumbers] = 0
                    starmap = (starflux+spot)/starfluxsum
                    FluxSave = np.dstack((FluxSave,starmap))
                    numerical_flux[i+sample_start] = sum(tempLP)
                    LPtime[i+sample_start] = tempLP
                current_rotation += 1
                sample_start = sample_start+time_samples
            if noiseon == True:
                numerical_flux = numerical_flux+max(numerical_flux)*np.random.normal(0,sigma,time_steps)
            self.StarMap = FluxSave[:,125:375,1:]
            self.line_profile = LPtime
            self.light_curve = numerical_flux
    
    
    def numerical_broadening_kernel(self): # Generate Gray Numerical Line Profiles/Light Curves/Maps
        ''' Function creates a line profile wihout noise and provides the numerical
        computation to also create light curves and surface maps.
        
        Check to see if spots have been defined'''
        if self.target.lat: 
            spots = self.target.lat,self.target.lon,self.target.radius,self.target.contrast
        else:
            spots = [],[],[],[]
        # inputs = self.vsini,self.inclination,self.time_steps,self.eps,self.n
        " Run numerical code."
        self.run()
        # self.line_profile,self.FluxNumerical,self.FluxMap = NumericalCore(inputs,spots,noiseon=False,sigma=0)
    
    def create_map(self,time_stamp):
        "Runs numerical code and generates surface map at input time interval."
        if self.target.lat: #Check to see if spots have been defined
            spots = self.target.lat,self.target.lon,self.target.radius,self.target.contrast
        else:
            spots = [],[],[],[]
        # inputs = self.vsini,self.inclination,self.time_steps,self.eps,self.n
        self.run()
        # line_profile,light_curve,FluxMap = NumericalCore(inputs,spots,noiseon=False,sigma=0)
        scheme = "gist_heat" 
        fig, ax = plt.subplots()
        extent = [-90,90,-90,90]
        plt.imshow(self.StarMap[:,:,time_stamp],cmap=scheme,extent = extent)
        
    def create_video(self):
        "Runs numerical code and generates video of 1 full rotation."
        self.run()
        # line_profile,light_curve,FluxMap = NumericalCore(inputs,spots,noiseon=False,sigma=0)
        scheme = "gist_heat" 
        fig, ax = plt.subplots()
        extent = [-90,90,-90,90]
        for i in range(self.time_steps):
            plt.imshow(self.StarMap[:,:,i],cmap=scheme,extent = extent)
            plt.xlabel('Longitude (deg)')
            plt.ylabel('Latitude (deg)')
            plt.pause(0.005)
            plt.clf()
        plt.imshow(self.StarMap[:,:,-1],cmap=scheme,extent = extent)




def add_instrument_broadening(line_profile,spectral_resolution,rv_array,time_steps,n,multiple=False):
    ''' Non-Class Associated Instrument Broadening Function'''
    sig = c/spectral_resolution/2
    LPinstrument = np.exp(-0.5 * np.power(rv_array/sig, 2.)) / (np.sqrt(2*np.pi)*sig)
    LPinstrument = LPinstrument/sum(LPinstrument)
    if multiple == True:
        Z = np.zeros((time_steps,n))
        for k in range(time_steps):
           Za = np.convolve(line_profile[k,:],LPinstrument,'same')
           Z[k,:] = Za/sum(Za)
    else:
        Za = np.convolve(line_profile,LPinstrument,'same')
        Z = Za/sum(Za) 
    return(Z)


def spectral_band_ranges(spectral_band,lambda_ref):
    ''' Set wavelength range based on desired band.'''
    if spectral_band == 'U':
        wav_range = [lambda_ref-0.066,lambda_ref+0.066]
    elif spectral_band == 'B': # L band
        wav_range = [lambda_ref-0.094,lambda_ref+0.094]
    elif spectral_band == 'V': # L band
        wav_range = [lambda_ref-0.088,lambda_ref+0.088]
    elif spectral_band == 'R': # L band
        wav_range = [lambda_ref-0.138,lambda_ref+0.138]
    elif spectral_band == 'I': # L band
        wav_range = [lambda_ref-0.149,lambda_ref+0.149]
    elif spectral_band == 'J': # L band
        wav_range = [lambda_ref-0.213,lambda_ref+0.213]
    elif spectral_band == 'H': # L band
        wav_range = [lambda_ref-0.307,lambda_ref+0.307]
    elif spectral_band == 'K': # L band
        wav_range = [lambda_ref-0.390,lambda_ref+0.390]
    elif spectral_band == 'L': # L band
        wav_range = [lambda_ref-0.15,lambda_ref+0.15] #Values from METIS webpage
    elif spectral_band == 'M': # M band
        wav_range = [lambda_ref-0.15,lambda_ref+0.15]
    elif spectral_band == 'Lsnr_test':
        wav_range = [3.430,3.470]
    elif spectral_band == 'JWST': # L band
        wav_range = [lambda_ref-2,lambda_ref+2]
    elif spectral_band == 'METIS':
        wav_range = [3.65,3.95]
    elif spectral_band == 'MODHIS':
        wav_range = [0.95,2.4] #Entire bandwidth
    elif spectral_band == 'GCLEF-Blue':
        wav_range = [0.35,0.54]
    elif spectral_band == 'GCLEF-Red':
        wav_range = [0.54,0.95]
    else:
        print('Did not input valid spectral band.')
        exit()
    return(wav_range)


def build_band_spectra(spectral_band,target,lambda_ref,normalize=False):
    ''' Takes target spectra, selects desired band, and normalizes
        spectra w.r.t the contiuum
    '''
    
    file = target.spectra_file
    # file_type = target.file_type
    ''' Select Wavelength Range Based on Spectral Band '''
    wav_range = spectral_band_ranges(spectral_band,lambda_ref)
    
    ''' Upload Spectrum Data'''
    # if file_type == 'FITS':
    File = fits.open(file)
    data = File[1].data
    template_wavelength = data['WAVELENGTH']
    template_flux = data['FLUX']
    band_wav_indices = np.where((template_wavelength > wav_range[0]) & (template_wavelength < wav_range[1]))
    # elif file_type == 'CSV':
    #     names = ['Wave','flux']
    #     data = pd.read_csv(file,names = names)
    #     np_data = pd.DataFrame(data).to_numpy()
    #     template_wavelength = np_data[:,0]/1E4
    #     template_flux = np_data[:,1]
    # else:
    #     print('Invalid Spectra File Type')
    #     exit()
    
    # fig,ax = plt.subplots()
    # plt.plot(template_wavelength,template_flux)
    
    ''' Select wavelength range'''
    band_wav_indices = np.where((template_wavelength > wav_range[0]) & (template_wavelength < wav_range[1]))
    x = band_wavelength = template_wavelength[band_wav_indices]
    y = band_flux = template_flux[band_wav_indices]

    if normalize == True:
        ''' Normalize Spectra '''
        spectrum = Spectrum1D(flux=y*u.Jy, spectral_axis=x*u.um)
        g1_fit = fit_generic_continuum(spectrum)
        y_continuum_fitted = g1_fit(x*u.um)
        y_fit = y_continuum_fitted.value
        band_flux = band_flux/y_fit
    return(band_wavelength,band_flux)


def blackbody(spectral_band,temperature,dlambda,lambda_ref):
    ''' Creates blackbody curve based on spectral_band and temperature.
    Returns wavelength in microns.
    '''
    c = 2.9979E8 #[m/s]
    h = 6.62607015E-34 # [J*s]
    kB = 1.380649E-23 #[J/K]
    ''' Select Wavelength Range Based on Spectral Band '''
    wav_range = spectral_band_ranges(spectral_band,lambda_ref)
    band_wavelength = np.arange(wav_range[0],wav_range[-1],dlambda)/(1E6) # [meters]
    '''Planck's Blackbody Equation'''
    band_flux= ((2*np.pi*h*c**2)/(band_wavelength**5))*(np.exp(h*c/(band_wavelength*kB*temperature))-1)**(-1)
    return(band_wavelength*1E6,band_flux)


def rotationalbroadening(rv_array,vsini,vstar,eps):
    ''' Creates RB-line profile across RV span of rv_array.
    Inputs:
        rv_array: radial velocity array
        vsini: rotational velocity v_r*sin(inclination)
        vstar: RV of star system w.r.t solar system
        eps: Linear Limb Darkening Coefficient
    '''
    c1 = 2*(1-eps)/(np.pi*vsini*(1-(eps/3)))
    c2 = eps/(2*vsini*(1-(eps/3)))
    fun = 1-((rv_array-vstar)/vsini)**2
    invalid_index = np.where(fun<0)
    fun[invalid_index] = 0
    Z = c1*np.sqrt(fun)+c2*fun
    whereNAN = np.isnan(Z)
    Z[whereNAN]=0
    return(Z)

def rotation_inclination(lat,lon,inclination):
    ''' Euler-Rodrigues (ER) equation for transforming coordinates based on 
        rotation due to inclination.
    '''
    ''' Spherical Coordinates'''
    r = 1
    theta = 90-lat
    phi = lon
    xi = 90-inclination
    ''' Convert into Cartesian'''
    x = r*np.cos(np.radians(phi))*np.sin(np.radians(theta))
    y = r*np.sin(np.radians(phi))*np.sin(np.radians(theta))
    z = r*np.cos(np.radians(theta))
    ''' Compute ER parameters'''
    a = np.cos(np.radians(xi/2))
    b = 0#np.sin(np.radians(xi/2))
    c = np.sin(np.radians(xi/2))
    d = 0#np.sin(np.radians(xi/2))*np.sin(np.radians(xi))
    '''Build coordinate transformation matrices'''
    M1 = [a**2+b**2-c**2-d**2, 2*(b*c-a*d),2*(b*d+a*c)]
    M2 = [2*(b*c+a*d),a**2+c**2-b**2-d**2, 2*(c*d-a*b)]
    M3 = [2*(b*d-a*c),2*(c*d+a*b),a**2+d**2-b**2-c**2]
    M = np.asarray([M1,M2,M3])
    M = M.reshape(3,3)
    X = np.asarray([x,y,z],dtype=object)
    ''' Compute transformed coordinates'''
    Xprime = np.matmul(M,X)
    xprime = Xprime[0]
    yprime = Xprime[1]
    zprime = Xprime[2]
    ''' Convert back into standard lat/lon coordinates'''
    rprime = np.sqrt(xprime**2+yprime**2+zprime**2)
    theta_prime = np.degrees(np.arccos(zprime/rprime))
    lat_prime = 90 - theta_prime
    lon_prime = np.degrees(np.arctan(yprime/xprime))
    return(lat_prime,lon_prime)

def viewing(lat,latspot,lonspot,time_steps,inc):
    ''' Determines if Euler-Rodrigues transformed lat/lon coordinates are
        viewable at each time step.
        
        Inputs:
            lat: Untransformed initial spot latitude
            latspot: Array of transformed latitudes
            lonspot: Array of transformed
            inc: Target inclincation w.r.t. the viewer
    '''
    if lat >= 0 and inc > lat:
        ''' The Euler-Rodrigues transformation creates discontinuities as the 
        spot rotates out of view. The in-view region will be concave up (with a
        local minimum) while the out-of-view region will be concave down (with a
        local maximum).
        '''
        '''ID discontinuites'''
        discon_loc = np.where(np.sign(np.diff(lonspot)) == -1)[0]+1
        ''' Account for wrapping at 0/360 deg.'''
        for i in range(len(discon_loc)):
            if discon_loc[i] == len(lonspot):
                discon_loc[i] =  0
        if len(discon_loc) == 1:
            ''' Find values for each regions' minimum.'''
            region1min = min(latspot[0:int(discon_loc[0])])
            region2min = min(latspot[int(discon_loc[0]):])
            ''' ID out-of-view region and set flux to zero. Out-of-view region
                will have larger minimum value.
            '''
            if region1min > region2min:
                no_flux_index1 = np.arange(0,int(discon_loc[0]),1)
            else: 
                no_flux_index1 = np.arange(int(discon_loc[0]),time_steps,1)
            frontside = np.ones(time_steps)
            frontside[no_flux_index1] = 0
        elif len(discon_loc) == 2:
            ''' Find values for each regions' minimum.'''
            region1min = min(np.append(latspot[0:int(discon_loc[0])],latspot[int(discon_loc[1]):]))
            region2min = min(latspot[int(discon_loc[0]):int(discon_loc[1])]) 
            ''' ID out-of-view region and set flux to zero. Out-of-view region
                will have larger minimum value.
            '''
            if region1min > region2min:
                no_flux_index1 = np.append(np.arange(0,int(discon_loc[0]),1),np.arange(int(discon_loc[1]),time_steps,1))
            else: 
                no_flux_index1 = np.arange(int(discon_loc[0]),int(discon_loc[1]),1)
            frontside = np.ones(time_steps)
            frontside[no_flux_index1] = 0
        else:
            frontside = np.ones(time_steps)
    elif lat >= 0 and inc <= lat:
        ''' If the spot latitude is greater than 0 and is greater than the inclination,
            it will remain visible throughout the entire rotation (the model only 
            considers inclination towards the viewer).
        '''
        frontside = np.ones(time_steps)
    elif lat < 0 and np.abs((inc-90)+lat) <= 90:
        '''ID discontinuites'''
        discon_loc = np.where(np.sign(np.diff(lonspot)) == -1)[0]+1
        ''' Account for wrapping at 0/360 deg.'''
        for i in range(len(discon_loc)):
            if discon_loc[i] == len(lonspot):
                discon_loc[i] =  0
        if len(discon_loc)==1:
            ''' Find values for each regions' minimum.'''
            region1min = min(latspot[0:int(discon_loc[0])])
            region2min = min(latspot[int(discon_loc[0]):])
            ''' ID out-of-view region and set flux to zero. Out-of-view region
                will have larger minimum value.
            '''
            if region1min > region2min:
                no_flux_index1 = np.arange(0,int(discon_loc[0]),1)
            else: 
                no_flux_index1 = np.arange(int(discon_loc[0]),time_steps,1)
            frontside = np.ones(time_steps)
            frontside[no_flux_index1] = 0
        elif len(discon_loc)==0:
            ''' This region is never in view.'''
            frontside = np.zeros(time_steps)
        else:
            ''' Find values for each regions' minimum.'''
            region1min = min(np.append(latspot[0:int(discon_loc[0])],latspot[int(discon_loc[1]):]))
            region2min = min(latspot[int(discon_loc[0]):int(discon_loc[1])]) 
            ''' ID out-of-view region and set flux to zero. Out-of-view region
                will have larger minimum value.
            '''
            if region1min > region2min:
                no_flux_index1 = np.append(np.arange(0,int(discon_loc[0]),1),np.arange(int(discon_loc[1]),time_steps,1))
            else: 
                no_flux_index1 = np.arange(int(discon_loc[0]),int(discon_loc[1]),1)
            frontside = np.ones(time_steps)
            frontside[no_flux_index1] = 0
    elif lat < 0 and np.abs((inc-90)+lat) > 90:
        ''' This region is never in view.'''
        frontside = np.zeros(time_steps)
    return(frontside)

def energy_contained(Rref,seeing):
    ''' Compute the percentage of the PSF within the S/N reference area using 
    Gaussian distribution for PSF.
    
    Inputs:
        Rref: Radius of S/N reference area [mas]
        seeing: Atmospheric seeing [mas]
    '''
    sig = seeing
    radius = Rref
    radius_array = np.linspace(-radius,radius,100) 
    energy_array = np.exp(-0.5 * np.power(radius_array/sig, 2.)) / (np.sqrt(2*np.pi)*sig)
    energy = sum(energy_array)*100
    return(energy)

# Create the Uncertainty Matrix, S
def make_S(temp,m):
    ''' Constructs the S Matrix for LSD.
        Inputs:
        -temp: measurement error (sigma)
        Outputs:
        S: 2D numpy array of S Matrix
    '''
    S = np.zeros((m,m))
    np.fill_diagonal(S, 1/temp)
    return S

############## Anusha Pai Asnodkar's and Ji Wang's Code **************

def make_M(temp, n, filename_M=None, filename_temp=None):
    ''' Constructs the M matrix for LSD.
        Inputs:
        - temp: 1D numpy array of stellar template
        - n: integer of how many points in deconvolved profile
        - filename_M: save M matrix to file
        - filename_temp: save template to file
        Outputs:
        - M: 2D numpy array of M matrix
    '''

    M = np.roll(scipy.linalg.toeplitz(temp)[:, 0:n], int(-n/2), axis = 0) # Ji's
    return M

def make_R(dim):
    ''' Generates regularization matrix.
        Inputs:
        - dim: integer representing size of square regularization matrix
        Outputs:
        - R: 2D numpy array of R matrix
    '''
    R = np.zeros((dim,dim))
    for i in range(dim):
        if i == 0:
            R[i,0:2] = [1, -1]
        elif i == dim-1:
            R[i, -2:] = [-1, 1]
        else:
            R[i,i-1] = -1
            R[i,i] = 2
            R[i,i+1] = -1
    return R

def leastSqDeconvolution(Y, temp, S, n, Lambda, filename_M=None, filename_temp=None, filename_LP=None):
    ''' Performs least-squares deconvolution.
        Inputs:
        - Y: 1-D numpy array observed spectrum
        - temp: 1-D numpy array of observed stellar spectrum
        - n: integer number of velocities in output line profile (must be odd for symmetry)
        - Lambda: float for regularization parameter
        - S: Uncertainty Matrix
        - filename_M: read M matrix from file
        - filename_temp: read template from file
        - filename_LP: save line profile to file
        Outputs:
        - prof: 1-D numpy array of deconvolved line profile
        - M: 2-D numpy array of M matrix (for testing)
    '''
    M = make_M(temp, n, filename_M, filename_temp)
    ACF = np.transpose(M).dot(M) + Lambda*make_R(n) #without S Matrices
    iACF = np.linalg.inv(ACF)
    CCF = np.transpose(M).dot(Y)
    prof = np.dot(iACF, CCF)

    return prof
