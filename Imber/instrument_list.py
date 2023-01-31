#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 11:25:47 2022

@author: plummer.323
"""

import Imber


'''
------------------------------------------------------------
GMT-GCLEF
'''
'Telescope Parameters'
aperture_diameter = 24.5 # Diameter [m]
collecting_area = 368 # [m^2]

'Sky Emission and Transmission Files'
sky_file = '/Users/plummer.323/Imber/SkyModels/skytable-GCLEF-pwv3.7-T283.fits' #Use the SkyTable from ESO for Transmission & Sky Background
file_type = 'FITS'

'Instrument Parameters'
name = 'GCLEF-Red'
spectral_resolution = 105E3
efficiency = 0.11
Nspec = 1
lambda_ref = 0.745 # [microns]
dark_current = 0.00056 # [electrons/pixel]
readout_noise = 2  # [electrons]

GCLEF = Imber.Spectroscopic_Instrument(aperture_diameter,collecting_area,\
                                          sky_file,file_type,name,spectral_resolution,\
                                              efficiency,Nspec,lambda_ref,dark_current,
                                              readout_noise)

    
'''
-------------------------------------------------------------
TMT-MODHIS
'''

'Telescope Parameters'
aperture_diameter = 30 # [m]
collecting_area = 651 # [m^2]

'Sky Emission and Transmission Files'
sky_file = '/Users/plummer.323/Imber/SkyModels/MKO_sky_emiss_NIR_1mm_1.5.csv'
transmission_file = '/Users/plummer.323/Imber/SkyModels/MKO_atm_trans_NIR_1.0mm_1.5.csv'
file_type = 'CSV'

'Instrument Parameters'
name = 'MODHIS'
spectral_resolution = 100E3
efficiency = 0.1 # MODHIS Website based on throughput
Nspec = 1
lambda_ref = 1.675
dark_current = 0.02 # [electrons/pixel] Based on ELT values for NIR
readout_noise = 7 # [electrons] Based on ELT values for NIR
starlight_suppression = 1E-4 #Mawet

MODHIS = Imber.Spectroscopic_Instrument(aperture_diameter,collecting_area,\
                                          sky_file,file_type,name,spectral_resolution,\
                                              efficiency,Nspec,lambda_ref,dark_current,\
                                              readout_noise,\
                                                  transmission_file = transmission_file,\
                                                  starlight_suppression = starlight_suppression)

'''
------------------------------------------------------------
ELT-METIS
'''
'Telescope Parameters'
aperture_diameter = 39 # Diameter [m]
collecting_area = 978 # [m^2]

'Sky Emission and Transmission Files'
sky_file = '/Users/plummer.323/Imber/SkyModels/skytable-ELT-L.fits' #Uses the SkyTable from ESO for Transmission & Sky Background
file_type = 'FITS'

'Instrument Parameters'
name = 'METIS'
spectral_resolution = 100E3
efficiency = 0.25
Nspec = 1
lambda_ref = 3.8 # [microns]
dark_current = 0.5555556 # [electrons/pixel]
readout_noise = 200 # [electrons]
starlight_suppression = 1E-4 #Brandl

METIS = Imber.Spectroscopic_Instrument(aperture_diameter,collecting_area,\
                                          sky_file,file_type,name,spectral_resolution,\
                                              efficiency,Nspec,lambda_ref,dark_current,
                                              readout_noise,\
                                              starlight_suppression = starlight_suppression)

'''
------------------------------------------------------------
ELT-Imager
'''

'Telescope Parameters'
aperture_diameter = 39 # [m]
collecting_area = 978 # [m^2]

'Sky Emission and Transmission Files'
sky_file = '/Users/plummer.323/Imber/SkyModels/MKO_sky_emiss_NIR_1mm_1.5.csv'
file_type = 'FITS'
airmass = 1.5

'Instrument Parameters'
name = 'ELTimager'
instrument_band = 'L'
central_wavelength_instrument = 3.79 # [microns]
instrument_band_range = 0.63 # [microns]
efficiency = 0.5 # Ratio of detected photons to # incident on telescope aperature (value from ELT ETC)
site = 'Paranal-like'
'Number of Pixels in S/N reference area'
Npix = 100 # Eiether 1, 9, or 25 corresponding to 1x1, 3x3, and 5,5
''' Encircled energy is computed using Table 7 from ELT Photometric ETC.
    Linear size of S/N area is cross-referenced w/ Spectral Band to find the
    proportion of energy (of the PSF) enclosed in the S/N reference area.
'''
encircled_energy = 0.607 
dark_current = 2000 # [electrons/s/pixel]
readout_noise = 200 # [electrons]
pixel_scale = 5 # [mas] Metis should be 5.5 mas [L/M band] but ETC only has 5 as option
starlight_suppression = 1E-4

METISimager = Imber.Photometric_Instrument(aperture_diameter,collecting_area,\
                                        sky_file,file_type,airmass,\
                                            name,instrument_band,\
                                                central_wavelength_instrument,\
                                                    instrument_band_range,\
                                                        efficiency, site, Npix,\
                                                            encircled_energy,\
                                                                dark_current,\
                                                                    readout_noise,pixel_scale,\
                                                                    starlight_suppression = starlight_suppression)
