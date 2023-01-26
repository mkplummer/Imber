#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 12:41:25 2022

@author: plummer.323
"""
import Imber


'''
Here is a pre-made list of Target-class objects that can be loaded into Imber's 
Observation, Numerical_Model, and Analytical_Model classes.

Inputs:
    vsini: Rotational velocity as a function of inclination [km/s]
    inclination: Orbital axis of target w.r.t. solar system [deg]
    period: Target rotational period [hrs]
    *vstar: Radial velocity of star system w.r.t. solar system [km/s], Default to 0

Observed Properties (Necessary for Observation-class objects)
    spectra_file: Spectrum filename
    temperature: Only used if using the blackbody mode (usually for diagnostics)
    apparent_magnitude: Needs to match the associated magnitude band
    magnitude_band: Band corresponding to supplied apparent magnitude
    zero: Magnitude system zero (either AB or Vega)
    *file_type: File format (normally FITS or CSV), Defaults to FITS
'''

'TRAPPIST 1'
vsini = 2.1 # [km/s]
inclination = 85 # [Deg]
period = 3.3*24 # [hrs] 
spectra_file = '/Users/plummer.323/Imber/BTSettl/lte2600-5.00-0.0a+0.0.BT-settl-giant-2013.cf250.tm1.0-0.0.R1-g12p0vo1.spid.fits'
zero = 'Vega'
apparent_magnitude = 10.296
magnitude_band = 'K'
TRAPPIST1_METIS = Imber.Target(vsini,inclination,period)
TRAPPIST1_METIS.observed_properties(spectra_file,apparent_magnitude,magnitude_band,zero)
apparent_magnitude = 10.718 #11.558
magnitude_band = 'H' 
TRAPPIST1_MODHIS = Imber.Target(vsini,inclination,period)
TRAPPIST1_MODHIS.observed_properties(spectra_file,apparent_magnitude,magnitude_band,zero)
apparent_magnitude = 14.024
magnitude_band = 'I'
TRAPPIST1_GCLEF = Imber.Target(vsini,inclination,period)
TRAPPIST1_GCLEF.observed_properties(spectra_file,apparent_magnitude,magnitude_band,zero)


'VHS 1256b'
vsini = 13.5  # [km/s]
inclination = 54 # [Deg]
period = 22.0 # [hrs]
spectra_file = '/Users/plummer.323/Imber/BTSettl/lte1200-4.50-0.0a+0.0.BT-settl-giant-2013.cf250.tm1.0-0.0.R1-g12p0sc2.spid.fits'
zero = 'Vega'
apparent_magnitude = 13.6
magnitude_band = 'L' 
VHS1256b_METIS = Imber.Target(vsini,inclination,period)
VHS1256b_METIS.observed_properties(spectra_file,apparent_magnitude,magnitude_band,zero)
apparent_magnitude = 15.595
magnitude_band = 'H' 
VHS1256b_MODHIS = Imber.Target(vsini,inclination,period)
VHS1256b_MODHIS.observed_properties(spectra_file,apparent_magnitude,magnitude_band,zero)
apparent_magnitude = 16.662
magnitude_band = 'J'
VHS1256b_GCLEF = Imber.Target(vsini,inclination,period)
VHS1256b_GCLEF.observed_properties(spectra_file,apparent_magnitude,magnitude_band,zero)


'SIMP 0136'
vsini = 52.8  # [km/s] Vos+17
inclination = 80.0 # [Deg] Vos+17 
period = 2.414 # [hrs] Yang+16
spectra_file = '/Users/plummer.323/Imber/BTSettl/lte1200-4.50-0.0a+0.0.BT-settl-giant-2013.cf250.tm1.0-0.0.R1-g12p0sc2.spid.fits'
zero = 'Vega'
apparent_magnitude = 11.94
magnitude_band = 'L' 
SIMP0136_METIS = Imber.Target(vsini,inclination,period)
SIMP0136_METIS.observed_properties(spectra_file,apparent_magnitude,magnitude_band,zero)
apparent_magnitude = 12.809
magnitude_band = 'H' 
SIMP0136_MODHIS = Imber.Target(vsini,inclination,period)
SIMP0136_MODHIS.observed_properties(spectra_file,apparent_magnitude,magnitude_band,zero)
apparent_magnitude = 13.252
magnitude_band = 'J'
SIMP0136_GCLEF = Imber.Target(vsini,inclination,period)
SIMP0136_GCLEF.observed_properties(spectra_file,apparent_magnitude,magnitude_band,zero)

'Beta Pic b'
vsini = 25.0
inclination = 57 # [Deg]
period = 8.1 # [hrs] 
spectra_file = '/Users/plummer.323/Imber/BTSettl/lte1700-4.00-0.0a+0.0.BT-settl-giant-2013.cf250.tm1.0-0.0.R1-g12p0vo1.spid.fits'
zero = 'Vega'
magnitude_band = 'L' 
apparent_magnitude = 11.24
host_magnitude_band = 'K'
host_magnitude = 3.48 # (K band) Ducati,2002
BetaPicb_METIS = Imber.Target(vsini,inclination,period)
BetaPicb_METIS.observed_properties(spectra_file,apparent_magnitude,magnitude_band,zero,host_magnitude_band = host_magnitude_band,host_magnitude = host_magnitude) 
magnitude_band = 'H'
apparent_magnitude = 13.32
host_magnitude_band = 'H'
host_magnitude = 3.48 # (H band) Ducati,2002
BetaPicb_MODHIS = Imber.Target(vsini,inclination,period)
BetaPicb_MODHIS.observed_properties(spectra_file,apparent_magnitude,magnitude_band,zero,host_magnitude_band = host_magnitude_band,host_magnitude = host_magnitude)


'HR 8799 e'
orbit = 'Random' # or 'Aligned'
if orbit == 'Aligned':
    inclination = 24.0 # [Deg]
    period = 4.1 # [hrs]
elif orbit == 'Random':
    inclination = 56.4
    period = 8.4
vsini = 15.0  # [km/s]
spectra_file = '/Users/plummer.323/Imber/BTSettl/lte1300-4.00-0.0a+0.0.BT-settl-giant-2013.cf250.tm1.0-0.0.R1-g12p0sc2.spid.fits'
zero = 'Vega'
magnitude_band = 'L'
apparent_magnitude = 14.57
host_magnitude_band = 'K'
host_magnitude = 5.240
HR8799e_METIS = Imber.Target(vsini,inclination,period)
HR8799e_METIS.observed_properties(spectra_file,apparent_magnitude,magnitude_band,zero,host_magnitude_band = host_magnitude_band,host_magnitude = host_magnitude)
magnitude_band = 'H' 
apparent_magnitude = 16.94
host_magnitude_band = 'H'
host_magnitude = 5.280
HR8799e_MODHIS = Imber.Target(vsini,inclination,period)
HR8799e_MODHIS.observed_properties(spectra_file,apparent_magnitude,magnitude_band,zero,host_magnitude_band = host_magnitude_band,host_magnitude = host_magnitude)

'HR 8799 d'
orbit = 'Random' #'Random' or 'Aligned'
if orbit == 'Aligned':
    inclination = 23.8 # [Deg]
    period = 6.0 # [hrs]
elif orbit == 'Random':
    inclination = 51.0
    period = 12.0
vsini = 10.1
spectra_file = '/Users/plummer.323/Imber/BTSettl/lte1600-5.00-0.0a+0.0.BT-settl-giant-2013.cf250.tm1.0-0.0.R1-g12p0vo1.spid.fits'
zero = 'Vega'
magnitude_band = 'L'
apparent_magnitude = 14.59
host_magnitude_band = 'K'
host_magnitude = 5.240
HR8799d_METIS = Imber.Target(vsini,inclination,period)
HR8799d_METIS.observed_properties(spectra_file,apparent_magnitude,magnitude_band,zero,host_magnitude_band = host_magnitude_band,host_magnitude = host_magnitude)
magnitude_band = 'H' 
apparent_magnitude = 17.29
host_magnitude_band = 'H'
host_magnitude = 5.280
HR8799d_MODHIS = Imber.Target(vsini,inclination,period)
HR8799d_MODHIS.observed_properties(spectra_file,apparent_magnitude,magnitude_band,zero,host_magnitude_band = host_magnitude_band,host_magnitude = host_magnitude)
