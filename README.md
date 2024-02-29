# Imber
Doppler imaging and light curve inverstion tool created by [Michael K. Plummer](https://www.michaelplummer.dev) for modeling stellar/substellar surfaces. The Python module simulates spectroscopic and photometric observations with both a gridded, numerical simulation and analytical model. Imber has been specifically designed to predict Extremely Large Telescope instrument (e.g. ELT/METIS and TMT/MODHIS) Doppler imaging performance. It has also been applied to existing, archival observations of spectroscopy and photometry to model surface features on Luhman 16B, a nearby L/T transition brown dwarf.

Version 1.0 is oriented for Doppler imaging performance testing as demonstrated in Plummer & Wang (2023).

Version 3.0 is optimized for lightcurve inversion. It adds multi-rotational spot evolution (temperature contrast and size) and also incorporates wave models for photometric variability. 

## References

- [Plummer & Wang (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...933..163P/abstract) A Unified Spectroscopic and Photometric Model to Infer Surface Inhomogeneity: Application to Luhman 16B, The Astrophysical Journal, Volume 933, Issue 2, id.163, 17 pp., July 2022.

- [Plummer & Wang (2023)](https://ui.adsabs.harvard.edu/abs/2023arXiv230408518P/abstract) Mapping the Skies of Ultracool Worlds: Detecting Storms and Spots with Extremely Large Telescopes, Accepted for publication in the Astrophysical Journal.

## Dependencies

AstroPy, Dynesty, Matplotlib, Numpy, Pandas, SciPy, SecretColors, SpecUtils, TQDM

