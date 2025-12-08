Python code for the creation of pupil image of ESO ELT main mirror M1.

This script uses Anisocado libraries (included as anisocado_pupUtils.py by Eric Gendron) to create and return an
image of the ELT M1. The image contains positions of the individual segments and the support beams of the M2.
It then reviews the operational status and the coating performance M1 segments reported in file StatusM1segments.xml
and adjusts the reflectivity levels of each segment. The result is output as a FITS file.

For the testing purposes, the data contained in StatusM1segments.xml is arbitrary.
