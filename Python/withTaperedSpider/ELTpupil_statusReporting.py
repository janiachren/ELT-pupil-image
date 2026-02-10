import re
import numpy as np
#import matplotlib.pyplot as plt
import anisocado_pupUtils_varDspider
from astropy.io import fits
import xml.etree.ElementTree as ET

from datetime import datetime, timezone
from astropy.time import Time

import warnings
# Suppress ERFA "dubious year" warnings
warnings.filterwarnings('ignore', message='.*dubious year.*')
# Suppress FITS VerifyWarnings for long ESO keywords
warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)


# ======= PARAMETERS =======
grid_size = 822
num_segments = 798
EeltDiam = 40.0 # 38.542
pixscale = EeltDiam / grid_size  * 1.0275
spider_width = 0.54 # from ESO-532755
gap = 0.0 # 0.002 from M. Cayrel et al, "E-ELT Optomechanics: Overview"
 

# Reflectivity constants
# Source: Schneider, "Silver Coating on large Telescope Mirrors Tutorial", OPTI521.
# Values approximated from Gemini South M1 coating data (in-situ washing effects ignored)
max_reflectivity = 0.96
min_reflectivity = 0.91
coating_degradation_per_day = 0.000125


# ======= XML LOADER =======
def load_segments_from_file(filename, num_segments):
    """
    Load segment reflectivities from a file.
    The file may contain only XML, or arbitrary text around a <segments>...</segments> block.
    """
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract only the <segments>...</segments> block
    match = re.search(r"<segments[\s\S]*?</segments>", content)
    if not match:
        raise ValueError("No <segments> block found in file!")

    xml_block = match.group(0)
    root = ET.fromstring(xml_block)

    today = datetime.today()
    F1 = np.zeros(num_segments)

    for seg in root.findall("segment"):
        seg_id_str = seg.get("id").strip().lstrip("\ufeff")  # clean BOM/whitespace
        seg_id = int(seg_id_str) - 1  # XML uses 1-based indexing

        operational = seg.get("operational").lower() == "true"
        last_recoating = datetime.strptime(seg.get("last_recoating"), "%Y-%m-%d")

        if operational:
            days_since = (today - last_recoating).days
            reflectivity = max_reflectivity - coating_degradation_per_day * days_since
            reflectivity = max(reflectivity, min_reflectivity)
        else:
            reflectivity = 0.0

        F1[seg_id] = reflectivity

    return F1



# ======= MAIN EXECUTION =======
xml_file = "StatusM1segments.xml"
# Can also be replaced by any TXT format file that contains the XML tag
# <segments> and contents from the original StatusM1segments.xml.
# Everything outside the first instance of <segments> are ignored.

F1 = load_segments_from_file(xml_file, num_segments)

# Generate pupil mask from today’s reflectivities
pupil_mask = anisocado_pupUtils_varDspider.generateEeltPupilReflectivity( #JA add. (varDspider))
    F1, grid_size, spider_width,
    410.5, 410.5,
    pixscale, gap, 0.0,
    EeltDiam, softGap=False
)

# Show pupil mask during console execution, optional
#plt.imshow(pupil_mask, cmap="gray", origin="lower")
#plt.title("ELT M1 Pupil Mask - Current Status")
#plt.colorbar()
#plt.show()

# Single UTC datetime
dt = datetime.now(timezone.utc)
t = Time(dt, format='datetime', scale='utc')

# Filename string
timestamp = dt.strftime("%Y-%m-%dT%H_%M_%S")
fits_filename = f"c.ELT.{timestamp}.pupil.segmentstatus.fits"

# FITS header numeric MJD
mjd_obs = Time(dt, format='datetime', scale='utc').mjd

# Create FITS HDU
hdu = fits.PrimaryHDU(pupil_mask)
hdr = hdu.header

# Standard FITS keywords
hdr['BITPIX']  = -64
hdr['NAXIS']   = 2
hdr['NAXIS1']  = grid_size
hdr['NAXIS2']  = grid_size
hdr['EXTEND']  = True

# ESO-specific keywords
hdr['INSTRUME']     = 'ELT'
hdr['MJD-OBS']      = t.mjd
hdr['DATE-OBS']     = t.isot
hdr['ESO DPR CATG'] = 'CALIB'
hdr['ESO DPR TYPE'] = 'PUPIL'
hdr['ESO DPR TECH'] = 'IMAGE'
hdr['ESO DET CHIP'] = 'PUPIL'
hdr['ESO INS OPTI'] = 'PRIMARY'
hdr['ESO INS MODE'] = 'SIMULATION'

# Metadata
hdr['ORIGIN']  = 'ESO'
hdr['OBJECT']  = 'ELT PUPIL REFLECTIVITY'
hdr['COMMENT'] = 'Mirror segment reflectivity and operational status snapshot'

# Save file
hdu.writeto(fits_filename, overwrite=True)

#print(f"Saved FITS file: {fits_filename}")
