import re
import numpy as np
import matplotlib.pyplot as plt
#from anisocado_pupil_Utils import generateEeltPupilReflectivity
import anisocado_pupUtils
from astropy.io import fits
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

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
pupil_mask = anisocado_pupUtils.generateEeltPupilReflectivity(
    F1, grid_size, spider_width,
    410.5, 410.5,
    pixscale, gap, 0.0,
    EeltDiam, softGap=False
)

# Show pupil mask
plt.imshow(pupil_mask, cmap="gray", origin="lower")
plt.title("ELT M1 Pupil Mask - Current Status")
plt.colorbar()
plt.show()

# Get ESO-style UTC timestamp
timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

# Filename with timestamp
fits_filename = f"ELTpupil_status_current_{timestamp}.fits"

# Create FITS HDU and add timestamp to header
hdu = fits.PrimaryHDU(pupil_mask)
hdu.header['DATE-OBS'] = timestamp   # ESO standard keyword
hdu.header['ORIGIN'] = 'ELT pupil status reporting script'
hdu.header['COMMENT'] = 'Mirror segment reflectivity and operational status snapshot'

# Write to FITS file
hdu.writeto(fits_filename, overwrite=True)
print(f"Saved FITS file: {fits_filename}")
