import os
import shutil
import subprocess
from astropy.io import fits
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import random
import xml.etree.ElementTree as ET

# ------------------------------------------------------------
# Configuration. You may edit these entries.
# ------------------------------------------------------------

PYTHON_SCRIPT = "./ELTpupil_statusReporting.py"
C_EXECUTABLE  = "./elt_pupil_status"
LOGFILE       = "regression_log.txt"

XML_DIR = "xml_randomized"             # Where randomized XMLs go
NUM_XML = 1000                        # How many randomized XMLs to generate

# Tolerances for errors arising from i.e. float32/64 diffs, CFITSIO
# read/writes, compiler reorder operations, trig function differences
# between glibc, libm and python's math, etc.
RTOL = 1e-5
ATOL = 2e-4


# ------------------------------------------------------------
# Please do not edit anything below this line.
# ------------------------------------------------------------



# ------------------------------------------------------------
# Random date helper
# ------------------------------------------------------------
def random_date():
    """Return a random date within the last 4 years."""
    end = datetime.now()
    start = end - timedelta(days=4*365)
    delta = end - start
    random_days = random.randrange(delta.days)
    return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")

# ------------------------------------------------------------
# Create randomized XML files
# ------------------------------------------------------------
def generate_randomized_xml_files():
    # Clean or create directory
    if os.path.exists(XML_DIR):
        shutil.rmtree(XML_DIR)
    os.makedirs(XML_DIR)

    NUM_SEGMENTS = 798  # ELT M1 segment count

    for i in range(NUM_XML):
        # Build XML structure
        root = ET.Element("segments")

        for seg_id in range(1, NUM_SEGMENTS + 1):
            seg = ET.SubElement(root, "segment")
            seg.set("id", str(seg_id))
            seg.set("operational", "true")
            seg.set("last_recoating", random_date())

        # Write file
        fname = f"test_{i:04d}.xml"
        tree = ET.ElementTree(root)
        tree.write(os.path.join(XML_DIR, fname), encoding="utf-8", xml_declaration=True)

# ------------------------------------------------------------
# FITS detection by prefix
# ------------------------------------------------------------
def find_newest_fits(prefix):
    fits_files = [f for f in os.listdir(".") if f.startswith(prefix) and f.endswith(".fits")]
    if not fits_files:
        return None
    fits_files.sort(key=lambda f: os.path.getmtime(f))
    return fits_files[-1]

# ------------------------------------------------------------
# FITS comparison
# ------------------------------------------------------------
def compare_fits(py_file, c_file):
    py = fits.getdata(py_file)
    c  = fits.getdata(c_file)

    if py.shape != c.shape:
        return False, f"Shape mismatch: py={py.shape}, c={c.shape}"

    if np.allclose(py, c, rtol=RTOL, atol=ATOL):
        return True, f"Match within tolerances (rtol={RTOL}, atol={ATOL})"

    diff = np.abs(py - c)
    return False, f"Mismatch: max diff={diff.max()}, mean diff={diff.mean()}"

# ------------------------------------------------------------
# Main regression loop
# ------------------------------------------------------------
def main():
    # Step 1: Generate randomized XML files
    print("Generating randomized XML files...")
    generate_randomized_xml_files()

    xml_files = sorted(os.listdir(XML_DIR))

    with open(LOGFILE, "w") as log:
        log.write(f"Regression test started {datetime.now()}\n")
        log.write(f"Generated {len(xml_files)} randomized XML files\n\n")

        passed = 0
        failed = 0

        for xml in tqdm(xml_files, desc="Testing"):
            xml_path = os.path.join(XML_DIR, xml)
            log.write(f"--- Testing {xml} ---\n")

            # Clean leftover FITS
            for f in os.listdir("."):
                if f.endswith(".fits"):
                    try: os.remove(f)
                    except: pass

            # Run Python generator
            r1 = subprocess.run(
                ["python3", PYTHON_SCRIPT, xml_path],
                capture_output=True, text=True
            )
            if r1.returncode != 0:
                log.write(f"Python generator failed:\n{r1.stderr}\n\n")
                failed += 1
                continue

            py_fits = find_newest_fits("p.")
            if py_fits is None:
                log.write("Python generator produced no FITS file\n\n")
                failed += 1
                continue

            # Run C generator
            r2 = subprocess.run(
                [C_EXECUTABLE, xml_path],
                capture_output=True, text=True
            )
            if r2.returncode != 0:
                log.write(f"C generator failed:\n{r2.stderr}\n\n")
                failed += 1
                try: os.remove(py_fits)
                except: pass
                continue

            c_fits = find_newest_fits("c.")
            if c_fits is None:
                log.write("C generator produced no FITS file\n\n")
                failed += 1
                try: os.remove(py_fits)
                except: pass
                continue

            # Compare FITS
            ok, msg = compare_fits(py_fits, c_fits)
            log.write(msg + "\n")

            if ok:
                passed += 1
            else:
                failed += 1

            # Delete FITS files immediately
            try: os.remove(py_fits)
            except: pass
            try: os.remove(c_fits)
            except: pass

            time.sleep(1.2)
            log.write("\n")

        log.write(f"Finished {datetime.now()}\n")
        log.write(f"PASSED: {passed}\n")
        log.write(f"FAILED: {failed}\n")

    print(f"Done. See {LOGFILE} for results.")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
