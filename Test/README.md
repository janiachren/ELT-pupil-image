Regression test HOWTO

Python script for running a regression test between FITS images created by ELTpupil_statusReporting.py and ./elt_pupil_status. Tested with v1.0.

Usage:
1. Copy the following files to the same directory:
   - run_regression.py (test main)
   - ELTpupil_sttusReporting.py (python main)
   - anisocado_pupUtils.py (python libs)
   - elt_pupil_status (C executable)
2. Adjust test parameters inside run_regression.py
     - create N xml files with randomized dates for last coating (default N=1000)
     - specifies relative and absolute tolerances for errors stemming from float32/64 differences, CFITSIO read/writes, compiler  reorder operations, trig function differences between glibc, libm, python maths, etc. (defaults RTOL = 1e-5, ATOL = 2e-4)
3. Start test with
   <code>python3 run_regression.py</code>
4. Script creates N xml files to a temporary directory, creates FITS files with N_0 and comapares, deletes FITS files, continues with same with N_1, and repeats to N.
5. Results are saved as "regression_log.txt"
   
