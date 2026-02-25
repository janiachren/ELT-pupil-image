## Compiling
cc -fPIC -shared -std=c99 -O2 micado_elt_pupil_status.c eelt_pupil.c $(pkg-config --cflags cpl libxml-2.0) $(pkg-config --libs cpl libxml-2.0) -lcfitsio -lm -o micado_elt_pupil_status.so


## Running via esorex
esorex micado_elt_pupil_status --xml_file=StatusM1segments.xml

