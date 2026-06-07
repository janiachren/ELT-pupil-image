## Compiling
## CPL 7.x, no esorex, no recipes

PKG_CONFIG_PATH=/opt/cpl-7.3.2/lib/pkgconfig \
gcc micado_elt_pupil_status.c eelt_pupil.c \
    -o micado_elt_pupil_status \
    $(pkg-config --cflags --libs cpl libxml-2.0) \
    -lm



## Running
./micado_elt_pupil_status StatusM1segments.xml






